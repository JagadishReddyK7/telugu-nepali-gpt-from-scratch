"""
One command that produces every Phase 3 evaluation artifact.

    python scripts/run_phase3.py --langs telugu nepali

For each language it performs, in order:

  1. reasoning test-set evaluation of the PRETRAINED checkpoint
     (exact match + scored-choice accuracy)               common.reasoning_eval
  2. the same for the FINETUNED checkpoint
  3. language-modelling evaluation of the finetuned checkpoint on the Phase 1
     validation / test binaries -- did finetuning cost general LM ability?
     (skipped if the binaries are not present)            common.evaluate
  4. pretrained-vs-finetuned attention comparison on the same reasoning
     prompts, with figures                                 common.attention_compare

and then writes ``report/phase3/stats/comparison.md`` with the side-by-side
tables, plus the finetuning curves via ``scripts/plot_finetune.py``.

Every stage writes its own JSON and is skipped when that JSON exists unless
``--force`` is passed, so a Kaggle session that dies part-way costs nothing
already computed, and ``--tables-only`` rebuilds the Markdown from JSON on a
laptop with no GPU and no checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

LABELS = {"telugu": "Model H — Telugu", "nepali": "Model L — Nepali"}
TASK_ORDER = ["superlative_chain", "numeric_compare", "chain_relation",
              "yes_no", "ordering"]


def _load(stats_dir: Path, name: str) -> dict | None:
    """Read a stats JSON if it exists."""
    p = stats_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stage_reasoning(lang: str, tag: str, config: dict, ckpt: Path, device: str,
                    stats_dir: Path, force: bool) -> None:
    """Reasoning test-set evaluation for one checkpoint."""
    import sentencepiece as spm

    from common.evaluate import load_model_from_checkpoint
    from common.reasoning_eval import (evaluate_reasoning, make_ctx,
                                       write_examples_md)
    from common.reasoning_gen import read_jsonl

    name = f"{lang}_{tag}"
    out = stats_dir / f"{name}_reasoning_test.json"
    if out.exists() and not force:
        logger.info("[%s] reasoning/%s  cached", lang, tag)
        return
    model, ck = load_model_from_checkpoint(ckpt, device)
    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])
    ctx = make_ctx(device, config["train"].get("dtype", "float16"))
    examples = read_jsonl(Path(config["data"]["test"]))
    t0 = time.time()
    res = evaluate_reasoning(model, sp, examples, device, ctx)
    res.update({"name": name, "language": lang, "checkpoint": str(ckpt),
                "checkpoint_step": ck.get("step"), "split": "test",
                "eval_seconds": round(time.time() - t0, 1)})
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_examples_md(res, stats_dir / f"{name}_reasoning_examples.md",
                      f"Reasoning examples — {LABELS.get(lang, lang)}, {tag}")
    logger.info("[%s] reasoning/%-10s exact match %.4f | choice %.4f (chance %.4f) "
                "(%.0fs)", lang, tag, res["exact_match"], res["choice_accuracy"],
                res["choice_chance"], res["eval_seconds"])


def stage_lm(lang: str, config: dict, ckpt: Path, device: str,
             stats_dir: Path, force: bool) -> None:
    """LM cross-entropy / perplexity / BPB of the finetuned model."""
    from common.evaluate import evaluate_split

    for split in ("validation", "test"):
        out = stats_dir / f"{lang}_finetuned_eval_{split}.json"
        if out.exists() and not force:
            logger.info("[%s] lm/%s  cached", lang, split)
            continue
        bin_path = Path(config["paths"].get(f"{'val' if split == 'validation' else split}_bin", ""))
        if not bin_path.exists():
            logger.info("[%s] lm/%s  skipped (%s not present)", lang, split, bin_path)
            continue
        t0 = time.time()
        res = evaluate_split(config, ckpt, split, device, max_batches=None)
        res["run"] = f"{lang}_finetuned"
        res["eval_seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        logger.info("[%s] lm/%-10s loss %.5f  ppl %.2f  bpb %s  (%.0fs)", lang,
                    split, res["cross_entropy_nats"], res["perplexity"],
                    res["bits_per_byte"], res["eval_seconds"])


def stage_attention(lang: str, config: dict, pre: Path, ft: Path, device: str,
                    stats_dir: Path, fig_dir: Path, force: bool) -> None:
    """Pretrained-vs-finetuned attention comparison on reasoning prompts."""
    from common.attention_compare import compare

    out = stats_dir / f"{lang}_attention_compare.json"
    if out.exists() and not force:
        logger.info("[%s] attention  cached", lang)
        return
    s = compare(lang, config, pre, ft, device, stats_dir, fig_dir)
    logger.info("[%s] attention  labels pre %s -> ft %s", lang,
                s["head_labels"]["pretrained"], s["head_labels"]["finetuned"])


# --------------------------------------------------------------------- tables

def build_tables(langs: list[str], stats_dir: Path, out_path: Path) -> None:
    """Assemble the Phase 3 side-by-side tables."""
    L = ["# Phase 3 — evaluation tables", "",
         "Generated by `scripts/run_phase3.py`. Reasoning accuracies are on the",
         "full synthetic **test** split (held-out entity names; half the",
         "examples also use held-out phrasings).", ""]

    # 1. headline
    L += ["## 1. Reasoning accuracy, pretrained vs finetuned", "",
          "Exact match: greedy decoding after the answer prefix, compared to the",
          "gold string. Scored choice: the candidate with the highest",
          "log-likelihood among the example's options (chance shown).", "",
          "| | " + " | ".join(f"{LABELS[l]} pretrained | {LABELS[l]} finetuned" for l in langs) + " |",
          "|---|" + "---:|---:|" * len(langs)]
    rows = [("exact match", "exact_match", "{:.4f}"),
            ("prediction is a valid option", "prediction_is_a_valid_choice", "{:.4f}"),
            ("scored-choice accuracy", "choice_accuracy", "{:.4f}"),
            ("scored-choice chance", "choice_chance", "{:.4f}")]
    data = {(l, t): _load(stats_dir, f"{l}_{t}_reasoning_test.json")
            for l in langs for t in ("pretrained", "finetuned")}
    for label, key, fmt in rows:
        cells = []
        for l in langs:
            for t in ("pretrained", "finetuned"):
                v = (data[(l, t)] or {}).get(key)
                cells.append(fmt.format(v) if v is not None else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    # 2. by task
    for metric, title in (("exact_match_by_task", "Exact match by task"),
                          ("choice_accuracy_by_task", "Scored-choice accuracy by task")):
        L += [f"## 2. {title}", "",
              "| task | " + " | ".join(f"{LABELS[l]} pre | {LABELS[l]} ft" for l in langs) + " |",
              "|---|" + "---:|---:|" * len(langs)]
        for task in TASK_ORDER:
            cells = []
            for l in langs:
                for t in ("pretrained", "finetuned"):
                    d = (data[(l, t)] or {}).get(metric, {}).get(task)
                    cells.append(f"{d['accuracy']:.4f}" if d else "—")
            L.append(f"| {task} | " + " | ".join(cells) + " |")
        if metric == "choice_accuracy_by_task":
            cells = []
            for l in langs:
                d = (data[(l, "finetuned")] or {}).get("choice_chance_by_task", {})
                cells.append(" | ".join(f"{d.get(task, 0):.2f}" for task in TASK_ORDER))
            L.append("")
            L.append("Chance per task (" + ", ".join(TASK_ORDER) + "): " +
                     "; ".join(f"{LABELS[l]}: {c}" for l, c in zip(langs, cells)))
        L.append("")

    # 3. generalisation slices (finetuned)
    L += ["## 3. Finetuned exact match by generalisation slice", "",
          "| slice | " + " | ".join(LABELS[l] for l in langs) + " |",
          "|---|" + "---:|" * len(langs)]
    slices = [("seen phrasing, held-out names", "exact_match_by_template_held_out", "False"),
              ("held-out phrasing, held-out names", "exact_match_by_template_held_out", "True"),
              ("1 hop", "exact_match_by_hops", "1"),
              ("2 hops", "exact_match_by_hops", "2"),
              ("3 hops", "exact_match_by_hops", "3"),
              ("2 entities", "exact_match_by_n_entities", "2"),
              ("3 entities", "exact_match_by_n_entities", "3"),
              ("4 entities", "exact_match_by_n_entities", "4")]
    for label, key, sub in slices:
        cells = []
        for l in langs:
            d = (data[(l, "finetuned")] or {}).get(key, {}).get(sub)
            cells.append(f"{d['accuracy']:.4f} (n={d['n']})" if d else "—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    # 4. LM forgetting
    lm = {(l, s): _load(stats_dir, f"{l}_finetuned_eval_{s}.json")
          for l in langs for s in ("validation", "test")}
    pre_lm = {(l, s): _load(stats_dir.parent.parent / "phase2" / "stats",
                            f"{l}_eval_{s}.json")
              for l in langs for s in ("validation", "test")}
    if any(lm.values()):
        L += ["## 4. Language modelling after finetuning (forgetting check)", "",
              "Cross-entropy / perplexity / bits-per-byte of the *finetuned*",
              "checkpoint on the Phase 1 held-out corpus, next to the Phase 2",
              "pretrained numbers.", "",
              "| split · metric | " + " | ".join(f"{LABELS[l]} pre | {LABELS[l]} ft" for l in langs) + " |",
              "|---|" + "---:|---:|" * len(langs)]
        for s in ("validation", "test"):
            for label, key, fmt in (("cross-entropy", "cross_entropy_nats", "{:.4f}"),
                                    ("perplexity", "perplexity", "{:.2f}"),
                                    ("bits per byte", "bits_per_byte", "{:.4f}")):
                cells = []
                for l in langs:
                    a = (pre_lm[(l, s)] or {}).get(key)
                    b = (lm[(l, s)] or {}).get(key)
                    cells.append(fmt.format(a) if a is not None else "—")
                    cells.append(fmt.format(b) if b is not None else "—")
                L.append(f"| {s} · {label} | " + " | ".join(cells) + " |")
        L.append("")

    # 5. attention
    L += ["## 5. Attention on reasoning prompts, pretrained vs finetuned", ""]
    for l in langs:
        a = _load(stats_dir, f"{l}_attention_compare.json")
        if not a:
            continue
        L += [f"### {LABELS[l]} — head types: pretrained {a['head_labels']['pretrained']} "
              f"→ finetuned {a['head_labels']['finetuned']}", "",
              "| layer | entropy pre | entropy ft | distance pre | distance ft | "
              "prev-tok pre | prev-tok ft | entity mass pre | entity mass ft | "
              "gold-entity mass pre | gold-entity mass ft |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        p, f = a["per_layer"]["pretrained"], a["per_layer"]["finetuned"]
        ep, ef = a["entity_attention"]["pretrained"], a["entity_attention"]["finetuned"]
        for i in range(len(p["entropy_normalized"])):
            L.append(f"| {i} | {p['entropy_normalized'][i]:.3f} | {f['entropy_normalized'][i]:.3f} | "
                     f"{p['distance_tokens'][i]:.2f} | {f['distance_tokens'][i]:.2f} | "
                     f"{p['previous_token_mass'][i]:.3f} | {f['previous_token_mass'][i]:.3f} | "
                     f"{ep['entity_per_layer'][i]:.3f} | {ef['entity_per_layer'][i]:.3f} | "
                     f"{ep['gold_per_layer'][i]:.3f} | {ef['gold_per_layer'][i]:.3f} |")
        L.append("")
        L.append(f"Entity tokens are {ep['entity_token_fraction']:.3f} of the sequence "
                 f"(uniform-attention baseline for the entity-mass columns).")
        L.append("")

    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    """CLI: run every stage for every language, then tabulate and plot."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=["telugu", "nepali"])
    p.add_argument("--ckpt-root", type=Path, default=Path("checkpoints"),
                   help="Holds <lang>_reasoning/best.pt (finetuned).")
    p.add_argument("--pretrained-root", type=Path, default=None,
                   help="Holds <lang>/best.pt (pretrained). Defaults to "
                        "--ckpt-root; on Kaggle the pretrained checkpoints "
                        "live read-only under /kaggle/input.")
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase3/stats"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase3/figures"))
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["pretrained", "finetuned", "lm", "attention"])
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.tables_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device: %s", device)
        for lang in args.langs:
            cfg = yaml.safe_load(Path(f"{lang}/configs/finetune.yaml").read_text(encoding="utf-8"))
            pre = (args.pretrained_root or args.ckpt_root) / lang / "best.pt"
            ft = args.ckpt_root / f"{lang}_reasoning" / "best.pt"
            for c in (pre, ft):
                if not c.exists():
                    raise SystemExit(f"[{lang}] checkpoint not found: {c}")
            # The finetune YAML carries no architecture block by design (it is
            # read from init_from); the Phase 2 tools need config["model"], so
            # take it from the finetuned checkpoint, which is self-describing.
            ft_ck = torch.load(ft, map_location="cpu", weights_only=False)
            cfg["model"] = ft_ck["config"]["model"]
            logger.info("\n=== %s  (finetuned step %s, from %s) ===", lang,
                        ft_ck.get("step"), ft_ck["config"].get("init_from"))
            del ft_ck
            if "pretrained" not in args.skip:
                stage_reasoning(lang, "pretrained", cfg, pre, device, args.stats_dir, args.force)
            if "finetuned" not in args.skip:
                stage_reasoning(lang, "finetuned", cfg, ft, device, args.stats_dir, args.force)
            if "lm" not in args.skip:
                stage_lm(lang, cfg, ft, device, args.stats_dir, args.force)
            if "attention" not in args.skip:
                stage_attention(lang, cfg, pre, ft, device, args.stats_dir,
                                args.fig_dir, args.force)

    print()
    build_tables(args.langs, args.stats_dir, args.stats_dir / "comparison.md")
    try:
        from scripts.plot_finetune import plot_all
        plot_all(args.langs, Path("report/phase3/logs"), args.stats_dir, args.fig_dir)
    except Exception as e:                                   # noqa: BLE001
        logger.warning("plotting skipped: %s", e)
    print()


if __name__ == "__main__":
    main()

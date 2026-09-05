"""
One command that produces every Phase 2 evaluation artifact.

    python scripts/run_eval.py --runs telugu nepali telugu_nopos

For each run it performs, in order:

  1. full-split cross-entropy / perplexity / BPB on validation AND test
     (``common.evaluate``, no batch cap, so the number is the whole split)
  2. generation under greedy and temperatures 0.5 / 1.0 / 1.5, scored with
     BLEU-4, chrF++, ROUGE-L, repetition rate and Distinct-1/2
     (``common.generate``)
  3. attention heatmaps, per-head entropy, mean attention distance
     (``common.attention``)

and then writes ``report/phase2/stats/comparison.md`` -- the side-by-side
tables the report needs, with Model H and Model L in the same rows so the
resource-tier comparison is legible at a glance.

WHY EVERY STEP IS RESUMABLE
---------------------------
Each stage writes its own JSON and is skipped when that JSON already exists,
unless ``--force`` is passed. A Kaggle session that dies during generation
therefore does not cost the evaluation that already succeeded, and the tables
can be regenerated from JSON alone with ``--tables-only`` at any time --
including on a laptop with no GPU and no checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

CONFIGS = {
    "telugu":       Path("telugu/configs/model.yaml"),
    "nepali":       Path("nepali/configs/model.yaml"),
    "telugu_nopos": Path("telugu/configs/model_nopos.yaml"),
}
LABELS = {
    "telugu":       "Model H — Telugu",
    "nepali":       "Model L — Nepali",
    "telugu_nopos": "Telugu ablation — no positional embeddings",
}


def load_config(run: str, ckpt_root: Path) -> tuple[dict, Path]:
    """Load a run's config and locate its checkpoint.

    Args:
        run: Run name, a key of :data:`CONFIGS`.
        ckpt_root: Directory holding ``<run>/best.pt``.

    Returns:
        ``(config, checkpoint_path)``.

    Raises:
        SystemExit: If the config or the checkpoint is missing.
    """
    cfg_path = CONFIGS.get(run)
    if cfg_path is None or not cfg_path.exists():
        raise SystemExit(f"[{run}] config not found: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ckpt = ckpt_root / run / "best.pt"
    if not ckpt.exists():
        ckpt = ckpt_root / run / "latest.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"[{run}] no checkpoint at {ckpt_root / run}/best.pt or latest.pt")
    return config, ckpt


def stage_eval(run: str, config: dict, ckpt: Path, device: str,
               stats_dir: Path, force: bool) -> None:
    """Full-split intrinsic evaluation on validation and test.

    Args:
        run: Run name.
        config: Parsed config.
        ckpt: Checkpoint path.
        device: Device string.
        stats_dir: Where the JSON goes.
        force: Recompute even if the output exists.
    """
    from common.evaluate import evaluate_split

    for split in ("validation", "test"):
        out = stats_dir / f"{run}_eval_{split}.json"
        if out.exists() and not force:
            logger.info("[%s] eval/%s  cached", run, split)
            continue
        t0 = time.time()
        # max_batches=None -> the entire split, which is the reported number.
        res = evaluate_split(config, ckpt, split, device, max_batches=None)
        res["run"] = run
        res["eval_seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        logger.info("[%s] eval/%-10s loss %.5f  ppl %8.2f  bpb %s  (%.0fs)",
                    run, split, res["cross_entropy_nats"], res["perplexity"],
                    res["bits_per_byte"], res["eval_seconds"])


def stage_generation(run: str, config: dict, ckpt: Path, device: str,
                     stats_dir: Path, n: int, force: bool) -> None:
    """Generation quality and diversity under four decoding settings.

    Args:
        run: Run name.
        config: Parsed config.
        ckpt: Checkpoint path.
        device: Device string.
        stats_dir: Where the JSON goes.
        n: Number of held-out prompts.
        force: Recompute even if the output exists.
    """
    from contextlib import nullcontext

    import sentencepiece as spm

    from common.data import TokenDataset
    from common.evaluate import load_model_from_checkpoint
    from common.generate import generate_samples

    out = stats_dir / f"{run}_generation.json"
    if out.exists() and not force:
        logger.info("[%s] generation  cached", run)
        return

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = "float32" if device_type == "cpu" else config["train"].get(
        "dtype", "float16")
    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    model, ck = load_model_from_checkpoint(ckpt, device)
    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])
    ds = TokenDataset(config["paths"]["test_bin"], config["model"]["block_size"],
                      device, seed=config["train"]["seed"])

    t0 = time.time()
    res = generate_samples(model, ds, sp, n, 64, 64, ctx, batch_size=16,
                           seed=config["train"]["seed"])
    payload = {"run": run, "language": config.get("language"),
               "checkpoint": str(ckpt), "checkpoint_step": ck.get("step"),
               "split": "test", "n_prompts": n, "prefix_tokens": 64,
               "continue_tokens": 64, "decodings": res,
               "generation_seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    logger.info("[%s] generation  %d prompts x 4 decodings  (%.0fs)",
                run, n, payload["generation_seconds"])


def stage_attention(run: str, config: dict, ckpt: Path, device: str,
                    stats_dir: Path, fig_dir: Path, force: bool) -> None:
    """Attention heatmaps, entropy and distance.

    Args:
        run: Run name.
        config: Parsed config.
        ckpt: Checkpoint path.
        device: Device string.
        stats_dir: Where the JSON goes.
        fig_dir: Where the PNGs go.
        force: Recompute even if the output exists.
    """
    from common.attention import analyze

    lang = config.get("language", run)
    out = stats_dir / f"{lang}_attention.json"
    if out.exists() and not force:
        logger.info("[%s] attention  cached", run)
        return
    s = analyze(config, ckpt, device, stats_dir, fig_dir, split="test",
                n_examples=3, positions=96, skip=16)
    logger.info("[%s] attention  head types %s", run, s["label_counts"])


# --------------------------------------------------------------------- tables

def _load(stats_dir: Path, name: str) -> dict | None:
    """Read a stats JSON if it exists, else return None."""
    p = stats_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_tables(runs: list[str], stats_dir: Path, out_path: Path) -> None:
    """Assemble every side-by-side table the Phase 2 report needs.

    Args:
        runs: Run names, in the order they should appear as columns.
        stats_dir: Directory holding the per-stage JSON.
        out_path: Markdown destination.
    """
    present = [r for r in runs
               if _load(stats_dir, f"{r}_eval_validation.json") is not None]
    L = ["# Phase 2 — evaluation tables", "",
         "Generated by `scripts/run_eval.py`. Every number is computed over the",
         "**entire** split, not the truncated monitor used during training.", ""]

    # --- intrinsic ----------------------------------------------------------
    L += ["## 1. Intrinsic language modelling", "",
          "Perplexity is per *token* and the two models have different",
          "tokenizers, so it is not comparable across languages. Bits-per-byte",
          "divides out the tokenizer and is the number to compare.", ""]
    for split in ("validation", "test"):
        L += [f"### {split.capitalize()} split", "",
              "| metric | " + " | ".join(LABELS.get(r, r) for r in present) + " |",
              "|---|" + "---:|" * len(present)]
        rows = [("tokens scored", "tokens_scored", "{:,}"),
                ("bytes per token", "bytes_per_token", "{:.4f}"),
                ("cross-entropy (nats/token)", "cross_entropy_nats", "{:.5f}"),
                ("**perplexity**", "perplexity", "{:.2f}"),
                ("**bits per byte**", "bits_per_byte", "{:.4f}")]
        data = {r: _load(stats_dir, f"{r}_eval_{split}.json") for r in present}
        for label, key, fmt in rows:
            cells = []
            for r in present:
                v = (data[r] or {}).get(key)
                cells.append(fmt.format(v) if v is not None else "—")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")

    # --- generation ---------------------------------------------------------
    gen = {r: _load(stats_dir, f"{r}_generation.json") for r in present}
    if any(gen.values()):
        L += ["## 2. Generation quality and diversity", "",
              "64-token prefixes from the held-out test split, 64 tokens",
              "generated. BLEU and ROUGE-L use whitespace tokenization; chrF++",
              "is character-based and is the most trustworthy of the three for",
              "these scripts. All three compare a generation against the one",
              "continuation that happened to follow in the corpus, which for",
              "open-ended generation is a weak proxy for quality — they are",
              "reported because they are required and because they are",
              "comparable *between* the two models under identical conditions.",
              ""]
        metrics = [("BLEU-4", "bleu4", "{:.4f}"),
                   ("chrF++", "chrf_pp", "{:.4f}"),
                   ("ROUGE-L", "rouge_l", "{:.4f}"),
                   ("repetition rate (4-gram)", "repetition_rate_4gram", "{:.4f}"),
                   ("Distinct-1", "distinct_1", "{:.4f}"),
                   ("Distinct-2", "distinct_2", "{:.4f}")]
        for r in present:
            g = gen.get(r)
            if not g:
                continue
            decs = list(g["decodings"].keys())
            L += [f"### {LABELS.get(r, r)}", "",
                  "| metric | " + " | ".join(decs) + " |",
                  "|---|" + "---:|" * len(decs)]
            for label, key, fmt in metrics:
                cells = []
                for d in decs:
                    v = g["decodings"][d]["metrics"].get(key)
                    cells.append(fmt.format(v) if v is not None else "—")
                L.append(f"| {label} | " + " | ".join(cells) + " |")
            L.append("")

    # --- attention ----------------------------------------------------------
    L += ["## 3. Attention summary", "",
          "Entropy is normalized by `ln(i+1)`, the maximum available at query",
          "position `i` under the causal mask; distance is the expected",
          "backward offset in tokens. Both exclude the first 16 positions,",
          "which are degenerate by construction.", ""]
    for r in present:
        cfg = yaml.safe_load(CONFIGS[r].read_text(encoding="utf-8"))
        a = _load(stats_dir, f"{cfg.get('language', r)}_attention.json")
        if not a:
            continue
        pl = a["per_layer"]
        L += [f"### {LABELS.get(r, r)} — head types: {a['label_counts']}", "",
              "| layer | norm. entropy | distance (tokens) | distance (rel.) | "
              "sink mass | prev-token mass |",
              "|---:|---:|---:|---:|---:|---:|"]
        for l in range(a["n_layer"]):
            L.append(f"| {l} | {pl['entropy_normalized'][l]:.3f} | "
                     f"{pl['distance_tokens'][l]:.2f} | "
                     f"{pl['distance_relative'][l]:.3f} | "
                     f"{pl['sink_mass_pos0'][l]:.3f} | "
                     f"{pl['previous_token_mass'][l]:.3f} |")
        L.append("")

    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    """CLI: run every evaluation stage for every requested run, then tabulate."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["telugu", "nepali"],
                   choices=list(CONFIGS))
    p.add_argument("--ckpt-root", type=Path, default=Path("checkpoints"))
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase2/stats"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase2/figures"))
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["eval", "generation", "attention"])
    p.add_argument("--tables-only", action="store_true",
                   help="Rebuild the Markdown tables from existing JSON only.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.tables_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device: %s", device)
        for run in args.runs:
            config, ckpt = load_config(run, args.ckpt_root)
            logger.info("\n=== %s  (%s, step %s) ===", run, ckpt.name,
                        torch.load(ckpt, map_location="cpu",
                                   weights_only=False).get("step"))
            if "eval" not in args.skip:
                stage_eval(run, config, ckpt, device, args.stats_dir, args.force)
            if "generation" not in args.skip:
                stage_generation(run, config, ckpt, device, args.stats_dir,
                                 args.n_prompts, args.force)
            if "attention" not in args.skip:
                stage_attention(run, config, ckpt, device, args.stats_dir,
                                args.fig_dir, args.force)

    print()
    build_tables(args.runs, args.stats_dir, args.stats_dir / "comparison.md")
    print()


if __name__ == "__main__":
    main()

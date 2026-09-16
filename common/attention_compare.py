"""
Pretrained-vs-finetuned attention comparison on comparative-reasoning prompts
(Phase 3.2).

    python -m common.attention_compare --lang telugu \\
        --pretrained checkpoints/telugu/best.pt \\
        --finetuned checkpoints/telugu_reasoning/best.pt

WHAT IS COMPARED, AND ON WHAT INPUT
-----------------------------------
The Phase 2 toolkit (``common/attention.py``) is reused unchanged for the
statistics -- normalized entropy, mean attention distance, sink mass,
previous-token mass, head labels. What changes is the *input*: instead of
random passages from the held-out corpus, both checkpoints are run on the SAME
fixed set of reasoning examples from the test split, rendered exactly as in
training (``<s> prompt answer </s>``). Same tokens in, so any difference in
the maps is finetuning and nothing else.

A REASONING-SPECIFIC MEASUREMENT
--------------------------------
Per-layer entropy and distance say whether attention became more local or
more diffuse, but not whether it became more *useful*. For a comparative
prompt the useful thing to look at, when producing the answer, is the entity
mentions in the premises -- and, for tasks whose answer is a name, the
mentions of the *correct* entity in particular. So for every sequence the
token spans of every entity mention are located (via SentencePiece's
character offsets), and for the query positions that generate the answer the
attention mass falling on (a) any entity token and (b) the gold entity's
tokens is averaged per layer. A head whose gold-entity mass rises after
finetuning is a head that learned to look at the right thing.

Figures produced (all labelled, all sharing one colour scale per figure):

    <lang>_attention_compare_heatmaps.png   early + late layer, pretrained row
                                             above finetuned row, six heads
    <lang>_attention_compare_profile.png    per-layer stats, both checkpoints
    <lang>_attention_delta.png              (finetuned - pretrained) per head
    <lang>_entity_attention.png             mass on entities from the answer
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from common.attention import analyze, collect_attention
from common.evaluate import load_model_from_checkpoint
from common.reasoning_eval import make_ctx
from common.reasoning_gen import TASKS, read_jsonl

logger = logging.getLogger(__name__)


# ------------------------------------------------------------- sequences

def pick_examples(examples: list[dict], n_per_task: int, seed: int) -> list[dict]:
    """Choose a fixed, task-balanced subset of test examples."""
    rng = np.random.default_rng(seed)
    out = []
    for t in TASKS:
        pool = [e for e in examples if e["task"] == t]
        idx = rng.choice(len(pool), size=min(n_per_task, len(pool)), replace=False)
        out.extend(pool[i] for i in idx)
    return out


def piece_offsets(sp, text: str) -> list[tuple[int, int, int]]:
    """Tokenize ``text`` and recover each piece's character span.

    Walks the pieces over the text: a piece's ``▁`` marker stands for a
    preceding space (or, for the first piece, SentencePiece's dummy prefix,
    which has no character in the text). This avoids the ``proto`` output
    APIs, which differ between SentencePiece releases. A piece that cannot be
    located (byte-fallback of an out-of-vocabulary character) is given an
    empty span, which simply excludes it from every mask.

    Returns:
        ``[(piece_id, begin, end)]`` with ``[begin, end)`` character offsets.
    """
    ids = sp.encode(text)
    out, pos = [], 0
    for pid in ids:
        piece = sp.id_to_piece(pid)
        if piece.startswith("<") and piece.endswith(">"):     # byte fallback / special
            out.append((pid, pos, pos))
            continue
        txt = piece.replace("▁", " ")
        if txt.startswith(" "):
            core = txt[1:]
            # skip the space in the text if present (not for the dummy prefix)
            if pos < len(text) and text[pos] == " ":
                pos += 1
        else:
            core = txt
        if not core:
            out.append((pid, pos, pos))
            continue
        i = text.find(core, pos)
        if i < 0 or i - pos > 2:                             # lost sync: skip
            out.append((pid, pos, pos))
            continue
        out.append((pid, i, i + len(core)))
        pos = i + len(core)
    return out


def encode_with_spans(sp, example: dict) -> dict:
    """Tokenize ``<s> prompt answer </s>`` and locate entity and answer spans.

    Returns:
        Dict with ``ids`` (np.ndarray), ``entity_mask`` (bool per token),
        ``gold_mask`` (bool per token; the gold entity's mentions, if the
        answer is an entity name, else all False) and ``answer_positions``
        (query positions whose *output* is an answer token: the last prompt
        token through the token before ``</s>``).
    """
    prompt, answer = example["prompt"], example["answer"]
    pieces = piece_offsets(sp, prompt)

    def spans_of(name: str) -> list[tuple[int, int]]:
        out, start = [], 0
        while True:
            i = prompt.find(name, start)
            if i < 0:
                return out
            out.append((i, i + len(name)))
            start = i + 1

    ent_spans = [s for e in example["entities"] for s in spans_of(e)]
    gold_spans = spans_of(answer) if answer in example["entities"] else []

    def mask_for(spans):
        m = [False]                                     # <s>
        for _, b, e in pieces:
            m.append(e > b and any(b < se and e > sb for sb, se in spans))
        return m

    ent_mask = mask_for(ent_spans)
    gold_mask = mask_for(gold_spans)
    p_ids = [sp.bos_id()] + [pid for pid, _, _ in pieces]
    a_ids = sp.encode(answer) + [sp.eos_id()]
    ids = p_ids + a_ids
    n_p = len(p_ids)
    pad = [False] * len(a_ids)
    return {
        "ids": np.asarray(ids, dtype=np.int64),
        "entity_mask": np.asarray(ent_mask + pad),
        "gold_mask": np.asarray(gold_mask + pad),
        # Position n_p-1 (the ':' of the answer prefix) predicts the first
        # answer token; position len(ids)-2 predicts </s>.
        "answer_positions": np.arange(n_p - 1, len(ids) - 1),
        "task": example["task"],
        "answer_is_entity": bool(gold_spans),
    }


# ------------------------------------------------------------ statistics

def entity_attention(attn: np.ndarray, enc: dict) -> dict[str, np.ndarray]:
    """Mass on entity tokens / gold-entity tokens from the answer positions.

    Args:
        attn: ``(L, H, T, T)``.
        enc: Output of :func:`encode_with_spans`.

    Returns:
        ``{"entity": (L, H), "gold": (L, H)}`` mean attention mass.
    """
    q = enc["answer_positions"]
    rows = attn[:, :, q, :]                                   # (L, H, Q, T)
    ent = rows[..., enc["entity_mask"]].sum(-1).mean(-1)      # (L, H)
    if enc["answer_is_entity"] and enc["gold_mask"].any():
        gold = rows[..., enc["gold_mask"]].sum(-1).mean(-1)
    else:
        gold = np.full(ent.shape, np.nan)
    return {"entity": ent, "gold": gold}


# ------------------------------------------------------------- figures
# Visualization only; every statistic above is computed without matplotlib.

def plot_paired_heatmaps(attn_pre: np.ndarray, attn_ft: np.ndarray,
                         layers: list[int], heads: list[int], out_path: Path,
                         language: str, answer_start: int,
                         gamma: float = 0.45) -> None:
    """Early and late layer, pretrained row above finetuned row, per layer.

    A horizontal line marks the first answer position, so the reader can see
    what the answer tokens attend to. One power-law colour scale is shared by
    every panel (see ``common.attention.plot_head_heatmaps`` for why).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    T = attn_pre.shape[-1]
    tri = np.tril(np.ones((T, T), dtype=bool))
    rows = []
    for l in layers:
        rows.append((f"layer {l}\npretrained", attn_pre[l]))
        rows.append((f"layer {l}\nfinetuned", attn_ft[l]))
    shown = np.stack([[r[1][h][:T, :T] for h in heads] for r in rows])
    vmax = float(np.percentile(shown[..., tri], 99.0))
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1e-6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#E8E8E8")

    n_rows, n_cols = len(rows), len(heads)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.3 * n_cols + 1.9, 2.3 * n_rows + 1.2),
                             squeeze=False, layout="constrained")
    im = None
    for r, (label, _) in enumerate(rows):
        for c, h in enumerate(heads):
            ax = axes[r][c]
            im = ax.imshow(np.where(tri, shown[r, c], np.nan), cmap=cmap,
                           norm=norm, aspect="equal", origin="upper",
                           interpolation="nearest")
            ax.axhline(answer_start - 0.5, color="#00E5FF", lw=0.8, ls="--")
            if r == 0:
                ax.set_title(f"head {h}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{label}\nquery position $i$", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("key position $j$", fontsize=9)
            ax.set_xticks([0, T // 2, T - 1])
            ax.set_yticks([0, T // 2, T - 1])
            ax.tick_params(labelsize=7)
            if c:
                ax.set_yticklabels([])
            if r != n_rows - 1:
                ax.set_xticklabels([])
    fig.suptitle(f"Attention on a comparative-reasoning prompt — {language}: "
                 f"pretrained vs finetuned\n(dashed line = first answer token; "
                 f"grey = masked future)", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.015)
    cbar.set_label(f"attention probability $p_{{ij}}$ (power scale, "
                   f"$\\gamma$={gamma}; shared across panels)", fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_profile_comparison(pre: dict, ft: dict, out_path: Path,
                            language: str) -> None:
    """Per-layer statistics for both checkpoints on the same prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(pre["n_layer"]))
    keys = [("entropy_normalized", "normalized entropy", "o"),
            ("distance_relative", "relative attention distance", "s"),
            ("previous_token_mass", "mass on previous token", "d"),
            ("sink_mass_pos0", "mass on position 0 (sink)", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for ax, (k, label, mk) in zip(axes.flat, keys):
        ax.plot(layers, pre["per_layer"][k], marker=mk, ls="--", color="#888888",
                label="pretrained")
        ax.plot(layers, ft["per_layer"][k], marker=mk, ls="-", color="#C0392B",
                label="finetuned")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("value in [0, 1]", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("layer index (0 = closest to input)", fontsize=10)
    fig.suptitle(f"Attention by depth on reasoning prompts — {language}: "
                 f"pretrained vs finetuned", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_delta(pre: dict, ft: dict, out_path: Path, language: str) -> None:
    """(finetuned - pretrained) per head for entropy and distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("entropy_normalized", "Δ normalized entropy"),
              ("distance_tokens", "Δ mean attention distance (tokens)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (k, title) in zip(axes, panels):
        d = np.array(ft["per_head"][k]) - np.array(pre["per_head"][k])
        lim = float(np.abs(d).max()) or 1e-6
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        L, H = d.shape
        for l in range(L):
            for h in range(H):
                ax.text(h, l, f"{d[l, h]:+.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("head index", fontsize=10)
        ax.set_ylabel("layer index (0 = closest to input)", fontsize=10)
        ax.set_xticks(range(H))
        ax.set_yticks(range(L))
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("finetuned − pretrained", fontsize=9)
    fig.suptitle(f"How finetuning moved each head — {language} "
                 f"(red = more diffuse / longer range after finetuning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_entity_attention(pre_e: dict, ft_e: dict, out_path: Path,
                          language: str) -> None:
    """Per-layer attention mass from the answer onto entity mentions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(pre_e["entity_per_layer"])
    layers = list(range(L))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(layers, pre_e["entity_per_layer"], "o--", color="#888888",
            label="any entity mention — pretrained")
    ax.plot(layers, ft_e["entity_per_layer"], "o-", color="#C0392B",
            label="any entity mention — finetuned")
    ax.plot(layers, pre_e["gold_per_layer"], "s--", color="#4A90D9",
            label="gold entity's mentions — pretrained")
    ax.plot(layers, ft_e["gold_per_layer"], "s-", color="#1F3A93",
            label="gold entity's mentions — finetuned")
    ax.axhline(pre_e["entity_token_fraction"], color="black", lw=0.8, ls=":",
               label="entity tokens' share of the sequence (uniform baseline)")
    ax.set_title(f"Where the answer positions look — {language}", fontsize=12)
    ax.set_xlabel("layer index (0 = closest to input)", fontsize=10)
    ax.set_ylabel("mean attention mass from answer positions", fontsize=10)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


# ---------------------------------------------------------------- driver

def compare(lang: str, config: dict, pre_ckpt: Path, ft_ckpt: Path,
            device: str, stats_dir: Path, fig_dir: Path,
            n_per_task: int = 3, skip: int = 4, seed: int = 0) -> dict:
    """Run the full pretrained-vs-finetuned comparison for one language.

    Args:
        lang: Language / run name, used as the file stem.
        config: The finetune YAML (paths, tokenizer, data).
        pre_ckpt: Pretrained checkpoint.
        ft_ckpt: Finetuned checkpoint.
        device: Device string.
        stats_dir: JSON / Markdown destination.
        fig_dir: PNG destination.
        n_per_task: Test examples per task family to analyze.
        skip: Leading query positions excluded from the averages. Reasoning
            prompts are ~40 tokens, so the Phase 2 default of 16 would discard
            half of them; 4 removes only the degenerate first positions.
        seed: Example-selection seed.

    Returns:
        The comparison summary written to ``<lang>_attention_compare.json``.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])
    examples = pick_examples(read_jsonl(Path(config["data"]["test"])),
                             n_per_task, seed)
    encs = [encode_with_spans(sp, e) for e in examples]
    seqs = [e["ids"] for e in encs]
    ctx = make_ctx(device, config["train"].get("dtype", "float16"))

    # 1. The Phase 2 statistics, on identical inputs, for both checkpoints.
    pre, attn_pre = analyze(config, pre_ckpt, device, stats_dir, fig_dir,
                            skip=skip, name=f"{lang}_pretrained_reasoning",
                            sequences=seqs, return_attention=True,
                            source_label="reasoning test prompt, with its gold answer")
    ft, attn_ft = analyze(config, ft_ckpt, device, stats_dir, fig_dir,
                          skip=skip, name=f"{lang}_finetuned_reasoning",
                          sequences=seqs, return_attention=True,
                          source_label="reasoning test prompt, with its gold answer")

    # 2. Entity-attention from the answer positions, averaged over sequences.
    def entity_stats(ckpt: Path) -> dict:
        model, _ = load_model_from_checkpoint(ckpt, device)
        ent, gold, frac = [], [], []
        for enc in encs:
            a = collect_attention(model, enc["ids"], device, ctx)
            r = entity_attention(a, enc)
            ent.append(r["entity"])
            gold.append(r["gold"])
            frac.append(enc["entity_mask"].mean())
        ent = np.mean(ent, axis=0)                              # (L, H)
        gold = np.nanmean(np.stack(gold), axis=0)               # (L, H)
        return {"entity_per_head": np.round(ent, 4).tolist(),
                "gold_per_head": np.round(np.nan_to_num(gold), 4).tolist(),
                "entity_per_layer": np.round(ent.mean(1), 4).tolist(),
                "gold_per_layer": np.round(np.nan_to_num(gold).mean(1), 4).tolist(),
                "entity_token_fraction": round(float(np.mean(frac)), 4),
                "n_sequences_with_entity_answer": int(sum(e["answer_is_entity"] for e in encs))}

    pre_e, ft_e = entity_stats(pre_ckpt), entity_stats(ft_ckpt)

    # 3. Figures.
    L = pre["n_layer"]
    heads = list(range(min(pre["n_head"], 6)))
    plot_paired_heatmaps(attn_pre, attn_ft, [0, L - 1], heads,
                         fig_dir / f"{lang}_attention_compare_heatmaps.png",
                         lang, int(encs[0]["answer_positions"][0]))
    plot_profile_comparison(pre, ft, fig_dir / f"{lang}_attention_compare_profile.png", lang)
    plot_delta(pre, ft, fig_dir / f"{lang}_attention_delta.png", lang)
    plot_entity_attention(pre_e, ft_e, fig_dir / f"{lang}_entity_attention.png", lang)

    # 4. Summary.
    def per_layer_delta(k):
        return np.round(np.array(ft["per_layer"][k]) - np.array(pre["per_layer"][k]), 4).tolist()

    summary = {
        "language": lang,
        "pretrained_checkpoint": str(pre_ckpt),
        "finetuned_checkpoint": str(ft_ckpt),
        "n_sequences": len(seqs),
        "sequence_tasks": [e["task"] for e in examples],
        "skip_positions": skip,
        "first_sequence": {"id": examples[0]["id"], "task": examples[0]["task"],
                           "prompt": examples[0]["prompt"],
                           "answer": examples[0]["answer"],
                           "answer_start_position": int(encs[0]["answer_positions"][0])},
        "head_labels": {"pretrained": pre["label_counts"],
                        "finetuned": ft["label_counts"]},
        "per_layer": {"pretrained": pre["per_layer"], "finetuned": ft["per_layer"],
                      "delta": {k: per_layer_delta(k) for k in pre["per_layer"]}},
        "entity_attention": {"pretrained": pre_e, "finetuned": ft_e},
        "largest_head_changes": _largest_changes(pre, ft, k=8),
    }
    out = stats_dir / f"{lang}_attention_compare.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("  wrote %s", out)
    return summary


def _largest_changes(pre: dict, ft: dict, k: int = 8) -> list[dict]:
    """Heads whose entropy or distance moved the most, for the write-up."""
    out = []
    for stat in ("entropy_normalized", "distance_tokens", "previous_token_mass"):
        d = np.array(ft["per_head"][stat]) - np.array(pre["per_head"][stat])
        flat = [(abs(d[l, h]), l, h) for l in range(d.shape[0]) for h in range(d.shape[1])]
        for _, l, h in sorted(flat, reverse=True)[:k]:
            out.append({"stat": stat, "layer": l, "head": h,
                        "pretrained": round(float(pre["per_head"][stat][l][h]), 4),
                        "finetuned": round(float(ft["per_head"][stat][l][h]), 4),
                        "delta": round(float(d[l, h]), 4)})
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="Finetune YAML; defaults to <lang>/configs/finetune.yaml")
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--n-per-task", type=int, default=3)
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase3/stats"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase3/figures"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg_path = args.config or Path(f"{args.lang}/configs/finetune.yaml")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    s = compare(args.lang, config, args.pretrained, args.finetuned, device,
                args.stats_dir, args.fig_dir, args.n_per_task, args.skip)
    print(f"\n  head labels  pretrained {s['head_labels']['pretrained']}")
    print(f"               finetuned  {s['head_labels']['finetuned']}")
    print(f"  Δ entropy per layer  {s['per_layer']['delta']['entropy_normalized']}")
    print(f"  Δ distance per layer {s['per_layer']['delta']['distance_tokens']}")


if __name__ == "__main__":
    main()

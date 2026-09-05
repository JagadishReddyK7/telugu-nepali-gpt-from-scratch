"""
Training curves from the JSONL logs. Needs no GPU and no checkpoints.

    python scripts/plot_training.py

Reads every ``report/phase2/logs/*_train.jsonl`` written by ``common.train`` and
produces the loss curves required by Phase 2 deliverable 4, plus the learning
rate, gradient norm and throughput traces that make the run auditable.

A NOTE ON THE TWO DIFFERENT VALIDATION NUMBERS
----------------------------------------------
``common/train.py`` evaluates on ``eval_batches`` batches during training
(line 519) but on ``eval_batches * 4`` at the end (line 559), and both start
from the beginning of the validation split. The in-training curve is therefore
a FIXED PREFIX MONITOR -- consistent with itself across the whole run, and so
perfectly good for watching convergence -- but it is not the same quantity as
the final figure, and for Telugu the two differ by roughly 0.8 nats because the
head of that split is harder than its tail.

Every figure this script draws labels the in-training series accordingly, and
the final point is drawn as a separate marker rather than joined to the curve.
Reporting them as one series would be a straightforward misrepresentation.

The number that belongs in the results table comes from ``common.evaluate``
over the whole split, not from either of these.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent identity per run across every figure.
STYLE = {
    "telugu":       dict(color="#B45309", label="Model H — Telugu"),
    "nepali":       dict(color="#1D4ED8", label="Model L — Nepali"),
    "telugu_nopos": dict(color="#059669", label="Telugu, no positional emb. (ablation)"),
}
FALLBACK = dict(color="#6B7280", label=None)


def load_log(path: Path) -> dict:
    """Parse one training log into its three kinds of record.

    Args:
        path: Path to a ``*_train.jsonl``.

    Returns:
        Dict with ``name``, ``train`` (step records), ``val`` (periodic
        evaluations), ``final`` (the terminal record or None) and ``malformed``
        (line numbers that failed to parse).

    Raises:
        FileNotFoundError: If the log does not exist.
    """
    rows, malformed = [], []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed.append(i)
    finals = [r for r in rows if "final_val_loss" in r]
    return {
        "name": path.stem.replace("_train", ""),
        "train": [r for r in rows if "train_loss" in r],
        "val": [r for r in rows if "val_loss" in r],
        "final": finals[-1] if finals else None,
        "malformed": malformed,
    }


def style_for(name: str) -> dict:
    """Return the plotting style for a run name."""
    s = dict(STYLE.get(name, FALLBACK))
    if s["label"] is None:
        s["label"] = name
    return s


def plot_loss_curves(runs: list[dict], out_path: Path) -> None:
    """Training and validation loss against tokens consumed, all runs overlaid.

    Args:
        runs: Parsed logs from :func:`load_log`.
        out_path: PNG destination.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for r in runs:
        s = style_for(r["name"])
        if r["train"]:
            ax.plot([d["tokens"] / 1e6 for d in r["train"]],
                    [d["train_loss"] for d in r["train"]],
                    color=s["color"], alpha=0.45, linewidth=1.0,
                    label=f"{s['label']} — train")
        if r["val"]:
            ax.plot([d["tokens"] / 1e6 for d in r["val"]],
                    [d["val_loss"] for d in r["val"]],
                    color=s["color"], linewidth=2.0, marker="o", markersize=3.5,
                    label=f"{s['label']} — val (819k-token monitor prefix)")
        if r["final"]:
            ax.scatter([r["final"]["tokens"] / 1e6],
                       [r["final"]["final_val_loss"]],
                       color=s["color"], marker="*", s=210, zorder=5,
                       edgecolors="black", linewidths=0.6,
                       label=f"{s['label']} — final val (3.3M-token pass)")

    ax.axhline(math.log(16000), color="#9CA3AF", linestyle=":", linewidth=1.2)
    ax.annotate("uniform over 16k vocabulary,  $\\ln(16000) = 9.68$",
                xy=(0.015, math.log(16000)), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color="#6B7280")

    ax.set_title("Pretraining loss — cross-entropy against tokens consumed",
                 fontsize=13)
    ax.set_xlabel("training tokens consumed (millions)", fontsize=11)
    ax.set_ylabel("cross-entropy loss (nats / token)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_perplexity(runs: list[dict], out_path: Path) -> None:
    """Validation perplexity on a log axis.

    Perplexity spans 16,000 to ~50 over a run, so a linear axis shows only the
    first few hundred steps. The log axis makes the late-run behaviour --
    where the models actually differ -- readable.

    Args:
        runs: Parsed logs.
        out_path: PNG destination.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for r in runs:
        if not r["val"]:
            continue
        s = style_for(r["name"])
        ax.plot([d["tokens"] / 1e6 for d in r["val"]],
                [math.exp(min(d["val_loss"], 20)) for d in r["val"]],
                color=s["color"], linewidth=2.0, marker="o", markersize=3.5,
                label=s["label"])
        if r["final"]:
            ax.scatter([r["final"]["tokens"] / 1e6],
                       [math.exp(min(r["final"]["final_val_loss"], 20))],
                       color=s["color"], marker="*", s=210, zorder=5,
                       edgecolors="black", linewidths=0.6)
    ax.set_yscale("log")
    ax.set_title("Validation perplexity (stars: final full-pass evaluation)",
                 fontsize=13)
    ax.set_xlabel("training tokens consumed (millions)", fontsize=11)
    ax.set_ylabel("perplexity  $e^{\\mathcal{L}}$  (log scale)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_diagnostics(runs: list[dict], out_path: Path) -> None:
    """Learning rate, gradient norm and throughput in one three-panel figure.

    These are the traces that show the run was healthy: the cosine schedule
    completed, gradients never diverged, and the GPU was fed at a steady rate.

    Args:
        runs: Parsed logs.
        out_path: PNG destination.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.6), sharex=True)
    panels = [
        ("lr", "learning rate", "Learning-rate schedule (cosine with warmup)", False),
        ("grad_norm", "gradient $\\ell_2$ norm",
         "Gradient norm (clipped at 1.0)", False),
        ("tokens_per_sec", "tokens / second",
         "Throughput on the Kaggle T4", False),
    ]
    for ax, (key, ylabel, title, logy) in zip(axes, panels):
        for r in runs:
            pts = [(d["tokens"] / 1e6, d[key]) for d in r["train"] if key in d]
            if not pts:
                continue
            s = style_for(r["name"])
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=s["color"], linewidth=1.2, alpha=0.85,
                    label=s["label"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[1].axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.0)
    axes[-1].set_xlabel("training tokens consumed (millions)", fontsize=11)
    fig.suptitle("Training diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary(runs: list[dict], out_path: Path) -> None:
    """Write a Markdown table of run outcomes, ready to paste into the report.

    Args:
        runs: Parsed logs.
        out_path: Markdown destination.
    """
    lines = ["# Pretraining run summary", "",
             "Generated by `scripts/plot_training.py` from the training logs.",
             "",
             "| run | steps | tokens | final train loss | final val loss | "
             "final PPL | wall clock | mean tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        if not r["train"]:
            continue
        last = r["train"][-1]
        fin = r["final"] or {}
        fv = fin.get("final_val_loss")
        wall = fin.get("wall_clock_s")
        tps = [d["tokens_per_sec"] for d in r["train"] if "tokens_per_sec" in d]

        cells = [
            style_for(r["name"])["label"],
            f"{fin.get('step', last['step']):,}",
            f"{fin.get('tokens', last['tokens']):,}",
            f"{last['train_loss']:.4f}",
            f"{fv:.5f}" if fv is not None else "—",
            f"{math.exp(min(fv, 20)):.2f}" if fv is not None else "—",
            f"{wall / 3600:.2f} h" if wall else "—",
            f"{sum(tps) / len(tps):,.0f}" if tps else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    bad = [(r["name"], r["malformed"]) for r in runs if r["malformed"]]
    if bad:
        lines += ["", "## Malformed log lines", ""]
        lines += [f"* `{n}`: lines {m}" for n, m in bad]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    """CLI: read every training log and draw the Phase 2 curve figures."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", type=Path, default=Path("report/phase2/logs"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase2/figures"))
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase2/stats"))
    args = p.parse_args()

    paths = sorted(args.log_dir.glob("*_train.jsonl"))
    paths = [p for p in paths if not p.name.startswith("_")]   # skip smoke tests
    if not paths:
        raise SystemExit(f"no *_train.jsonl found in {args.log_dir}")

    runs = [load_log(p) for p in paths]
    print(f"\nloaded {len(runs)} run(s)")
    for r in runs:
        fv = r["final"]["final_val_loss"] if r["final"] else None
        print(f"  {r['name']:14s} {len(r['train']):>4} train, "
              f"{len(r['val']):>3} val rows"
              + (f", final val {fv:.5f} (ppl {math.exp(fv):.2f})" if fv else
                 ", NO FINAL ROW — run did not complete")
              + (f"  [{len(r['malformed'])} MALFORMED LINES]"
                 if r["malformed"] else ""))

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    print()
    plot_loss_curves(runs, args.fig_dir / "loss_curves.png")
    plot_perplexity(runs, args.fig_dir / "val_perplexity.png")
    plot_diagnostics(runs, args.fig_dir / "training_diagnostics.png")
    write_summary(runs, args.stats_dir / "run_summary.md")
    print()


if __name__ == "__main__":
    main()

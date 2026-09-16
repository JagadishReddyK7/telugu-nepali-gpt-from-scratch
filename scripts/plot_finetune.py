"""
Finetuning curves and reasoning-accuracy figures. Needs no GPU, no checkpoints.

    python scripts/plot_finetune.py

Reads ``report/phase3/logs/<lang>_finetune.jsonl`` (written by
``common.finetune``) and ``report/phase3/stats/<lang>_{pretrained,finetuned}
_reasoning_test.json`` (written by ``scripts/run_phase3.py``) and draws:

    finetune_curves.png           train loss, validation loss, validation
                                  answer-token loss, validation exact match
                                  against optimizer step, both languages
    reasoning_accuracy_by_task.png  pretrained vs finetuned exact match per
                                  task family, both languages
    reasoning_generalisation.png  finetuned exact match by hops and by
                                  seen / held-out phrasing

Computation (reading and reducing the logs) and drawing are kept in separate
functions, per the implementation guidelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STYLE = {"telugu": {"color": "#C0392B", "label": "Model H — Telugu"},
         "nepali": {"color": "#1F3A93", "label": "Model L — Nepali"}}
TASK_ORDER = ["superlative_chain", "numeric_compare", "chain_relation",
              "yes_no", "ordering"]


# ---------------------------------------------------------------- reading

def read_log(path: Path) -> dict:
    """Split a finetune JSONL log into train and eval series.

    A log may hold more than one session (resume appends). Records are kept
    in file order; a resumed run's steps continue from where it stopped, so
    the series stay monotone in step.
    """
    train, evals, final = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "train_loss" in r:
            train.append(r)
        elif "val_exact_match" in r:
            evals.append(r)
        elif "final_val_exact_match" in r:
            final = r
    return {"train": train, "eval": evals, "final": final}


def load_results(stats_dir: Path, lang: str) -> dict:
    """Load pretrained / finetuned reasoning results for one language."""
    out = {}
    for tag in ("pretrained", "finetuned"):
        p = stats_dir / f"{lang}_{tag}_reasoning_test.json"
        out[tag] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


# ---------------------------------------------------------------- drawing

def plot_curves(logs: dict[str, dict], out_path: Path) -> None:
    """Four panels of finetuning progress for every language present."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [("train_loss", "train", "training loss (weighted CE, nats/token)"),
              ("val_loss", "eval", "validation loss (nats/token)"),
              ("val_answer_loss", "eval", "validation answer-token loss (nats/token)"),
              ("val_exact_match", "eval", "validation exact match")]
    for ax, (key, series, ylabel) in zip(axes.flat, panels):
        for lang, log in logs.items():
            rows = log[series]
            if not rows:
                continue
            st = STYLE.get(lang, {"color": None, "label": lang})
            xs = [r["step"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, marker="o" if series == "eval" else None,
                    ms=3, lw=1.4, color=st["color"], label=st["label"])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split(" (")[0], fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        if key == "val_exact_match":
            ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step", fontsize=10)
    fig.suptitle("Reasoning finetuning — step 0 is the pretrained checkpoint", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_by_task(results: dict[str, dict], out_path: Path) -> None:
    """Grouped bars: exact match per task, pretrained vs finetuned, per language."""
    langs = [l for l, r in results.items() if r["finetuned"]]
    fig, axes = plt.subplots(1, len(langs), figsize=(6.2 * len(langs), 4.8),
                             squeeze=False, sharey=True)
    x = range(len(TASK_ORDER))
    w = 0.38
    for ax, lang in zip(axes[0], langs):
        r = results[lang]
        pre = [r["pretrained"]["exact_match_by_task"].get(t, {}).get("accuracy", 0)
               if r["pretrained"] else 0 for t in TASK_ORDER]
        ft = [r["finetuned"]["exact_match_by_task"].get(t, {}).get("accuracy", 0)
              for t in TASK_ORDER]
        chance = [r["finetuned"].get("choice_chance_by_task", {}).get(t, 0)
                  for t in TASK_ORDER]
        ax.bar([i - w / 2 for i in x], pre, w, color="#BBBBBB", label="pretrained (exact match)")
        ax.bar([i + w / 2 for i in x], ft, w, color=STYLE[lang]["color"],
               label="finetuned (exact match)")
        ax.scatter(list(x), chance, marker="_", s=400, color="black", zorder=3,
                   label="chance (1 / |choices|)")
        for i, v in enumerate(ft):
            ax.text(i + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([t.replace("_", "\n") for t in TASK_ORDER], fontsize=9)
        ax.set_title(STYLE[lang]["label"], fontsize=12)
        ax.set_xlabel("task family", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("exact match on the synthetic test split", fontsize=10)
    fig.suptitle("Reasoning accuracy by task — pretrained vs finetuned", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generalisation(results: dict[str, dict], out_path: Path) -> None:
    """Finetuned exact match by hop count and by seen / held-out phrasing."""
    langs = [l for l, r in results.items() if r["finetuned"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # hops
    ax = axes[0]
    hops = ["1", "2", "3"]
    w = 0.8 / max(len(langs), 1)
    for k, lang in enumerate(langs):
        d = results[lang]["finetuned"]["exact_match_by_hops"]
        ys = [d.get(h, {}).get("accuracy", 0) for h in hops]
        ax.bar([i + (k - (len(langs) - 1) / 2) * w for i in range(len(hops))], ys, w,
               color=STYLE[lang]["color"], label=STYLE[lang]["label"])
    ax.set_xticks(range(len(hops)))
    ax.set_xticklabels([f"{h} hop{'s' if h != '1' else ''}" for h in hops])
    ax.set_xlabel("relations that must be chained", fontsize=10)
    ax.set_ylabel("finetuned exact match", fontsize=10)
    ax.set_title("Accuracy vs reasoning depth", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    # phrasing
    ax = axes[1]
    slices = [("False", "seen phrasing\n(held-out names)"),
              ("True", "held-out phrasing\n(held-out names)")]
    for k, lang in enumerate(langs):
        d = results[lang]["finetuned"]["exact_match_by_template_held_out"]
        ys = [d.get(s, {}).get("accuracy", 0) for s, _ in slices]
        ax.bar([i + (k - (len(langs) - 1) / 2) * w for i in range(len(slices))], ys, w,
               color=STYLE[lang]["color"], label=STYLE[lang]["label"])
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels([s for _, s in slices])
    ax.set_xlabel("test slice", fontsize=10)
    ax.set_ylabel("finetuned exact match", fontsize=10)
    ax.set_title("Generalisation to unseen wording", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("Finetuned reasoning accuracy by generalisation slice", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_all(langs: list[str], log_dir: Path, stats_dir: Path, fig_dir: Path) -> None:
    """Draw every figure for which inputs exist."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs = {l: read_log(log_dir / f"{l}_finetune.jsonl")
            for l in langs if (log_dir / f"{l}_finetune.jsonl").exists()}
    if logs:
        plot_curves(logs, fig_dir / "finetune_curves.png")
    results = {l: load_results(stats_dir, l) for l in langs}
    if any(r["finetuned"] for r in results.values()):
        plot_by_task(results, fig_dir / "reasoning_accuracy_by_task.png")
        plot_generalisation(results, fig_dir / "reasoning_generalisation.png")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=["telugu", "nepali"])
    p.add_argument("--log-dir", type=Path, default=Path("report/phase3/logs"))
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase3/stats"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase3/figures"))
    args = p.parse_args()
    plot_all(args.langs, args.log_dir, args.stats_dir, args.fig_dir)


if __name__ == "__main__":
    main()

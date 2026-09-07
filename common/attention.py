"""
Attention analysis for a trained model: heatmaps, entropy, and distance.

    python -m common.attention --config telugu/configs/model.yaml

Satisfies Phase 2 sec 2.3 "Attention analysis":

  * heatmaps for at least one early and one late layer, multiple heads
  * attention entropy per head and per layer
  * mean attention distance per head and per layer

WHY THE RAW NUMBERS ARE MISLEADING, AND WHAT IS REPORTED INSTEAD
---------------------------------------------------------------
Causal masking makes every per-head statistic a function of position, which
contaminates any average taken naively over the sequence.

At query position ``i`` the head may only distribute mass over ``i + 1`` keys.
Its entropy is therefore bounded above by ``ln(i + 1)``: position 0 has entropy
exactly 0 no matter what the head learned, and position 5 cannot exceed
``ln 6 = 1.79`` even if it is perfectly uniform. Averaging raw entropy over the
sequence measures mostly how long the sequence is.

The same applies to distance. A query at position ``i`` cannot attend further
back than ``i`` tokens, so early positions drag the mean down mechanically.

Two corrections are applied, and both raw and corrected values are reported:

  1. The first ``skip`` positions are excluded from every average (default 16).
     They are degenerate by construction and dominated by the boundary effect.

  2. Entropy is also reported NORMALIZED, as ``H_i / ln(i + 1)`` -- the fraction
     of the maximum entropy available at that position. A value near 1 means
     the head is close to uniform over everything it can see; near 0 means it
     has committed to a few keys. This is comparable across positions, across
     layers, and across the two models, which raw entropy is not.

     Distance is likewise reported both absolutely (in tokens) and relatively
     (as a fraction of ``i``, the distance available).

ATTENTION SINKS
---------------
Transformers reliably learn to dump surplus probability mass on the first
token, which acts as a no-op target when a head has nothing to contribute
(Xiao et al., 2023). This inflates entropy-based "diffuseness" readings and
deflates mean distance, so the fraction of mass landing on position 0 is
reported per head as well. A head with 0.9 sink mass is not a "local" head; it
is an idle head, and the distinction matters for the write-up.

SCRIPT RENDERING IN FIGURES
---------------------------
Matplotlib's bundled fonts carry no Telugu or Devanagari glyphs, so token
strings on a figure axis render as empty boxes. Axes are therefore labelled
with token POSITIONS, and the decoded token strings are written alongside the
figure as Markdown (``*_tokens.md``), which renders both scripts correctly.
The example sentence is reproduced there in full, satisfying the requirement
that heatmaps be accompanied by example sentences in the model's language.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from common.data import TokenDataset
from common.evaluate import load_model_from_checkpoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ collection

@torch.no_grad()
def collect_attention(model, token_ids: np.ndarray, device: str,
                      ctx) -> np.ndarray:
    """Run one sequence through the model and return every attention map.

    Args:
        model: Trained model in eval mode.
        token_ids: 1-D array of token ids, length <= block_size.
        device: Device string.
        ctx: Autocast context.

    Returns:
        Array of shape ``(n_layer, n_head, T, T)`` in float32, where entry
        ``[l, h, i, j]`` is the probability head ``h`` of layer ``l`` assigns to
        key position ``j`` when the query is position ``i``. Rows sum to 1 over
        ``j <= i`` and are exactly 0 for ``j > i``.
    """
    idx = torch.from_numpy(token_ids.astype(np.int64))[None, :].to(device)
    with ctx:
        _, _, attentions = model(idx, need_weights=True)
    # (B=1, H, T, T) per layer -> (L, H, T, T); cast from fp16 before numpy.
    return np.stack([a[0].float().cpu().numpy() for a in attentions])


# ------------------------------------------------------------------- statistics

def _valid_rows(attn: np.ndarray, skip: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the query positions used for averaging and their indices.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Number of leading query positions to discard.

    Returns:
        ``(rows, positions)`` where ``rows`` is ``(L, H, T', T)`` and
        ``positions`` is the ``(T',)`` vector of original query indices.
    """
    T = attn.shape[-1]
    start = min(skip, max(T - 1, 0))
    positions = np.arange(start, T)
    return attn[:, :, start:, :], positions


def entropy_per_head(attn: np.ndarray, skip: int = 16,
                     normalized: bool = True) -> np.ndarray:
    """Mean Shannon entropy of each head's attention distribution.

    Entropy is computed in nats per query position and then averaged over
    positions ``>= skip``. With ``normalized=True`` each position's entropy is
    first divided by ``ln(i + 1)``, the maximum achievable under the causal
    mask, giving a position-independent number in ``[0, 1]``.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Leading query positions to exclude.
        normalized: Divide by the per-position maximum entropy.

    Returns:
        ``(L, H)`` array of mean entropies.
    """
    rows, positions = _valid_rows(attn, skip)
    # Zeros contribute 0 to the sum; clip only to keep log finite.
    p = np.clip(rows, 1e-12, 1.0)
    h = -(rows * np.log(p)).sum(axis=-1)               # (L, H, T')
    if normalized:
        h_max = np.log(positions + 1.0)                # (T',)
        h_max[h_max <= 0] = 1.0                        # position 0 -> avoid 0/0
        h = h / h_max
    return h.mean(axis=-1)


def mean_attention_distance(attn: np.ndarray, skip: int = 16,
                            relative: bool = False) -> np.ndarray:
    """Mean number of tokens each head looks back.

    For query ``i`` the distance is ``sum_j p_ij * (i - j)``, the expected
    backward offset of the key under the head's own distribution. Averaging
    over ``i >= skip`` gives one number per head.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Leading query positions to exclude.
        relative: Divide each position's distance by ``i``, the maximum
            distance available there, giving a value in ``[0, 1]``.

    Returns:
        ``(L, H)`` array; tokens when ``relative`` is False, fraction otherwise.
    """
    rows, positions = _valid_rows(attn, skip)
    T = attn.shape[-1]
    keys = np.arange(T)                                       # (T,)
    offset = positions[:, None] - keys[None, :]               # (T', T)
    d = (rows * offset[None, None, :, :]).sum(axis=-1)        # (L, H, T')
    if relative:
        denom = np.maximum(positions, 1).astype(np.float64)
        d = d / denom
    return d.mean(axis=-1)


def first_token_attention(attn: np.ndarray, skip: int = 16) -> np.ndarray:
    """Mean probability mass each head places on token position 0.

    High values identify "attention sink" heads that park surplus mass on the
    first token rather than attending to content. See the module docstring.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Leading query positions to exclude.

    Returns:
        ``(L, H)`` array of mean sink mass in ``[0, 1]``.
    """
    rows, _ = _valid_rows(attn, skip)
    return rows[:, :, :, 0].mean(axis=-1)


def previous_token_attention(attn: np.ndarray, skip: int = 16) -> np.ndarray:
    """Mean probability mass each head places on the immediately previous token.

    A head with high mass on ``j = i - 1`` is a positional / n-gram head; these
    appear early in training and typically live in the lower layers.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Leading query positions to exclude.

    Returns:
        ``(L, H)`` array in ``[0, 1]``.
    """
    rows, positions = _valid_rows(attn, skip)
    prev = positions - 1
    take = rows[:, :, np.arange(len(positions)), prev]         # (L, H, T')
    return take.mean(axis=-1)


def summarize(attn: np.ndarray, skip: int = 16) -> dict:
    """Compute every per-head statistic and reduce them to per-layer means.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        skip: Leading query positions to exclude from averages.

    Returns:
        JSON-serializable dict with ``per_head`` (each an ``L x H`` nested list)
        and ``per_layer`` (each a length-``L`` list).
    """
    stats = {
        "entropy_normalized": entropy_per_head(attn, skip, normalized=True),
        "entropy_nats": entropy_per_head(attn, skip, normalized=False),
        "distance_tokens": mean_attention_distance(attn, skip, relative=False),
        "distance_relative": mean_attention_distance(attn, skip, relative=True),
        "sink_mass_pos0": first_token_attention(attn, skip),
        "previous_token_mass": previous_token_attention(attn, skip),
    }
    return {
        "n_layer": int(attn.shape[0]),
        "n_head": int(attn.shape[1]),
        "seq_len": int(attn.shape[-1]),
        "skip_positions": skip,
        "per_head": {k: np.round(v, 5).tolist() for k, v in stats.items()},
        "per_layer": {k: np.round(v.mean(axis=1), 5).tolist()
                      for k, v in stats.items()},
    }


def classify_heads(summary: dict, local_threshold: float = 0.35,
                   sink_threshold: float = 0.60) -> list[dict]:
    """Label each head local / sink / diffuse, for the report's discussion.

    The thresholds are heuristics, not established constants; they are exposed
    as arguments and recorded in the output so the labelling is reproducible
    and arguable rather than hidden.

    Args:
        summary: Output of :func:`summarize`.
        local_threshold: Relative distance below which a head is "local".
        sink_threshold: Sink mass above which a head is called "sink".

    Returns:
        One dict per head with its layer, head index, statistics and label.
    """
    ph = summary["per_head"]
    out = []
    for l in range(summary["n_layer"]):
        for h in range(summary["n_head"]):
            sink = ph["sink_mass_pos0"][l][h]
            rel = ph["distance_relative"][l][h]
            ent = ph["entropy_normalized"][l][h]
            if sink >= sink_threshold:
                label = "sink"          # idle: mass parked on position 0
            elif rel <= local_threshold:
                label = "local"         # short-range / positional
            elif ent >= 0.70:
                label = "diffuse"       # near-uniform over its context
            else:
                label = "content"       # selective and long-range
            out.append({"layer": l, "head": h, "label": label,
                        "entropy_normalized": ent, "distance_relative": rel,
                        "distance_tokens": ph["distance_tokens"][l][h],
                        "sink_mass_pos0": sink,
                        "previous_token_mass": ph["previous_token_mass"][l][h]})
    return out


# ----------------------------------------------------------------- visualization
# Kept strictly separate from the computation above (Phase 2 guideline 4).

def plot_head_heatmaps(attn: np.ndarray, layers: list[int], heads: list[int],
                       out_path: Path, language: str, max_positions: int = 96,
                       gamma: float = 0.45) -> None:
    """Grid of attention heatmaps: one row per layer, one column per head.

    THREE CHOICES THAT MATTER FOR HONESTY AND LEGIBILITY
    ----------------------------------------------------
    *One shared colour scale.* Every panel is normalized identically and the
    single colour bar therefore describes all of them. Per-panel autoscaling
    with a shared bar -- the obvious first implementation -- makes the bar a
    lie for every panel but one, and invites the reader to compare intensities
    that are not on the same scale.

    *A power-law scale rather than a linear one.* Attention mass concentrates:
    a few cells hold most of the probability and the rest sit near zero, so on
    a linear scale everything except the sink column and the diagonal washes
    out to background. ``PowerNorm(gamma=0.45)`` lifts the low end while
    preserving the ordering, which is what makes the faint vertical stripes --
    content heads locking onto specific keys -- visible at all.

    *The masked future is grey, not dark.* Rendering the forbidden upper
    triangle as NaN on a colormap with ``set_bad`` distinguishes "this position
    could not be attended to" from "this position was attended to with
    probability ~0". Those are different statements and a dark-on-dark
    rendering conflates them.

    Args:
        attn: ``(L, H, T, T)`` attention array.
        layers: Layer indices to show; an early and a late one at minimum.
        heads: Head indices to show.
        out_path: PNG destination.
        language: Language name, for the figure title.
        max_positions: Crop to the leading ``T x T`` corner. A full 512x512 map
            has 262,144 cells and renders as an unreadable smear at any
            printable size.
        gamma: Exponent of the power-law colour scale; lower brightens the low
            end further.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    T = min(max_positions, attn.shape[-1])
    n_rows, n_cols = len(layers), len(heads)
    shown = np.stack([[attn[l][h][:T, :T] for h in heads] for l in layers])

    # Percentile over the allowed (lower-triangular) cells only: including the
    # structural zeros above the diagonal would drag every percentile down.
    tri = np.tril(np.ones((T, T), dtype=bool))
    vmax = float(np.percentile(shown[..., tri], 99.0))
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1e-6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#E8E8E8")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.35 * n_cols + 1.9, 2.35 * n_rows + 1.15),
                             squeeze=False, layout="constrained")
    im = None
    for r, l in enumerate(layers):
        for c, h in enumerate(heads):
            ax = axes[r][c]
            block = np.where(tri, shown[r, c], np.nan)
            im = ax.imshow(block, cmap=cmap, norm=norm, aspect="equal",
                           origin="upper", interpolation="nearest")
            # Label only the outer edge: 18 copies of the same axis label is
            # noise, and at this panel size they overlap the row below.
            if r == 0:
                ax.set_title(f"head {h}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {l}\n\nquery position $i$", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("key position $j$", fontsize=9)
            ax.set_xticks([0, T // 2, T - 1])
            ax.set_yticks([0, T // 2, T - 1])
            ax.tick_params(labelsize=7)
            if c:
                ax.set_yticklabels([])
            if r != n_rows - 1:
                ax.set_xticklabels([])

    fig.suptitle(f"Causal self-attention weights \u2014 {language}  "
                 f"(first {T} positions; grey = masked future)", fontsize=13)
    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.015)
    cbar.set_label(f"attention probability $p_{{ij}}$  (power scale, "
                   f"$\\gamma$={gamma}; shared across all panels)", fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_layer_head_matrix(values: np.ndarray, out_path: Path, title: str,
                           cbar_label: str, cmap: str = "magma",
                           annotate: bool = True) -> None:
    """Heatmap of one statistic across every (layer, head) pair.

    Args:
        values: ``(L, H)`` array.
        out_path: PNG destination.
        title: Figure title.
        cbar_label: Label for the colour bar.
        cmap: Matplotlib colormap name.
        annotate: Print each cell's value on the cell.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, H = values.shape
    fig, ax = plt.subplots(figsize=(1.05 * H + 3.0, 0.46 * L + 2.2))
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("head index", fontsize=10)
    ax.set_ylabel("layer index (0 = closest to input)", fontsize=10)
    ax.set_xticks(range(H))
    ax.set_yticks(range(L))
    if annotate:
        lo, hi = float(values.min()), float(values.max())
        mid = (lo + hi) / 2.0
        for l in range(L):
            for h in range(H):
                v = values[l, h]
                ax.text(h, l, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < mid else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def plot_layer_profiles(summary: dict, out_path: Path, language: str) -> None:
    """Per-layer means of entropy, relative distance and sink mass.

    One line per statistic against layer depth, which is the figure that
    actually answers "do lower layers look local and upper layers global?".

    Args:
        summary: Output of :func:`summarize`.
        out_path: PNG destination.
        language: Language name, for the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pl = summary["per_layer"]
    layers = list(range(summary["n_layer"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(layers, pl["entropy_normalized"], marker="o",
            label="normalized entropy  $H_i / \\ln(i{+}1)$")
    ax.plot(layers, pl["distance_relative"], marker="s",
            label="relative attention distance")
    ax.plot(layers, pl["sink_mass_pos0"], marker="^",
            label="mass on position 0 (sink)")
    ax.plot(layers, pl["previous_token_mass"], marker="d",
            label="mass on previous token")
    ax.set_title(f"Attention behaviour by depth — {language}", fontsize=12)
    ax.set_xlabel("layer index (0 = closest to input)", fontsize=10)
    ax.set_ylabel("value (all quantities are in [0, 1])", fontsize=10)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path)


def write_token_table(token_ids: np.ndarray, sp, out_path: Path,
                      language: str, max_positions: int) -> str:
    """Write the example sentence and its per-position tokens as Markdown.

    Figure axes carry positions rather than token strings because Matplotlib
    cannot render Telugu or Devanagari; this file supplies the mapping so a
    reader can interpret any cell of the heatmap.

    Args:
        token_ids: The token ids fed to the model.
        sp: SentencePieceProcessor for this language.
        out_path: Markdown destination.
        language: Language name.
        max_positions: How many positions the heatmap shows.

    Returns:
        The decoded example sentence.
    """
    ids = token_ids[:max_positions].tolist()
    sentence = sp.decode(token_ids.tolist())
    pieces = [sp.id_to_piece(int(i)) for i in ids]

    lines = [f"# Attention example — {language}", "",
             "Heatmap axes are labelled with token **positions**; this table maps",
             "each position to its token. Matplotlib ships no Telugu or Devanagari",
             "glyphs, so token strings cannot be drawn on the figure axes.", "",
             "## Example passage (held-out test split)", "",
             "> " + sentence.replace("\n", " ")[:1200], "",
             f"## Positions 0–{len(ids) - 1}", "",
             "| pos | token | pos | token | pos | token | pos | token |",
             "|----:|:------|----:|:------|----:|:------|----:|:------|"]
    # SentencePiece marks a word boundary with U+2581; show it as a visible dot.
    show = [p.replace("▁", "·") for p in pieces]
    for r in range(0, len(show), 4):
        row = show[r:r + 4]
        cells = "".join(f" {r + k} | `{t}` |" for k, t in enumerate(row))
        lines.append("|" + cells + " | |" * (4 - len(row)))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("  wrote %s", out_path)
    return sentence


# ------------------------------------------------------------------------- cli

def analyze(config: dict, ckpt_path: Path, device: str, out_dir: Path,
            fig_dir: Path, split: str = "test", n_examples: int = 3,
            positions: int = 96, skip: int = 16, seed: int | None = None,
            name: str | None = None) -> dict:
    """Run the whole attention analysis for one checkpoint.

    Statistics are averaged over ``n_examples`` independent held-out passages
    so that no conclusion rests on a single sentence.

    Args:
        config: Parsed model config.
        ckpt_path: Checkpoint to analyze.
        device: Device string.
        out_dir: Directory for the JSON summary and token tables.
        fig_dir: Directory for the PNG figures.
        split: Which held-out split to draw passages from.
        n_examples: Number of passages to average over.
        positions: Sequence length to analyze and plot.
        skip: Leading query positions excluded from averages.
        seed: Passage-selection seed; defaults to the config's train seed.
        name: Stem for every output file. Defaults to the config's ``language``,
            but MUST be set to the run name whenever two runs share a language.
            The no-positional-embedding ablation declares ``language: telugu``
            by design -- it is the same corpus and tokenizer, which is what
            makes it a controlled comparison -- so without this it would
            silently overwrite Model H's attention results and figures.

    Returns:
        The JSON-serializable summary that was written to disk.
    """
    import sentencepiece as spm

    language = config.get("language", "model")
    stem = name or language
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = config["train"].get("dtype", "float16")
    if device_type == "cpu":
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    model, ck = load_model_from_checkpoint(ckpt_path, device)
    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])
    ds = TokenDataset(config["paths"][f"{split}_bin"],
                      config["model"]["block_size"], device,
                      seed=config["train"]["seed"])

    rng = np.random.default_rng(seed if seed is not None
                                else config["train"]["seed"])
    starts = rng.integers(0, len(ds) - positions, size=n_examples)

    per_example, first_attn = [], None
    for k, s in enumerate(starts):
        ids = np.asarray(ds.data[s:s + positions])
        attn = collect_attention(model, ids, device, ctx)
        per_example.append(summarize(attn, skip))
        if k == 0:
            first_attn = attn
            sentence = write_token_table(
                ids, sp, out_dir / f"{stem}_attention_tokens.md",
                stem, positions)

    # Average each statistic across the example passages.
    L, H = per_example[0]["n_layer"], per_example[0]["n_head"]
    keys = list(per_example[0]["per_head"].keys())
    mean_head = {k: np.mean([np.array(e["per_head"][k]) for e in per_example],
                            axis=0) for k in keys}

    summary = {
        "run": stem,
        "language": language,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ck.get("step"),
        "split": split,
        "n_examples": n_examples,
        "positions_analyzed": positions,
        "skip_positions": skip,
        "n_layer": L, "n_head": H,
        "example_sentence": sentence,
        "per_head": {k: np.round(v, 5).tolist() for k, v in mean_head.items()},
        "per_layer": {k: np.round(v.mean(axis=1), 5).tolist()
                      for k, v in mean_head.items()},
    }
    summary["head_labels"] = classify_heads(summary)
    counts: dict[str, int] = {}
    for h in summary["head_labels"]:
        counts[h["label"]] = counts.get(h["label"], 0) + 1
    summary["label_counts"] = counts

    # Figures. An early and a late layer are always included, per the spec.
    early, late = 0, L - 1
    mid = L // 2
    heads = list(range(min(H, 6)))
    plot_head_heatmaps(first_attn, [early, mid, late], heads,
                       fig_dir / f"{stem}_attention_heatmaps.png",
                       stem, positions)
    plot_layer_head_matrix(
        mean_head["entropy_normalized"],
        fig_dir / f"{stem}_attention_entropy.png",
        f"Normalized attention entropy per head — {stem}",
        "$H_i / \\ln(i{+}1)$   (1 = uniform, 0 = peaked)", cmap="magma")
    plot_layer_head_matrix(
        mean_head["distance_tokens"],
        fig_dir / f"{stem}_attention_distance.png",
        f"Mean attention distance per head — {stem}",
        "expected backward offset (tokens)", cmap="viridis")
    plot_layer_profiles(summary, fig_dir / f"{stem}_attention_profile.png",
                        stem)

    out_json = out_dir / f"{stem}_attention.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("  wrote %s", out_json)
    return summary


def main() -> None:
    """CLI: analyze one checkpoint's attention and write figures plus JSON."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Defaults to <out_dir>/best.pt.")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--positions", type=int, default=96,
                   help="Sequence length analyzed and plotted.")
    p.add_argument("--skip", type=int, default=16,
                   help="Leading query positions excluded from averages.")
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase2/stats"))
    p.add_argument("--fig-dir", type=Path, default=Path("report/phase2/figures"))
    p.add_argument("--name", default=None,
                   help="Output filename stem; defaults to the config language. "
                        "Set it when two runs share a language (the ablation).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = args.checkpoint or Path(config["paths"]["out_dir"]) / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[%s] attention analysis on %s (%d passages x %d positions)",
                config.get("language"), ckpt.name, args.n_examples,
                args.positions)
    s = analyze(config, ckpt, device, args.stats_dir, args.fig_dir,
                args.split, args.n_examples, args.positions, args.skip,
                name=args.name)

    print(f"\n  head types: {s['label_counts']}")
    print(f"  {'layer':>5}  {'entropy':>8}  {'dist(tok)':>10}  "
          f"{'dist(rel)':>10}  {'sink':>6}  {'prev-tok':>8}")
    pl = s["per_layer"]
    for l in range(s["n_layer"]):
        print(f"  {l:>5}  {pl['entropy_normalized'][l]:>8.3f}  "
              f"{pl['distance_tokens'][l]:>10.2f}  "
              f"{pl['distance_relative'][l]:>10.3f}  "
              f"{pl['sink_mass_pos0'][l]:>6.3f}  "
              f"{pl['previous_token_mass'][l]:>8.3f}")


if __name__ == "__main__":
    main()

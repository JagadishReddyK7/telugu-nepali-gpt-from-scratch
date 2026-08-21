"""
Token-frequency statistics and tokenization examples, per language.

    python scripts/token_stats.py --lang telugu
    python scripts/token_stats.py --lang nepali

Phase 1 §1.3 asks for five things per language: vocabulary size, token-frequency
statistics, average characters per token, tokenization examples, and
unknown-token statistics. The sweep and the encoder already produce the first,
third and fifth. This script produces the other two.

WHY IT READS train.bin RATHER THAN THE CORPUS
---------------------------------------------
Frequencies must describe the tokens the model will actually be trained on, so
they are counted over the encoded ``train.bin`` -- the exact uint16 stream Phase
2 samples from -- not over a re-tokenization of the text, which could drift from
it if the tokenizer were ever swapped without re-encoding. ``np.bincount`` over
the whole array is a single pass with no Python-level loop.

Examples are decoded back out of the same array, so what is shown is
demonstrably what was encoded rather than a fresh tokenization of a sentence
chosen for looking good.

WHAT THE COVERAGE CURVE IS FOR
------------------------------
A vocabulary is not used uniformly. Reporting that the top 100 pieces cover some
large fraction of all occurrences says how much of the model's embedding table
does most of the work, and the count of never-used pieces says how much of the
parameter budget bought nothing at all. For a model where the embedding is ~25%
of total parameters (§6.1 of the report), that is a design-relevant number, not
a curiosity.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOP_N_REPORTED = 30
COVERAGE_POINTS = (10, 100, 500, 1_000, 5_000, 10_000)


def load_counts(bin_path: Path, vocab_size: int):
    """Count occurrences of every token id in an encoded split.

    Args:
        bin_path: ``.bin`` file of uint16 token ids.
        vocab_size: Size of the tokenizer vocabulary.

    Returns:
        A numpy array of length ``vocab_size`` holding occurrence counts.
    """
    import numpy as np

    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    logger.info("counting %s tokens in %s ...", f"{arr.size:,}", bin_path.name)
    return np.bincount(arr, minlength=vocab_size).astype("int64")


def decode_examples(bin_path: Path, sp, n_examples: int, eos_id: int,
                    skip_tokens: int = 5_000_000):
    """Recover whole documents from the encoded stream and show their pieces.

    Args:
        bin_path: ``.bin`` file of uint16 token ids.
        sp: Loaded SentencePieceProcessor.
        n_examples: How many documents to return.
        eos_id: End-of-document token id, used to find document boundaries.
        skip_tokens: Offset into the stream, so the examples are not all drawn
            from whichever source happens to sort first.

    Returns:
        A list of dicts with the detokenized text, its pieces, and fertility.
    """
    import numpy as np

    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    start = min(skip_tokens, max(arr.size - 200_000, 0))
    window = np.asarray(arr[start:start + 200_000])

    bounds = np.flatnonzero(window == eos_id)
    examples = []
    # Real documents run to a few hundred tokens, so do not filter on length --
    # take whole documents and truncate only what is DISPLAYED. Fertility is
    # still computed over the entire document, not over the excerpt, so the
    # number quoted beside an example describes the document it came from.
    for i in range(len(bounds) - 1):
        if len(examples) >= n_examples:
            break
        ids = [int(t) for t in window[bounds[i] + 1:bounds[i + 1]]
               if int(t) not in (sp.bos_id(), sp.eos_id())]
        if len(ids) < 25:
            continue
        text = sp.decode(ids)
        if len(text) < 60:
            continue
        n_words = len(text.split())
        excerpt_ids = ids[:40]
        examples.append({
            "text": text[:400],
            "text_truncated": len(text) > 400,
            "excerpt_pieces": [sp.id_to_piece(t) for t in excerpt_ids],
            "excerpt_text": sp.decode(excerpt_ids),
            "n_words": n_words,
            "n_tokens": len(ids),
            "fertility": round(len(ids) / max(n_words, 1), 3),
        })
    return examples


def main() -> None:
    """Compute frequency statistics and examples for one language."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["telugu", "nepali"])
    p.add_argument("--examples", type=int, default=3)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import numpy as np
    import sentencepiece as spm

    lang = args.lang
    root = Path(f"data/{lang}")
    sweep = json.loads((root / "tokenizer" / "tokenizer_sweep.json")
                       .read_text(encoding="utf-8"))
    vocab = sweep["chosen_vocab_size"]
    model_path = Path(f"{lang}/tokenizer") / sweep["chosen_model"]
    bin_path = root / "tokenized" / "train.bin"

    for path in (model_path, bin_path):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    counts = load_counts(bin_path, sp.get_piece_size())
    total = int(counts.sum())

    order = np.argsort(-counts)
    nonzero = int((counts > 0).sum())
    unused = int(sp.get_piece_size() - nonzero)
    singletons = int((counts == 1).sum())

    coverage = {}
    cum = np.cumsum(counts[order])
    for n in COVERAGE_POINTS:
        if n <= len(cum):
            coverage[n] = round(float(cum[n - 1]) / total, 6)

    unk_id = sp.unk_id()
    unk_count = int(counts[unk_id]) if 0 <= unk_id < len(counts) else 0

    top = []
    for rank, tid in enumerate(order[:TOP_N_REPORTED], start=1):
        tid = int(tid)
        top.append({
            "rank": rank,
            "id": tid,
            "piece": sp.id_to_piece(tid),
            "count": int(counts[tid]),
            "share": round(float(counts[tid]) / total, 6),
        })

    examples = decode_examples(bin_path, sp, args.examples, sp.eos_id())

    stats = {
        "language": lang,
        "tokenizer_model": model_path.name,
        "vocab_size": int(sp.get_piece_size()),
        "configured_vocab_size": vocab,
        "train_tokens_counted": total,
        "distinct_pieces_used": nonzero,
        "pieces_never_used": unused,
        "pieces_never_used_share": round(unused / sp.get_piece_size(), 6),
        "pieces_used_once": singletons,
        "unk_id": unk_id,
        "unk_occurrences": unk_count,
        "unk_rate": round(unk_count / total, 12),
        "coverage_by_top_n_pieces": coverage,
        "top_pieces": top,
        "examples": examples,
    }

    out_dir = Path("report/phase1/stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{lang}_token_frequency.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = np.arange(1, nonzero + 1)
    freqs = counts[order][:nonzero]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.loglog(ranks, freqs, label=f"{lang.title()} (vocab {vocab:,})")
    ax1.set_title(f"{lang.title()}: token frequency by rank (Zipf)")
    ax1.set_xlabel("Rank of token piece (log scale)")
    ax1.set_ylabel("Occurrences in train split (log scale)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    cov_x = list(coverage)
    cov_y = [coverage[n] * 100 for n in cov_x]
    ax2.semilogx(cov_x, cov_y, marker="o", label="Cumulative coverage")
    for xv, yv in zip(cov_x, cov_y):
        ax2.annotate(f"{yv:.1f}%", (xv, yv), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_title(f"{lang.title()}: corpus coverage by most frequent pieces")
    ax2.set_xlabel("Number of most-frequent pieces (log scale)")
    ax2.set_ylabel("Share of all token occurrences (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    figdir = Path("report/phase1/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    figpath = figdir / f"{lang}_token_frequency.png"
    fig.savefig(figpath, dpi=150)
    plt.close(fig)

    # ---- console summary, formatted for pasting into the report ---------
    print()
    print(f"  {lang.upper()}  vocab {sp.get_piece_size():,} | "
          f"{total:,} training tokens counted")
    print(f"  pieces never used : {unused:,} ({unused/sp.get_piece_size():.2%})")
    print(f"  pieces used once  : {singletons:,}")
    print(f"  UNK occurrences   : {unk_count:,} (rate {unk_count/total:.3e})")
    print()
    print("  coverage by most frequent pieces")
    for n, c in coverage.items():
        print(f"    top {n:>6,} pieces -> {c:.2%} of all occurrences")
    print()
    print(f"  top {TOP_N_REPORTED} pieces")
    print(f"    {'rank':>4}  {'piece':<18} {'count':>14}  share")
    for t in top:
        print(f"    {t['rank']:>4}  {t['piece']:<18} {t['count']:>14,}  "
              f"{t['share']:.3%}")
    print()
    for i, ex in enumerate(examples, 1):
        print(f"  example {i}: whole document = {ex['n_words']} words -> "
              f"{ex['n_tokens']} tokens (fertility {ex['fertility']})")
        print(f"    excerpt : {ex['excerpt_text']}")
        print(f"    pieces  : {' '.join(ex['excerpt_pieces'])}")
        print()
    print(f"  wrote {out}")
    print(f"  wrote {figpath}")
    print()


if __name__ == "__main__":
    main()

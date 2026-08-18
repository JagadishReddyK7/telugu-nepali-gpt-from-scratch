"""
Encode the corpora, then generate every Phase 1 table, figure, and the report.

Run this last, after cleaning and tokenizer training:

    python scripts/build_phase1_report.py

It produces, for both languages:
  * exact token counts (by encoding the corpus, not estimating)
  * report/phase1/REPORT.md          -- the graded write-up
  * report/phase1/figures/*.png      -- labelled figures
  * data/<lang>/tokenized/*.bin      -- uint16 arrays for Phase 2

Every figure carries a title, axis labels and a legend, because the spec zeroes
figures that omit them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.encode_corpus import encode_corpus  # noqa: E402
from common.run_crawl import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

LANGS = ("telugu", "nepali")
FIGDIR = Path("report/phase1/figures")


def load_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict if it is missing."""
    if not path.exists():
        logger.warning("missing %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("could not parse %s", path)
        return {}


# ---------------------------------------------------------------- figures

def fig_cleaning_funnel(stats: dict[str, dict]) -> Path:
    """Plot how many documents each cleaning stage removed, per language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    reasons = sorted({r for s in stats.values()
                      for r in s.get("rejected_by_reason", {})})
    if not reasons:
        return None
    x = np.arange(len(reasons))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, (lang, s) in enumerate(stats.items()):
        read = max(s.get("documents_read", 1), 1)
        vals = [s.get("rejected_by_reason", {}).get(r, 0) / read * 100
                for r in reasons]
        ax.bar(x + (i - 0.5) * width, vals, width, label=lang.title())

    ax.set_title("Phase 1: documents rejected at each cleaning stage")
    ax.set_xlabel("Rejection reason")
    ax.set_ylabel("Share of documents read (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ") for r in reasons],
                       rotation=35, ha="right", fontsize=8)
    ax.legend(title="Language")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "cleaning_funnel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_provenance(stats: dict[str, dict]) -> Path:
    """Plot the manual vs downloaded character split against the 20% rule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    langs = list(stats)
    manual = [stats[l].get("chars_per_provenance", {}).get("manual_scrape", 0) / 1e6
              for l in langs]
    downloaded = [stats[l].get("chars_per_provenance", {}).get("downloaded", 0) / 1e6
                  for l in langs]
    x = np.arange(len(langs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.bar(x, manual, 0.5, label="Manual (scraped)")
    ax1.bar(x, downloaded, 0.5, bottom=manual, label="Downloaded (public corpora)")
    ax1.set_title("Corpus composition by provenance")
    ax1.set_xlabel("Language")
    ax1.set_ylabel("Characters (millions)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.title() for l in langs])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    frac = [stats[l].get("manual_char_fraction", 0) * 100 for l in langs]
    colors = ["seagreen" if f >= 20 else "indianred" for f in frac]
    ax2.bar(x, frac, 0.5, color=colors, label="Manual share achieved")
    ax2.axhline(20, color="black", linestyle="--",
                label="Requirement (>=20%)")
    ax2.set_title("Manual-collection share vs the project requirement")
    ax2.set_xlabel("Language")
    ax2.set_ylabel("Manual share of characters (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.title() for l in langs])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGDIR / "provenance_split.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_sources(stats: dict[str, dict]) -> Path:
    """Plot the document count per source, per language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stats), figsize=(6 * len(stats), 5))
    if len(stats) == 1:
        axes = [axes]
    for ax, (lang, s) in zip(axes, stats.items()):
        per = s.get("documents_per_source", {})
        items = sorted(per.items(), key=lambda kv: -kv[1])[:14]
        if not items:
            continue
        names = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax.barh(names, vals, label=f"{lang.title()} sources")
        ax.set_title(f"{lang.title()}: documents kept per source")
        ax.set_xlabel("Documents after cleaning")
        ax.set_ylabel("Source")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "documents_per_source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------- report

def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(clean: dict, tok: dict, counts: dict, figs: list) -> str:
    """Assemble the Phase 1 report markdown.

    Args:
        clean: Per-language cleaning statistics.
        tok: Per-language tokenizer sweep reports.
        counts: Per-language exact token counts.
        figs: Figure paths to reference.

    Returns:
        The report as a markdown string.
    """
    L = [l for l in LANGS if l in clean]

    def g(lang, *keys, default=0):
        d = clean.get(lang, {})
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if d not in ({}, None) else default

    lines = [
        "# Phase 1 — Data Collection and Tokenizer Construction",
        "",
        "Model H: **Telugu** (higher-resource) · Model L: **Nepali** "
        "(lower-resource, from the permitted list).",
        "",
        "The two corpora share no document, no tokenizer, and no vocabulary. "
        "Pipeline *code* is shared; all *data* is separate by construction, "
        "and the statistics below are computed independently per language.",
        "",
        "## 1. Corpus size",
        "",
    ]

    rows = []
    for lang in L:
        c = counts.get(lang, {})
        rows.append([
            lang.title(),
            f"{clean[lang].get('documents_kept', 0):,}",
            f"{clean[lang].get('chars_kept', 0):,}",
            f"{c.get('total_tokens', 0):,}",
            f"{c.get('overall_chars_per_token', 0):.2f}",
        ])
    lines += [md_table(
        ["Language", "Documents", "Characters", "**Tokens (exact)**",
         "Chars/token"], rows), ""]
    lines += [
        "Token counts are measured by encoding the corpus with the trained "
        "tokenizer, not estimated from a characters-per-token assumption.",
        "",
        "## 2. Provenance and the ≥20% manual requirement",
        "",
    ]

    rows = []
    for lang in L:
        s = clean[lang]
        m = s.get("chars_per_provenance", {}).get("manual_scrape", 0)
        d = s.get("chars_per_provenance", {}).get("downloaded", 0)
        frac = s.get("manual_char_fraction", 0)
        rows.append([lang.title(), f"{m:,}", f"{d:,}", f"{frac:.1%}",
                     "PASS" if frac >= 0.20 else "FAIL"])
    lines += [md_table(
        ["Language", "Manual chars", "Downloaded chars", "Manual share",
         "≥20%?"], rows), ""]
    lines += [
        "The manual share is computed from per-document provenance tags "
        "recorded at collection time, so it is auditable rather than asserted. "
        "Surplus downloaded text is discarded by the cleaning stage to hold the "
        "ratio — a smaller corpus that satisfies the constraint is preferable "
        "to a larger one that violates it.",
        "",
        "![Provenance](figures/provenance_split.png)",
        "",
        "## 3. Cleaning funnel",
        "",
    ]

    rows = []
    for lang in L:
        s = clean[lang]
        rows.append([lang.title(), f"{s.get('documents_read', 0):,}",
                     f"{s.get('documents_kept', 0):,}",
                     f"{s.get('keep_rate', 0):.1%}"])
    lines += [md_table(["Language", "Read", "Kept", "Keep rate"], rows), ""]

    for lang in L:
        rej = clean[lang].get("rejected_by_reason", {})
        if not rej:
            continue
        read = max(clean[lang].get("documents_read", 1), 1)
        lines += [f"**{lang.title()} — rejections by reason**", "",
                  md_table(["Reason", "Documents", "% of read"],
                           [[k.replace('_', ' '), f"{v:,}", f"{v/read:.2%}"]
                            for k, v in rej.items()]), ""]
    lines += ["![Cleaning funnel](figures/cleaning_funnel.png)", "",
              "![Sources](figures/documents_per_source.png)", "",
              "## 4. Tokenizers", ""]

    for lang in L:
        t = tok.get(lang, {})
        if not t.get("sweep"):
            continue
        lines += [f"### {lang.title()}", "",
                  md_table(
                      ["Vocab", "Fertility (tok/word)", "Chars/token",
                       "UNK rate", "Embedding params"],
                      [[f"{r['vocab_size']:,}", f"{r['fertility']:.4f}",
                        f"{r['chars_per_token']:.3f}", f"{r['unk_rate']:.2e}",
                        f"{r['embedding_params']/1e6:.2f}M"]
                       for r in t["sweep"]]), "",
                  f"**Chosen: {t.get('chosen_vocab_size', 0):,}** "
                  f"(`{t.get('chosen_model', '')}`). Selected as the smallest "
                  f"vocabulary within 3% of the best fertility — vocabulary is "
                  f"bought directly out of the ~25M parameter budget, since the "
                  f"embedding matrix is tied to the output head.", "",
                  f"![{lang} tokenizer](figures/{lang}_tokenizer_sweep.png)", ""]
        ex = (t.get("examples") or [{}])[0]
        if ex.get("pieces"):
            lines += ["Tokenization example:", "", "```",
                      ex["text"][:150], "->", " ".join(ex["pieces"][:30]),
                      f"({ex['n_words']} words -> {ex['n_tokens']} tokens, "
                      f"fertility {ex['fertility']})", "```", ""]

    lines += ["## 5. Splits", ""]
    rows = []
    for lang in L:
        c = counts.get(lang, {}).get("splits", {})
        for split in ("train", "validation", "test"):
            s = c.get(split, {})
            if s:
                rows.append([lang.title(), split, f"{s['documents']:,}",
                             f"{s['tokens']:,}", f"{s['utf8_bytes']:,}"])
    lines += [md_table(["Language", "Split", "Documents", "Tokens",
                        "UTF-8 bytes"], rows), "",
              "Splits are assigned per **document** by hashing the document id, "
              "never per line — a line-level split would place sentences from "
              "one article in both train and test. Deduplication runs *before* "
              "splitting, so a document and its near-duplicate cannot straddle "
              "the boundary and leak test data into training.", "",
              "UTF-8 byte counts are recorded because Phase 2 reports "
              "bits-per-byte, which is the only fair way to compare two models "
              "whose tokenizers compress at different rates.", ""]
    return "\n".join(lines)


def main() -> None:
    """Encode both corpora and build the Phase 1 report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-encode", action="store_true",
                   help="Reuse existing token_counts.json instead of re-encoding.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    clean, tok, counts = {}, {}, {}
    for lang in LANGS:
        root = Path(f"data/{lang}")
        cs = load_json(root / "clean" / "clean_stats.json")
        if not cs:
            logger.warning("[%s] no cleaning stats; run the cleaning stage first",
                           lang)
            continue
        clean[lang] = cs

        ts = load_json(root / "tokenizer" / "tokenizer_sweep.json")
        if ts:
            tok[lang] = ts

        cnt_path = root / "tokenized" / "token_counts.json"
        if args.skip_encode and cnt_path.exists():
            counts[lang] = load_json(cnt_path)
        elif ts.get("chosen_model"):
            model = root / "tokenizer" / ts["chosen_model"]
            logger.info("[%s] encoding corpus with %s ...", lang, model.name)
            counts[lang] = encode_corpus(lang, root, model)
        else:
            logger.warning("[%s] no tokenizer; skipping encode", lang)

    if not clean:
        raise SystemExit("Nothing to report. Run cleaning first.")

    figs = [f for f in (fig_provenance(clean), fig_cleaning_funnel(clean),
                        fig_sources(clean)) if f]
    report = build_report(clean, tok, counts, figs)
    out = Path("report/phase1/REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    for lang in clean:
        c = counts.get(lang, {})
        s = clean[lang]
        print(f"  {lang.upper():8s} {s.get('documents_kept',0):>10,} docs  "
              f"{c.get('total_tokens',0):>13,} tokens  "
              f"manual {s.get('manual_char_fraction',0):.1%}")
    print("=" * 70)
    print(f"  report : {out}")
    for f in figs:
        print(f"  figure : {f}")
    print()


if __name__ == "__main__":
    main()

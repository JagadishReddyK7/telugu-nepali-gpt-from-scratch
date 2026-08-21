"""
Download the public-corpus share of the Nepali (Model L) dataset.

Run this on Colab. Mirrors the Telugu downloader but with its own sources,
budgets, and output directory -- no data crosses between the two languages.

    python -m nepali.collect.download_corpora
    python -m nepali.collect.download_corpora --scale 0.02   # smoke test

Output lands in ``data/nepali/raw/*.jsonl`` with ``provenance="downloaded"``.

BUDGET REASONING
----------------
SUPERSEDED PLAN, KEPT FOR THE RECORD. This file previously argued for a ~250M
token Nepali target. The reasoning: the news crawl yielded only ~50M manual
tokens, because most Nepali news sites either blocked the crawler (403/429) or
published no usable sitemap, and a 50M manual share caps the corpus at 250M
tokens under the spec's >=20% manual rule.

The reasoning was sound but its premise expired. Widening the archive.org
search past the ``language:("Nepali")`` metadata field -- to the ``nep`` alias,
then to subject and collection queries that reach volumes whose language field
is blank -- raised manual collection from ~50M to 614M characters of books, on
top of ~51M from news. The manual share is no longer the constraint it was, so
the 250M target is withdrawn and Nepali is aimed at the same ~500M as Telugu.

CURRENT SIZING
--------------
The binding constraint is still the 20% rule, which is a CEILING as much as a
floor: ``total_tokens <= manual_tokens / 0.20``. The cleaner enforces it as a
character-ratio cap (``max_downloaded_to_manual_ratio: 5.4``), discarding
surplus downloaded text rather than using it. So these budgets are sized to sit
just past the point where that cap stops binding, and no further -- anything
beyond it is downloaded, cleaned, and then thrown away.

Working from measured constants (3.33 chars/token manual, 4.67 downloaded,
~51% manual and ~87.5% downloaded survival through cleaning):

    manual      740M raw chars ->   377M clean ->  113M manual tokens
    downloaded 2,230M raw chars -> 1,951M clean ->  418M downloaded tokens
                                          total ~531M tokens, 21.3% manual

740M raw manual characters is the number that matters: at that point the ratio
cap stops binding at any setting between 5.0 and 5.4, so the corpus size stops
depending on a threshold that was picked by judgement. Collection stood at 666M
when these budgets were set, so the manual side still needs roughly +74M
characters. Run ``scripts/check_target.py --lang nepali`` for the live version
of this arithmetic against what is actually on disk.

Do not raise ``iriisnepal_corpus`` further without confirming the manual side
kept pace. Past the cap, extra downloaded characters cost Colab download time
and buy exactly zero tokens.

THE HINDI PROBLEM
-----------------
Every source here is Devanagari, and none of them guarantees Nepali rather than
Hindi. CC-100 and OSCAR-derived corpora in particular are known to contain
misfiled documents. Ingest applies only a loose script filter; the real
separation happens in the cleaning stage via fastText language ID, and the
rejection counts are reported. Do not assume a dataset labelled "Nepali" is
monolingual Nepali.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hf_ingest import CorpusIngestor, SourceSpec, resolve_wikipedia_config
from common.run_crawl import setup_logging

BUDGETS = {
    # Raised 350M -> 1.25B. This is the only source with real headroom (6.4M
    # docs / 27.5GB), so it absorbs the whole increase; the others are already
    # near what they can supply without scraping the barrel for duplicates.
    "iriisnepal_corpus": 1_250_000_000,
    "indiccorp_v2_ne": 600_000_000,
    "nepalitext": 300_000_000,
    "wikipedia_ne": 80_000_000,
}   # total 2,230M raw chars -- see CURRENT SIZING above before changing this


def build_specs(scale: float = 1.0) -> list[SourceSpec]:
    """Construct the Nepali source specifications.

    Args:
        scale: Multiplier applied to every character budget.

    Returns:
        Source specs in priority order.
    """
    specs = [
        SourceSpec(
            name="iriisnepal_corpus",
            hf_dataset="IRIISNEPAL/Nepali-Text-Corpus",
            target_chars=int(BUDGETS["iriisnepal_corpus"] * scale),
            notes="Largest public Nepali corpus (6.4M docs / 27.5GB). "
                  "News, blogs, and web text.",
        ),
        SourceSpec(
            name="indiccorp_v2_ne",
            hf_dataset="ai4bharat/IndicCorpV2",
            # See the Telugu downloader for the full explanation of this layout.
            data_files="data/ne.txt",
            load_kwargs={
                "sample_by": "paragraph",
                # WINDOWS. The text builder otherwise opens the file with the
                # platform default codepage -- cp1252 on a Western Windows
                # install -- and dies on the first Telugu or Devanagari byte
                # with UnicodeDecodeError. Linux and Colab default to UTF-8 and
                # never hit this, which is exactly why it must be explicit.
                "encoding": "utf-8",
                "encoding_errors": "replace",
            },
            min_chars=100,
            target_chars=int(BUDGETS["indiccorp_v2_ne"] * scale),
            notes="AI4Bharat IndicCorpV2 Nepali (~36.5M docs).",
        ),
        SourceSpec(
            name="nepalitext",
            hf_dataset="Sakonii/nepalitext-language-model-dataset",
            target_chars=int(BUDGETS["nepalitext"] * scale),
            notes="Established Nepali LM corpus; adds source diversity.",
        ),
    ]

    wiki_config = resolve_wikipedia_config("ne")
    if wiki_config:
        specs.append(SourceSpec(
            name="wikipedia_ne",
            hf_dataset="wikimedia/wikipedia",
            hf_config=wiki_config,
            target_chars=int(BUDGETS["wikipedia_ne"] * scale),
            max_chars=200_000,
            notes=f"Nepali Wikipedia ({wiki_config}). Encyclopedic register; "
                  f"also the most reliably Nepali-rather-than-Hindi source here.",
        ))
    return specs


def main() -> None:
    """Parse arguments and run the Nepali corpus download."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale all character budgets. Try 0.02 for a smoke test.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    specs = build_specs(args.scale)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            p.error(f"No Nepali sources matched --only={args.only}")

    ingestor = CorpusIngestor(
        language="nepali", script="devanagari", output_root=Path("data/nepali")
    )
    summary = ingestor.ingest_all(specs, resume=not args.no_resume)

    total = sum(v.get("chars_kept", 0) for v in summary.values())
    # 4.67 chars/token measured on downloaded Nepali text at vocab 16,000, and
    # ~87.5% of these characters survive cleaning. The old 3.2 constant here
    # overstated the yield by ~45%.
    print(f"\nNepali downloaded share: {total:,} raw characters "
          f"(~{total*0.875/4.67/1e6:.0f}M tokens after cleaning, "
          f"before the 20% ratio cap)")


if __name__ == "__main__":
    main()

"""
Download the public-corpus share of the Nepali (Model L) dataset.

Run this on Colab. Mirrors the Telugu downloader but with its own sources,
budgets, and output directory -- no data crosses between the two languages.

    python -m nepali.collect.download_corpora
    python -m nepali.collect.download_corpora --scale 0.02   # smoke test

Output lands in ``data/nepali/raw/*.jsonl`` with ``provenance="downloaded"``.

BUDGET REASONING
----------------
The Nepali target is deliberately lower than Telugu's. The crawl showed the
manual share landing near 50M tokens: most Nepali news sites either blocked the
crawler outright (403/429) or published no usable sitemap, leaving one outlet
carrying almost the whole scrape. Since the spec requires at least 20% of final
tokens to be manually collected, a 50M manual share caps the total corpus at
about 250M tokens.

That is not a failure to report apologetically -- it IS the result. Nepali is
the lower-resource model, the spec explicitly permits collecting the maximum
feasible amount with the exact count and a justification, and a 250M vs 500M
split is a cleaner resource-tier contrast than forcing the two to match.

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
    "iriisnepal_corpus": 350_000_000,
    "indiccorp_v2_ne": 200_000_000,
    "nepalitext": 100_000_000,
    "wikipedia_ne": 80_000_000,
}


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
    print(f"\nNepali downloaded share: {total:,} characters "
          f"(~{total/3.2/1e6:.0f}M tokens at 3.2 chars/token)")


if __name__ == "__main__":
    main()

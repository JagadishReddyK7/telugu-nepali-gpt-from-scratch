"""
Download the public-corpus share of the Telugu (Model H) dataset.

Run this on Colab, not locally -- it streams multi-gigabyte datasets and writes
to Colab's ephemeral local disk. Only the tokenized output ever goes to Drive.

    python -m telugu.collect.download_corpora                  # all sources
    python -m telugu.collect.download_corpora --only wikipedia
    python -m telugu.collect.download_corpora --scale 0.5      # half budgets

Output lands in ``data/telugu/raw/*.jsonl`` with ``provenance="downloaded"``,
alongside the scraped news shards. Interrupt and re-run freely: progress is
recorded per source in ``data/telugu/download_progress.json``.

BUDGET REASONING
----------------
Target corpus is ~500M tokens. Measured Telugu article length and a projected
manual yield put the scraped share at roughly 140M tokens (~450M characters),
which comfortably clears the required 20%. The remaining ~360M tokens is about
1.15B characters at Telugu's expected 3.0-3.5 characters per token.

Budgets are set PER SOURCE rather than as one global cap. That is deliberate: a
single global budget would be filled almost entirely by whichever corpus streams
fastest -- in practice IndicCorpV2, which is mostly news and web text. Capping
each source preserves the register mix that Wikipedia and Sangraha contribute,
and register diversity is precisely what a news-only corpus lacks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hf_ingest import CorpusIngestor, SourceSpec, resolve_wikipedia_config
from common.run_crawl import setup_logging

# Characters, not tokens -- we cannot count tokens before the tokenizer exists.
BUDGETS = {
    "indiccorp_v2_te": 700_000_000,
    "sangraha_te": 300_000_000,
    "wikipedia_te": 200_000_000,
}


def build_specs(scale: float = 1.0) -> list[SourceSpec]:
    """Construct the Telugu source specifications.

    Args:
        scale: Multiplier applied to every character budget. Use a value below
            1.0 for a fast trial run before committing to the full download.

    Returns:
        Source specs in priority order.
    """
    specs = [
        SourceSpec(
            name="indiccorp_v2_te",
            hf_dataset="ai4bharat/IndicCorpV2",
            # CONFIRMED LAYOUT: one plain-text file per language, keyed by
            # 2-letter ISO code -- data/te.txt, data/ne.txt, data/en.txt, ...
            # NOT the "tel_Telu" subsets its dataset viewer advertises, and NOT
            # a per-language directory. Pointing at the directory would load
            # EVERY language at once and silently produce a multilingual corpus,
            # violating the spec's hard monolingual constraint.
            data_files="data/te.txt",
            # A .txt repo is read by the text builder, which yields one row per
            # LINE by default -- that shreds documents into fragments, most of
            # which the 200-character floor then discards. "paragraph" groups on
            # blank lines, restoring document-like units.
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
            target_chars=int(BUDGETS["indiccorp_v2_te"] * scale),
            notes="AI4Bharat IndicCorpV2 Telugu (~31M docs). Primary bulk source; "
                  "web and news text.",
        ),
        SourceSpec(
            name="sangraha_te",
            hf_dataset="ai4bharat/sangraha",
            hf_config="verified",
            data_dir="verified/tel",
            target_chars=int(BUDGETS["sangraha_te"] * scale),
            notes="AI4Bharat Sangraha verified Telugu subset. Higher-quality "
                  "filtering than IndicCorpV2.",
        ),
    ]

    # Wikipedia's config name is a dated snapshot, so resolve it at runtime
    # rather than hardcoding a date that will eventually stop existing.
    wiki_config = resolve_wikipedia_config("te")
    if wiki_config:
        specs.append(SourceSpec(
            name="wikipedia_te",
            hf_dataset="wikimedia/wikipedia",
            hf_config=wiki_config,
            target_chars=int(BUDGETS["wikipedia_te"] * scale),
            # Wikipedia articles are long and structured; the listing-page
            # ceiling that applies to scraped news would wrongly reject them.
            max_chars=200_000,
            notes=f"Telugu Wikipedia ({wiki_config}). Encyclopedic register and "
                  f"broad topical coverage absent from news.",
        ))
    return specs


def main() -> None:
    """Parse arguments and run the Telugu corpus download."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated source names to restrict to.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale all character budgets. Try 0.02 for a smoke test.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore saved progress and start each source over.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    specs = build_specs(args.scale)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            p.error(f"No Telugu sources matched --only={args.only}")

    ingestor = CorpusIngestor(
        language="telugu", script="telugu", output_root=Path("data/telugu")
    )
    summary = ingestor.ingest_all(specs, resume=not args.no_resume)

    total = sum(v.get("chars_kept", 0) for v in summary.values())
    print(f"\nTelugu downloaded share: {total:,} characters "
          f"(~{total/3.2/1e6:.0f}M tokens at 3.2 chars/token)")


if __name__ == "__main__":
    main()

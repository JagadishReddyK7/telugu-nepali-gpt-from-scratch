"""
Nepali news crawler -- the manual-collection half of the Model L corpus.

Run this FIRST, in parallel with the Telugu crawler.

    python -m nepali.collect.scrape_news --discover
    python -m nepali.collect.scrape_news --run

Output lands in ``data/nepali/raw/*.jsonl`` with ``provenance="manual_scrape"``.

A NEPALI-SPECIFIC WARNING
-------------------------
Devanagari is shared with Hindi, and several of these outlets syndicate or
translate Indian content. Scraping from a ``.np`` domain does NOT guarantee
Nepali text. The fastText language-ID filter in ``common/langid.py`` is what
actually enforces monolinguality, and it runs in the cleaning stage rather than
here -- we deliberately keep the crawler dumb and the filtering auditable, so
the report can state exactly how many documents were rejected as Hindi.

Expect a non-trivial rejection rate. That number is a genuinely interesting
result and belongs in the Phase 1 report.

SOURCE SELECTION RATIONALE
--------------------------
Nepal's online news ecosystem is large relative to the language's speaker count,
which is the main reason Nepali is the most tractable choice on the assignment's
allowed list. The sites below span broadsheet dailies, digital-native outlets,
and a broadcaster, giving register variety rather than seven copies of the same
wire copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_crawl import NewsSite, build_arg_parser, crawl_sites, setup_logging

NEPALI_SITES: list[NewsSite] = [
    NewsSite(
        name="ekantipur",
        base_url="https://ekantipur.com",
        delay_seconds=1.5,
        notes="Kantipur, Nepal's largest daily. Deep archive, high editorial quality.",
    ),
    NewsSite(
        name="setopati",
        base_url="https://www.setopati.com",
        delay_seconds=1.5,
        notes="Leading digital-native outlet; strong long-form and opinion writing.",
    ),
    NewsSite(
        name="onlinekhabar",
        base_url="https://www.onlinekhabar.com",
        delay_seconds=1.5,
        notes="High-volume digital news portal (WordPress; sitemap_index.xml).",
    ),
    NewsSite(
        name="ratopati",
        base_url="https://www.ratopati.com",
        delay_seconds=2.0,
        notes="Popular news portal; broad general-interest coverage.",
    ),
    NewsSite(
        name="nagariknews",
        base_url="https://nagariknews.nagariknetwork.com",
        delay_seconds=2.0,
        notes="Nagarik daily. Mainstream broadsheet register.",
    ),
    NewsSite(
        name="bbc_nepali",
        base_url="https://www.bbc.com",
        sitemap_urls=["https://www.bbc.com/nepali/sitemap.xml"],
        url_must_contain=["/nepali/"],
        delay_seconds=2.0,
        notes="BBC Nepali. Carefully edited prose; reliably Nepali rather than Hindi.",
    ),
    NewsSite(
        name="himalkhabar",
        base_url="https://www.himalkhabar.com",
        delay_seconds=2.0,
        notes="Himal Khabarpatrika. Magazine-style long-form; distinct register.",
    ),
    NewsSite(
        name="annapurnapost",
        base_url="https://annapurnapost.com",
        delay_seconds=2.0,
        notes="Annapurna Post daily. Additional mainstream source.",
    ),
]


def main() -> None:
    """Parse arguments and run the Nepali crawl."""
    parser = build_arg_parser("Nepali")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not (args.discover or args.run):
        parser.error("Specify at least one of --discover or --run.")

    sites = NEPALI_SITES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sites = [s for s in sites if s.name in wanted]
        if not sites:
            parser.error(f"No Nepali sites matched --only={args.only}")

    crawl_sites(
        sites=sites,
        output_root=Path("data/nepali"),
        do_discover=args.discover,
        do_run=args.run,
        max_docs_per_site=args.max_docs_per_site,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

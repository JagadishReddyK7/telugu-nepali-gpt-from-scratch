"""
Telugu news crawler -- the manual-collection half of the Model H corpus.

Run this FIRST and leave it running. It is the long pole in Phase 1.

    # Step 1: build the URL frontier from sitemaps (minutes)
    python -m telugu.collect.scrape_news --discover

    # Step 2: fetch articles (hours to days; interrupt and re-run freely)
    python -m telugu.collect.scrape_news --run

Output lands in ``data/telugu/raw/*.jsonl`` with ``provenance="manual_scrape"``
on every record, which is what makes the >=20% manual-collection requirement
auditable in the dataset statistics report.

SOURCE SELECTION RATIONALE
--------------------------
The sites below were chosen for three properties:

1. *Genre variety.* A corpus scraped entirely from one newspaper's politics
   section teaches the model a very narrow register. Mixing mainstream dailies,
   a broadcaster, and BBC Telugu (which has a distinctly more formal written
   style) broadens the distribution.

2. *Deep archives.* Sites with dated sitemap indices going back years yield far
   more tokens than a site exposing only recent articles.

3. *Clean markup.* Sites whose article pages extract well with trafilatura.

Editorial text is generally protected by copyright. We are collecting it for
non-commercial academic research and redistributing derived statistics and
model weights rather than the raw text -- which is why the README instructs you
to publish tokenized binaries and checkpoints, not the scraped articles.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_crawl import NewsSite, build_arg_parser, crawl_sites, setup_logging

# Sitemap URLs are left as None wherever auto-discovery is reliable, so the code
# does not rot when a site reorganizes its paths.
TELUGU_SITES: list[NewsSite] = [
    NewsSite(
        name="eenadu",
        base_url="https://www.eenadu.net",
        delay_seconds=1.5,
        notes="Largest-circulation Telugu daily. Broad news coverage, deep archive.",
    ),
    NewsSite(
        name="sakshi",
        base_url="https://www.sakshi.com",
        delay_seconds=1.5,
        notes="Major Telugu daily; strong sections on sport, cinema, business.",
    ),
    NewsSite(
        name="andhrajyothy",
        base_url="https://www.andhrajyothy.com",
        delay_seconds=1.5,
        notes="Mainstream daily; editorial and opinion writing adds formal register.",
    ),
    NewsSite(
        name="ntnews",
        base_url="https://ntnews.com",
        delay_seconds=1.5,
        notes="Namasthe Telangana. Telangana-region usage, complements AP-centric sources.",
    ),
    NewsSite(
        name="v6velugu",
        base_url="https://www.v6velugu.com",
        delay_seconds=2.0,
        notes="Telangana daily with colloquial register; useful lexical diversity.",
    ),
    NewsSite(
        name="bbc_telugu",
        base_url="https://www.bbc.com",
        sitemap_urls=["https://www.bbc.com/telugu/sitemap.xml"],
        url_must_contain=["/telugu/"],
        delay_seconds=2.0,
        notes="BBC Telugu. Carefully edited long-form prose; high text quality.",
    ),
    NewsSite(
        name="prabhanews",
        base_url="https://www.prabhanews.com",
        delay_seconds=2.0,
        notes="Andhra Prabha. Additional mainstream daily for source diversity.",
    ),
]


def main() -> None:
    """Parse arguments and run the Telugu crawl."""
    parser = build_arg_parser("Telugu")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not (args.discover or args.run):
        parser.error("Specify at least one of --discover or --run.")

    sites = TELUGU_SITES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        sites = [s for s in sites if s.name in wanted]
        if not sites:
            parser.error(f"No Telugu sites matched --only={args.only}")

    crawl_sites(
        sites=sites,
        output_root=Path("data/telugu"),
        do_discover=args.discover,
        do_run=args.run,
        max_docs_per_site=args.max_docs_per_site,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

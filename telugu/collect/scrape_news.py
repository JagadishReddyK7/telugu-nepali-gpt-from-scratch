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
        # Auto-discovery found nothing and robots.txt declared no sitemap.
        # These are last-attempt guesses at common Indian-CMS layouts. If they
        # also fail, eenadu gets dropped rather than debugged further -- one
        # hostile source is not worth the deadline risk.
        sitemap_urls=[
            "https://www.eenadu.net/sitemap.xml",
            "https://www.eenadu.net/sitemap-index.xml",
            "https://www.eenadu.net/sitemap/sitemap.xml",
            "https://www.eenadu.net/telugu-news/sitemap.xml",
        ],
        delay_seconds=0.6,
        notes="Largest-circulation Telugu daily. Broad news coverage, deep archive.",
    ),
    NewsSite(
        name="sakshi",
        base_url="https://www.sakshi.com",
        delay_seconds=0.6,
        notes="Major Telugu daily; strong sections on sport, cinema, business.",
    ),
    NewsSite(
        name="andhrajyothy",
        base_url="https://www.andhrajyothy.com",
        delay_seconds=0.6,
        notes="Mainstream daily; editorial and opinion writing adds formal register.",
    ),
    NewsSite(
        name="ntnews",
        base_url="https://ntnews.com",
        delay_seconds=0.6,
        notes="Namasthe Telangana. Telangana-region usage, complements AP-centric sources.",
    ),
    NewsSite(
        name="v6velugu",
        base_url="https://www.v6velugu.com",
        delay_seconds=0.5,
        notes="Telangana daily with colloquial register; useful lexical diversity.",
    ),
    NewsSite(
        name="bbc_telugu",
        base_url="https://www.bbc.com",
        sitemap_urls=[
            "https://www.bbc.com/telugu/sitemap.xml",
            "https://www.bbc.com/sitemaps/https-index-com-telugu.xml",
            "https://www.bbc.com/sitemaps/https-sitemap-com-archive-telugu.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive-telugu.xml",
        ],
        url_must_contain=["/telugu/"],
        delay_seconds=1.0,
        notes="BBC Telugu. Carefully edited long-form prose; high text quality.",
    ),
    NewsSite(
        name="prabhanews",
        base_url="https://www.prabhanews.com",
        delay_seconds=0.5,
        notes="Andhra Prabha. Additional mainstream daily for source diversity.",
    ),

    # -----------------------------------------------------------------------
    # SECONDARY SOURCES
    # -----------------------------------------------------------------------
    # Added after the first discovery pass yielded ~192k URLs -- roughly half
    # of what the >=20% manual-collection rule needs at a 500M-token target.
    # These broaden volume AND register: broadcast newsrooms, a left-leaning
    # daily, and an entertainment portal whose colloquial style differs sharply
    # from broadsheet prose.
    #
    # These domains are UNVERIFIED. Any without a reachable sitemap are logged
    # and skipped automatically by the crash-isolation handling in run_crawl,
    # so a speculative source costs seconds of discovery and nothing else.
    NewsSite(
        name="tv9telugu",
        base_url="https://tv9telugu.com",
        delay_seconds=0.6,
        notes="TV9 Telugu newsroom. High publication volume.",
    ),
    NewsSite(
        name="10tv",
        base_url="https://10tv.in",
        delay_seconds=0.6,
        notes="10TV Telugu. Broadcast newsroom, broad daily coverage.",
    ),
    NewsSite(
        name="ntvtelugu",
        base_url="https://ntvtelugu.com",
        delay_seconds=0.6,
        notes="NTV Telugu. Additional broadcast source.",
    ),
    NewsSite(
        name="telugu_samayam",
        base_url="https://telugu.samayam.com",
        delay_seconds=0.6,
        notes="Times group Telugu portal. Deep archive, clean markup.",
    ),
    # RETIRED: robots.txt disallows crawling. Discovery found
    # 43,342 URLs and every one was refused. We respect robots.txt,
    # so this source is unusable -- kept here (commented) so the
    # Phase 1 report can state why it was dropped.
    # NewsSite(
    #     name="telugu_oneindia",
    #     base_url="https://telugu.oneindia.com",
    #     delay_seconds=0.6,
    #     notes="OneIndia Telugu. Large general-interest archive.",
    # ),
    NewsSite(
        name="prajasakti",
        base_url="https://www.prajasakti.com",
        delay_seconds=0.6,
        notes="Left-leaning daily. Political register absent from other sources.",
    ),
    NewsSite(
        name="navatelangana",
        base_url="https://www.navatelangana.com",
        delay_seconds=0.6,
        notes="Nava Telangana daily. Telangana regional coverage.",
    ),
    NewsSite(
        name="andhrabhoomi",
        base_url="https://www.andhrabhoomi.net",
        delay_seconds=0.6,
        notes="Andhra Bhoomi. Long-running daily with literary supplements.",
    ),
    NewsSite(
        name="suryaa",
        base_url="https://www.suryaa.com",
        delay_seconds=0.6,
        notes="Surya daily. Additional mainstream source.",
    ),
    NewsSite(
        name="greatandhra",
        base_url="https://www.greatandhra.com",
        delay_seconds=0.8,
        notes="Entertainment/cinema portal. Colloquial register, lexical diversity.",
    ),

    # -----------------------------------------------------------------------
    # INSTITUTIONAL AND LITERARY SOURCES
    # -----------------------------------------------------------------------
    # A corpus built only from newspapers learns exactly one register:
    # present-tense reportage with a narrow topical band. These sources supply
    # formal administrative prose and literary long-form, neither of which
    # appears in daily news. Volume here is small; the value is distributional.
    #
    # Wikipedia and Wikisource are deliberately ABSENT -- they are downloaded as
    # dumps by the ingestion script, not crawled, and count toward the
    # "downloaded" share rather than the >=20% manual share.
    NewsSite(
        name="andhrabharati",
        base_url="https://www.andhrabharati.com",
        delay_seconds=1.0,
        notes="Telugu literature archive. Classical and modern literary prose.",
    ),
    NewsSite(
        name="pib_telugu",
        base_url="https://www.pib.gov.in",
        url_must_contain=["telugu", "/te/"],
        delay_seconds=1.0,
        notes="Press Information Bureau Telugu releases. Formal government register.",
    ),
    NewsSite(
        name="ap_gov",
        base_url="https://www.ap.gov.in",
        delay_seconds=1.0,
        notes="Andhra Pradesh state portal. Administrative prose.",
    ),
    NewsSite(
        name="telangana_gov",
        base_url="https://www.telangana.gov.in",
        delay_seconds=1.0,
        notes="Telangana state portal. Administrative prose.",
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
        parallel_sites=args.parallel_sites,
    )


if __name__ == "__main__":
    main()

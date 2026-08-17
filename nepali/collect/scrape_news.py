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
        delay_seconds=0.6,
        notes="Kantipur, Nepal's largest daily. Deep archive, high editorial quality.",
    ),
    NewsSite(
        name="setopati",
        base_url="https://www.setopati.com",
        delay_seconds=0.6,
        notes="Leading digital-native outlet; strong long-form and opinion writing.",
    ),
    NewsSite(
        name="onlinekhabar",
        base_url="https://www.onlinekhabar.com",
        delay_seconds=0.6,
        notes="High-volume digital news portal (WordPress; sitemap_index.xml).",
    ),
    NewsSite(
        name="ratopati",
        base_url="https://www.ratopati.com",
        delay_seconds=0.6,
        notes="Popular news portal; broad general-interest coverage.",
    ),
    NewsSite(
        name="nagariknews",
        base_url="https://nagariknews.nagariknetwork.com",
        delay_seconds=0.6,
        notes="Nagarik daily. Mainstream broadsheet register.",
    ),
    NewsSite(
        name="bbc_nepali",
        base_url="https://www.bbc.com",
        sitemap_urls=[
            "https://www.bbc.com/nepali/sitemap.xml",
            "https://www.bbc.com/sitemaps/https-index-com-nepali.xml",
            "https://www.bbc.com/sitemaps/https-sitemap-com-archive-nepali.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive-nepali.xml",
        ],
        url_must_contain=["/nepali/"],
        delay_seconds=2.0,
        notes="BBC Nepali. Carefully edited prose; reliably Nepali rather than Hindi.",
    ),
    NewsSite(
        name="himalkhabar",
        base_url="https://www.himalkhabar.com",
        delay_seconds=0.6,
        notes="Himal Khabarpatrika. Magazine-style long-form; distinct register.",
    ),
    NewsSite(
        name="annapurnapost",
        base_url="https://annapurnapost.com",
        delay_seconds=0.6,
        notes="Annapurna Post daily. Additional mainstream source.",
    ),

    # -----------------------------------------------------------------------
    # SECONDARY SOURCES
    # -----------------------------------------------------------------------
    # Nepali is the binding constraint on the >=20% manual-collection rule, so
    # the source list is deliberately wide. Domain variety matters here beyond
    # raw volume: health, tech and business outlets carry vocabulary that a
    # politics-heavy news corpus never contains, which directly affects
    # tokenizer fertility and downstream coverage.
    #
    # UNVERIFIED domains. Sources without a reachable sitemap are logged and
    # skipped automatically, so listing speculative ones is nearly free.
    NewsSite(
        name="gorkhapatra",
        base_url="https://gorkhapatraonline.com",
        delay_seconds=0.6,
        notes="Gorkhapatra, Nepal's oldest newspaper (state-owned). Deep archive.",
    ),
    NewsSite(
        name="nayapatrika",
        base_url="https://www.nayapatrikadaily.com",
        delay_seconds=0.6,
        notes="Naya Patrika daily. Mainstream broadsheet.",
    ),
    NewsSite(
        name="baahrakhari",
        base_url="https://www.baahrakhari.com",
        delay_seconds=0.6,
        notes="Baahrakhari. High-volume digital news portal.",
    ),
    NewsSite(
        name="nepalkhabar",
        base_url="https://nepalkhabar.com",
        delay_seconds=0.6,
        notes="Nepal Khabar. General news portal.",
    ),
    NewsSite(
        name="ujyaaloonline",
        base_url="https://ujyaaloonline.com",
        delay_seconds=0.6,
        notes="Ujyaalo Online. Radio network's news arm; spoken-adjacent register.",
    ),
    NewsSite(
        name="nepalpress",
        base_url="https://nepalpress.com",
        delay_seconds=0.6,
        notes="Nepal Press. Digital-native outlet.",
    ),
    # RETIRED: robots.txt disallows crawling. Discovery found
    # 222,181 URLs and every one was refused. We respect robots.txt,
    # so this source is unusable -- kept here (commented) so the
    # Phase 1 report can state why it was dropped.
    # NewsSite(
    #     name="deshsanchar",
    #     base_url="https://deshsanchar.com",
    #     delay_seconds=0.6,
    #     notes="Desh Sanchar. General-interest portal.",
    # ),
    NewsSite(
        name="swasthyakhabar",
        base_url="https://swasthyakhabar.com",
        delay_seconds=0.8,
        notes="Health news. Medical vocabulary absent from general news sources.",
    ),
    # RETIRED: robots.txt disallows crawling. Discovery found
    # 34,089 URLs and every one was refused. We respect robots.txt,
    # so this source is unusable -- kept here (commented) so the
    # Phase 1 report can state why it was dropped.
    # NewsSite(
    #     name="techpana",
    #     base_url="https://techpana.com",
    #     delay_seconds=0.8,
    #     notes="Technology news. Technical register and loanword handling.",
    # ),
    # RETIRED: robots.txt disallows crawling. Discovery found
    # 124,505 URLs and every one was refused. We respect robots.txt,
    # so this source is unusable -- kept here (commented) so the
    # Phase 1 report can state why it was dropped.
    # NewsSite(
    #     name="bizmandu",
    #     base_url="https://bizmandu.com",
    #     delay_seconds=0.8,
    #     notes="Business and finance. Numeric/economic vocabulary.",
    # ),
    NewsSite(
        name="shilapatra",
        base_url="https://shilapatra.com",
        delay_seconds=0.8,
        notes="Shilapatra. Additional digital outlet.",
    ),
    NewsSite(
        name="lokaantar",
        base_url="https://lokaantar.com",
        delay_seconds=0.8,
        notes="Lokaantar. Additional digital outlet.",
    ),

    # -----------------------------------------------------------------------
    # INSTITUTIONAL SOURCES
    # -----------------------------------------------------------------------
    # Nepali is source-starved and currently ~99% one news outlet, so register
    # diversity matters even more here than for Telugu. Government sites are
    # also the least likely to block a research crawler.
    #
    # lawcommission.gov.np is the standout: it publishes Nepal's entire statute
    # book in Nepali. Legal prose is long, formally structured, and completely
    # absent from news -- exactly the distribution gap in this corpus.
    NewsSite(
        name="lawcommission",
        base_url="https://www.lawcommission.gov.np",
        delay_seconds=1.0,
        notes="Nepal Law Commission. Full statute book in Nepali; formal legal prose.",
    ),
    NewsSite(
        name="opmcm_gov",
        base_url="https://www.opmcm.gov.np",
        delay_seconds=1.0,
        notes="Office of the Prime Minister. Official notices and decisions.",
    ),
    NewsSite(
        name="moe_gov",
        base_url="https://www.moe.gov.np",
        delay_seconds=1.0,
        notes="Ministry of Education. Circulars and policy documents.",
    ),
    NewsSite(
        name="mof_gov",
        base_url="https://www.mof.gov.np",
        delay_seconds=1.0,
        notes="Ministry of Finance. Budget documents; economic register.",
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
        parallel_sites=args.parallel_sites,
    )


if __name__ == "__main__":
    main()

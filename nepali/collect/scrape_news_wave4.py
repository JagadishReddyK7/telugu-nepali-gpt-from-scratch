"""Wave 4 Nepali sources -- added to push manual collection past 333M chars.

    python -m nepali.collect.scrape_news_wave4 --discover
    python -m nepali.collect.scrape_news_wave4 --run --parallel-sites 8

Imports the existing NEPALI_SITES and appends only sources not already tried,
so state, output shards and provenance tagging are identical to the main
crawler. Sites without a reachable sitemap are logged and skipped, so listing
speculative domains costs seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_crawl import NewsSite, build_arg_parser, crawl_sites, setup_logging
from nepali.collect.scrape_news import NEPALI_SITES

_EXISTING = {s.name for s in NEPALI_SITES}

WAVE4: list[NewsSite] = [
    NewsSite(name="souryadaily", base_url="https://www.souryadaily.com",
             delay_seconds=0.8, notes="Sourya Daily. Broadsheet."),
    NewsSite(name="nepalsamaya", base_url="https://nepalsamaya.com",
             delay_seconds=0.8, notes="Nepal Samaya. Digital-native outlet."),
    NewsSite(name="nepalnews", base_url="https://www.nepalnews.com",
             delay_seconds=0.8, notes="Nepal News. Long-running portal."),
    NewsSite(name="nepal24hours", base_url="https://nepal24hours.com",
             delay_seconds=0.8, notes="Nepal 24 Hours."),
    NewsSite(name="nepalitelegraph", base_url="https://nepalitelegraph.com",
             delay_seconds=0.8, notes="Nepali Telegraph."),
    NewsSite(name="nepalipatra", base_url="https://nepalipatra.com",
             delay_seconds=0.8, notes="Nepali Patra."),
    NewsSite(name="nepalpage", base_url="https://nepalpage.com",
             delay_seconds=0.8, notes="Nepal Page."),
    NewsSite(name="nepalipost", base_url="https://nepalipost.com",
             delay_seconds=0.8, notes="Nepali Post."),
    NewsSite(name="imagenews", base_url="https://www.imagenews.com.np",
             delay_seconds=0.8, notes="Image News."),
    NewsSite(name="kathmandutoday", base_url="https://kathmandutoday.com",
             delay_seconds=0.8, notes="Kathmandu Today."),
    NewsSite(name="nepallive", base_url="https://nepallive.com",
             delay_seconds=0.8, notes="Nepal Live. High-volume portal."),
    NewsSite(name="samudrapari", base_url="https://samudrapari.com",
             delay_seconds=0.8, notes="Samudrapari. Diaspora coverage."),
    NewsSite(name="harekpal", base_url="https://harekpal.com",
             delay_seconds=0.8, notes="Harek Pal."),
    NewsSite(name="kathmandutribune", base_url="https://kathmandutribune.com",
             delay_seconds=0.8, notes="Kathmandu Tribune."),
    NewsSite(name="nepalnews24", base_url="https://nepalnews24.com",
             delay_seconds=0.8, notes="Nepal News 24."),
    NewsSite(name="samacharpati", base_url="https://samacharpati.com",
             delay_seconds=0.8, notes="Samacharpati."),
    NewsSite(name="bigulnews", base_url="https://bigulnews.com",
             delay_seconds=0.8, notes="Bigul News."),
    NewsSite(name="pahilpost", base_url="https://www.pahilpost.com",
             delay_seconds=0.8, notes="Pahil Post (distinct from pahilopost)."),
    NewsSite(name="makalukhabar", base_url="https://www.makalukhabar.com",
             delay_seconds=0.8, notes="Makalu Khabar. Established portal."),
    NewsSite(name="ekagaj", base_url="https://ekagaj.com",
             delay_seconds=0.8, notes="Ekagaj. Digital daily."),
    NewsSite(name="rajdhanidaily", base_url="https://rajdhanidaily.com",
             delay_seconds=0.8, notes="Rajdhani Daily. Broadsheet."),
    NewsSite(name="himalayatimes", base_url="https://www.himalayatimes.com.np",
             delay_seconds=0.8, notes="Himalaya Times."),
    NewsSite(name="karobardaily", base_url="https://karobardaily.com",
             delay_seconds=0.8, notes="Karobar Daily. Business register."),
    NewsSite(name="arthasarokar", base_url="https://arthasarokar.com",
             delay_seconds=0.8, notes="Artha Sarokar. Economics vocabulary."),
    NewsSite(name="khabardabali", base_url="https://khabardabali.com",
             delay_seconds=0.8, notes="Khabar Dabali."),
    NewsSite(name="everestpost", base_url="https://everestpost.com",
             delay_seconds=0.8, notes="Everest Post."),
    NewsSite(name="nepalaaja", base_url="https://nepalaaja.com",
             delay_seconds=0.8, notes="Nepal Aaja."),
    NewsSite(name="ratopati_np", base_url="https://ratopati.com",
             delay_seconds=0.8, notes="Ratopati apex domain (no www)."),

    # Government. Formal administrative prose -- a register news never covers,
    # and the least likely class of site to refuse a research crawler.
    NewsSite(name="moha_gov", base_url="https://www.moha.gov.np",
             delay_seconds=1.0, notes="Ministry of Home Affairs."),
    NewsSite(name="mofa_gov", base_url="https://mofa.gov.np",
             delay_seconds=1.0, notes="Ministry of Foreign Affairs."),
    NewsSite(name="moest_gov", base_url="https://moest.gov.np",
             delay_seconds=1.0, notes="Ministry of Education, Science and Tech."),
    NewsSite(name="nepal_gov", base_url="https://nepal.gov.np",
             delay_seconds=1.0, notes="Nepal government portal."),
    NewsSite(name="dohs_gov", base_url="https://dohs.gov.np",
             delay_seconds=1.0, notes="Department of Health Services."),
    NewsSite(name="election_gov", base_url="https://election.gov.np",
             delay_seconds=1.0, notes="Election Commission. Formal notices."),
    NewsSite(name="npc_gov", base_url="https://npc.gov.np",
             delay_seconds=1.0, notes="National Planning Commission. Policy prose."),
]

NEW_SITES = [s for s in WAVE4 if s.name not in _EXISTING]


def main() -> None:
    p = build_arg_parser("Nepali (wave 4)")
    args = p.parse_args()
    setup_logging(getattr(args, "verbose", False))
    if not (args.discover or args.run):
        p.error("Pass --discover, --run, or both.")
    print(f"wave 4: {len(NEW_SITES)} new sites "
          f"({len(WAVE4) - len(NEW_SITES)} already in the main list)")
    crawl_sites(
        NEW_SITES, Path("data/nepali"),
        do_discover=args.discover, do_run=args.run,
        max_docs_per_site=args.max_docs_per_site,
        max_workers=getattr(args, "max_workers", 4),
        parallel_sites=args.parallel_sites,
    )


if __name__ == "__main__":
    main()

"""Wave 4 Telugu sources -- optional insurance for the manual share.

    python -m telugu.collect.scrape_news_wave4 --discover
    python -m telugu.collect.scrape_news_wave4 --run --parallel-sites 8

Telugu already has enough manual text for a 548M-token ceiling, so this is not
required to reach 500M. It is worth running anyway: more manual characters raise
the ceiling and improve source diversity, both of which the report is graded on.

NOTE ON BILINGUAL SITES. greatandhra contributed 48,612 documents with a median
Telugu-script ratio of 0.00 because it runs a large English section. Where a
site is known to be bilingual, ``url_must_contain`` restricts the crawl to its
Telugu paths rather than relying on the cleaner to throw the work away later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_crawl import NewsSite, build_arg_parser, crawl_sites, setup_logging
from telugu.collect.scrape_news import TELUGU_SITES

_EXISTING = {s.name for s in TELUGU_SITES}

WAVE4: list[NewsSite] = [
    NewsSite(name="telanganatoday", base_url="https://telanganatoday.com",
             delay_seconds=0.8, notes="Telangana Today."),
    NewsSite(name="webdunia_telugu", base_url="https://telugu.webdunia.com",
             delay_seconds=0.8, notes="Webdunia Telugu. Lifestyle and culture."),
    NewsSite(name="vaartha", base_url="https://www.vaartha.com",
             delay_seconds=0.8, notes="Vaartha daily."),
    NewsSite(name="boldsky_telugu", base_url="https://telugu.boldsky.com",
             delay_seconds=0.8, notes="Boldsky Telugu. Health and lifestyle."),
    NewsSite(name="oneindia_telugu", base_url="https://telugu.oneindia.com",
             delay_seconds=0.8, notes="OneIndia Telugu. High volume."),
    NewsSite(name="filmibeat_telugu", base_url="https://telugu.filmibeat.com",
             delay_seconds=0.8, notes="Film journalism; distinct vocabulary."),
    NewsSite(name="news18_telugu", base_url="https://telugu.news18.com",
             delay_seconds=0.8, notes="News18 Telugu."),
    NewsSite(name="abplive_telugu", base_url="https://telugu.abplive.com",
             delay_seconds=0.8, notes="ABP Telugu."),
    NewsSite(name="asianetnews_telugu", base_url="https://telugu.asianetnews.com",
             delay_seconds=0.8, notes="Asianet Telugu."),
    NewsSite(name="etvbharat_telugu", base_url="https://www.etvbharat.com",
             url_must_contain=["/telugu/"], delay_seconds=0.8,
             notes="ETV Bharat Telugu section. Path filter keeps Telugu."),
    NewsSite(name="zee_telugu", base_url="https://zeenews.india.com",
             url_must_contain=["/telugu/"], delay_seconds=0.8,
             notes="Zee News Telugu section."),
    NewsSite(name="ap7am", base_url="https://www.ap7am.com",
             delay_seconds=0.8, notes="AP7AM. Regional news."),
    NewsSite(name="mahaanews", base_url="https://mahaanews.com",
             delay_seconds=0.8, notes="Mahaa News."),
    NewsSite(name="hmtvlive", base_url="https://www.hmtvlive.com",
             delay_seconds=0.8, notes="HMTV. Broadcast news portal."),
    NewsSite(name="telugustop", base_url="https://telugustop.com",
             delay_seconds=0.8, notes="TeluguStop. Entertainment register."),
    NewsSite(name="vikaspedia_te", base_url="https://te.vikaspedia.in",
             delay_seconds=1.0,
             notes="Vikaspedia Telugu. Government knowledge portal covering "
                   "agriculture, health, education, energy and law -- "
                   "encyclopedic register absent from news."),
    NewsSite(name="ts_gov", base_url="https://www.telangana.gov.in",
             delay_seconds=1.0, notes="Telangana government portal."),
]

NEW_SITES = [s for s in WAVE4 if s.name not in _EXISTING]


def main() -> None:
    p = build_arg_parser("Telugu (wave 4)")
    args = p.parse_args()
    setup_logging(args.verbose)
    if not (args.discover or args.run):
        p.error("Pass --discover, --run, or both.")
    print(f"wave 4: {len(NEW_SITES)} new sites")
    crawl_sites(
        NEW_SITES, Path("data/telugu"),
        do_discover=args.discover, do_run=args.run,
        max_docs_per_site=args.max_docs_per_site,
        max_workers=args.max_workers,
        parallel_sites=args.parallel_sites,
    )


if __name__ == "__main__":
    main()

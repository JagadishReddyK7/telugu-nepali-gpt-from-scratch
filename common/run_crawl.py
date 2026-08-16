"""
Shared entry point for running a set of news crawlers for one language.

Both language pipelines call this with their own site list, so the crawling
logic lives in exactly one place while the *data* stays rigorously separated:
each language writes to its own output directory and its own state database,
and no document is ever shared between them.

Usage (from the per-language wrapper scripts)::

    python -m telugu.collect.scrape_news --discover --run --max-docs-per-site 60000

Run ``--discover`` first (populates the URL frontier from sitemaps; fast), then
``--run`` (fetches articles; slow, resumable). Passing both does them in order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from common.scraper import NewsScraper, ScraperConfig
from common.sitemap_discovery import discover_sitemaps

logger = logging.getLogger(__name__)


@dataclass
class NewsSite:
    """One news source to crawl.

    Attributes:
        name: Short slug used for output filenames and provenance tagging.
        base_url: Site root used for sitemap auto-discovery.
        sitemap_urls: Explicit sitemap URLs. If empty, they are auto-discovered.
        url_must_contain: Optional path substrings that mark article URLs.
        delay_seconds: Per-site politeness delay. Raise for smaller sites.
        notes: Free-text description, quoted in the dataset statistics report.
    """

    name: str
    base_url: str
    sitemap_urls: list[str] | None = None
    url_must_contain: list[str] | None = None
    delay_seconds: float = 1.0
    notes: str = ""


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for crawler runs.

    Args:
        verbose: Emit DEBUG-level records if True.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # trafilatura and urllib3 are extremely chatty at INFO level.
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def crawl_sites(
    sites: list[NewsSite],
    output_root: Path,
    do_discover: bool,
    do_run: bool,
    max_docs_per_site: int | None,
    max_workers: int,
) -> dict:
    """Run discovery and/or fetching across a list of sites.

    Args:
        sites: Sources to crawl.
        output_root: Directory under which ``raw/`` and ``state/`` are created.
        do_discover: Populate URL frontiers from sitemaps.
        do_run: Fetch and extract pending URLs.
        max_docs_per_site: Per-site extraction cap, or None for unlimited.
        max_workers: Concurrent fetch threads per site.

    Returns:
        Mapping of site name to that site's statistics dictionary.
    """
    output_root = Path(output_root)
    raw_dir = output_root / "raw"
    state_dir = output_root / "state"
    summary: dict[str, dict] = {}

    for site in sites:
        logger.info("=" * 70)
        logger.info("SITE: %s (%s)", site.name, site.base_url)
        logger.info("=" * 70)

        sitemaps = site.sitemap_urls
        if not sitemaps:
            sitemaps = discover_sitemaps(site.base_url)
        if not sitemaps:
            logger.error("Skipping %s: no sitemaps found", site.name)
            summary[site.name] = {"error": "no_sitemap"}
            continue

        config = ScraperConfig(
            name=site.name,
            sitemap_urls=sitemaps,
            output_dir=raw_dir,
            state_db=state_dir / f"{site.name}.db",
            delay_seconds=site.delay_seconds,
            max_workers=max_workers,
            url_must_contain=site.url_must_contain or [],
        )
        scraper = NewsScraper(config)
        try:
            if do_discover:
                scraper.discover()
            counts = scraper.state.counts()
            logger.info("[%s] frontier status: %s", site.name, counts)
            if do_run:
                stats = scraper.run(max_docs=max_docs_per_site)
                stats["frontier"] = scraper.state.counts()
                summary[site.name] = stats
            else:
                summary[site.name] = {"frontier": counts}
        finally:
            scraper.close()

    # Persist the summary so the dataset-statistics step can read per-source
    # yields without re-scanning every shard.
    summary_path = output_root / "crawl_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    existing.update(summary)
    summary_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote crawl summary to %s", summary_path)
    return summary


def build_arg_parser(language: str) -> argparse.ArgumentParser:
    """Construct the shared CLI for per-language crawl scripts.

    Args:
        language: Language name, used only in help text.

    Returns:
        A configured argument parser.
    """
    p = argparse.ArgumentParser(
        description=f"Crawl {language} news sites for manual corpus collection."
    )
    p.add_argument(
        "--discover", action="store_true",
        help="Walk sitemaps and populate the URL frontier (run this first).",
    )
    p.add_argument(
        "--run", action="store_true",
        help="Fetch and extract pending URLs. Resumable; safe to re-run.",
    )
    p.add_argument(
        "--max-docs-per-site", type=int, default=None,
        help="Stop each site after roughly this many extracted documents.",
    )
    p.add_argument(
        "--max-workers", type=int, default=4,
        help="Concurrent fetch threads per site. Keep low to stay polite.",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated site names to restrict the run to.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p

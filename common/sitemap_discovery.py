"""
Locate a site's XML sitemaps without hardcoding fragile paths.

Sitemap locations vary by CMS. WordPress sites (very common for Nepali news)
typically expose ``/sitemap_index.xml``; Drupal and custom stacks often use
``/sitemap.xml``; some sites only declare it in ``robots.txt``. Rather than
guessing wrong and silently crawling nothing, this module tries the declared
location first and falls back to a ranked list of conventional paths.

Always run :func:`discover_sitemaps` before a long crawl and check that it
returned something. A crawl that reports "0 new URLs" almost always means the
sitemap path was wrong, not that the site is empty.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "LMA-Course-Research-Crawler/1.0 "
    "(academic monolingual corpus collection; contact: hrushikratan.j@vaayum.com)"
)

# Ordered by how likely each is to be the full archive index rather than a
# recent-articles-only feed.
COMMON_SITEMAP_PATHS = [
    "/sitemap_index.xml",
    "/sitemap.xml",
    "/sitemap-index.xml",
    "/sitemaps.xml",
    "/wp-sitemap.xml",
    "/sitemap/sitemap-index.xml",
    "/news-sitemap.xml",
]

_SITEMAP_DIRECTIVE = re.compile(r"^\s*sitemap:\s*(\S+)", re.IGNORECASE | re.MULTILINE)


def discover_sitemaps(base_url: str, timeout: int = 15) -> list[str]:
    """Find sitemap URLs for a site.

    Args:
        base_url: Site root, e.g. ``"https://ekantipur.com"``.
        timeout: Per-request timeout in seconds.

    Returns:
        Sitemap URLs, declared ones first. Empty if none could be found, which
        means the site needs manual investigation before crawling.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    found: list[str] = []

    # 1. Preferred: whatever robots.txt declares. This is authoritative.
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        resp = session.get(robots_url, timeout=timeout)
        if resp.status_code == 200:
            for match in _SITEMAP_DIRECTIVE.finditer(resp.text):
                url = match.group(1).strip()
                if url not in found:
                    found.append(url)
            if found:
                logger.info("%s: %d sitemap(s) declared in robots.txt",
                            base_url, len(found))
    except requests.RequestException as exc:
        logger.warning("Could not read robots.txt for %s: %s", base_url, exc)

    # 2. Fallback: probe conventional paths with cheap HEAD-like GETs.
    if not found:
        for path in COMMON_SITEMAP_PATHS:
            candidate = urljoin(base_url, path)
            try:
                resp = session.get(candidate, timeout=timeout, stream=True)
                if resp.status_code == 200:
                    head = resp.raw.read(512, decode_content=True)
                    # Confirm it is actually XML and not a styled 404 page.
                    if b"<sitemapindex" in head or b"<urlset" in head:
                        found.append(candidate)
                        logger.info("%s: found sitemap at %s", base_url, path)
                        break
                resp.close()
            except requests.RequestException:
                continue

    if not found:
        logger.error(
            "%s: NO SITEMAP FOUND. Inspect the site manually before crawling.",
            base_url,
        )
    return found

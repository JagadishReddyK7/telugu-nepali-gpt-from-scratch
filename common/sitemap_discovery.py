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

ON EXCEPTION HANDLING
---------------------
Network code fails in more ways than its library's own exception hierarchy
admits. ``requests`` wraps most errors in ``RequestException``, but a truncated
chunked-transfer response surfaces as ``urllib3.exceptions.ProtocolError``,
which is NOT a subclass of ``RequestException`` and therefore escapes the
obvious ``except requests.RequestException`` clause. During the first Nepali
crawl that exact case took down the entire run at the fifth site.

The lesson generalizes: in a long unattended crawl, a single unreachable host
must never be able to kill the process. Every network call here catches broad
``Exception`` and degrades to "this source yielded nothing", because losing one
source is recoverable and losing the whole crawl is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "LMA-Course-Research-Crawler/1.0 "
    "(academic monolingual corpus collection; contact: hrushikratan.j@vaayum.com)"
)

# Sent alongside the User-Agent. These are headers any ordinary HTTP client
# sends; some CDN front-ends reject requests that omit them outright, which
# produces a confusing 403 that looks like a deliberate block but is not.
DEFAULT_HEADERS = {
    "Accept": "application/xml,text/xml,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ne;q=0.8,te;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Ordered by how likely each is to be the full archive index rather than a
# recent-articles-only feed. Expanded after observing real failures on Nepali
# news sites, several of which use non-standard paths.
COMMON_SITEMAP_PATHS = [
    "/sitemap_index.xml",
    "/sitemap.xml",
    "/sitemap-index.xml",
    "/sitemaps.xml",
    "/wp-sitemap.xml",
    "/sitemap/sitemap-index.xml",
    "/sitemap/index.xml",
    "/sitemaps/sitemap-index.xml",
    "/news-sitemap.xml",
    "/sitemap-news.xml",
    "/sitemap/news.xml",
    "/rss.xml",          # last resort: an RSS feed still yields article URLs
    "/feed",
]

_SITEMAP_DIRECTIVE = re.compile(r"^\s*sitemap:\s*(\S+)", re.IGNORECASE | re.MULTILINE)


@dataclass
class ProbeResult:
    """Outcome of probing one candidate sitemap URL.

    Attributes:
        url: The URL probed.
        status: HTTP status code, or None if the request never completed.
        looks_like_sitemap: Whether the response body actually began with XML
            sitemap markup, as opposed to a styled 404 page returning HTTP 200.
        error: Exception class name if the request failed outright.
        snippet: First bytes of the body, for eyeballing what came back.
    """

    url: str
    status: int | None = None
    looks_like_sitemap: bool = False
    error: str | None = None
    snippet: str = ""


def _make_session(user_agent: str) -> requests.Session:
    """Build a requests session with our standard headers.

    Args:
        user_agent: User-Agent header value.

    Returns:
        A configured :class:`requests.Session`.
    """
    session = requests.Session()
    session.headers.update({**DEFAULT_HEADERS, "User-Agent": user_agent})
    return session


def probe_url(
    url: str,
    session: requests.Session,
    timeout: int = 15,
    snippet_bytes: int = 400,
) -> ProbeResult:
    """Fetch a candidate URL and report what actually came back.

    Used both by :func:`discover_sitemaps` and by the standalone diagnostic
    script, so that a failing site can be triaged by HTTP status rather than
    guessed at.

    Args:
        url: Candidate sitemap URL.
        session: Session to fetch with.
        timeout: Request timeout in seconds.
        snippet_bytes: How many bytes of the body to keep for inspection.

    Returns:
        A :class:`ProbeResult`. Never raises.
    """
    result = ProbeResult(url=url)
    try:
        # No streaming. Reading the body incrementally via ``resp.raw`` is what
        # exposed us to ProtocolError on truncated chunked responses; letting
        # requests buffer the whole (small) body is both simpler and safer.
        resp = session.get(url, timeout=timeout)
        result.status = resp.status_code
        body = resp.content[:snippet_bytes]
        result.snippet = body.decode("utf-8", errors="replace")
        if resp.status_code == 200:
            head = resp.content[:2048]
            result.looks_like_sitemap = (
                b"<sitemapindex" in head or b"<urlset" in head or b"<rss" in head
            )
    except Exception as exc:  # deliberately broad -- see module docstring
        result.error = type(exc).__name__
    return result


def discover_sitemaps(
    base_url: str,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[str]:
    """Find sitemap URLs for a site.

    Args:
        base_url: Site root, e.g. ``"https://ekantipur.com"``.
        timeout: Per-request timeout in seconds.
        user_agent: User-Agent header to identify with.

    Returns:
        Sitemap URLs, declared ones first. Empty if none could be found, which
        means the site needs manual investigation -- run
        ``scripts/diagnose_sitemaps.py`` against it.
    """
    session = _make_session(user_agent)
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
        else:
            logger.info("%s: robots.txt returned HTTP %s",
                        base_url, resp.status_code)
    except Exception as exc:
        logger.warning("%s: could not read robots.txt (%s)",
                       base_url, type(exc).__name__)

    # 2. Fallback: probe conventional paths.
    if not found:
        for path in COMMON_SITEMAP_PATHS:
            candidate = urljoin(base_url, path)
            probe = probe_url(candidate, session, timeout=timeout)
            if probe.looks_like_sitemap:
                found.append(candidate)
                logger.info("%s: found sitemap at %s", base_url, path)
                break
            if probe.error:
                logger.debug("%s: %s -> %s", base_url, path, probe.error)

    if not found:
        logger.error(
            "%s: NO SITEMAP FOUND. Run:  python scripts/diagnose_sitemaps.py %s",
            base_url, base_url,
        )
    return found

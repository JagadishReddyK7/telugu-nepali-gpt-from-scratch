"""
Triage a site whose sitemap could not be found.

``discover_sitemaps`` reports only success or failure, which is not enough to
fix a broken source. This script probes every candidate path and prints the
actual HTTP status and response snippet, so the failure mode is identifiable
rather than guessable:

    403 on everything, including robots.txt
        A CDN or WAF is refusing non-browser clients. The site may still be
        crawlable at a slower rate, or may need to be dropped.

    404 on every candidate but 200 on robots.txt
        The site uses a non-standard sitemap path. Read the robots.txt snippet
        printed below -- it often names the real one.

    200 but looks_like_sitemap = False
        A "soft 404": the server returns an HTML error page with status 200.
        This is why we sniff the body instead of trusting the status code.

    Connection errors / timeouts
        Host is unreachable from here, or geo-restricted.

Usage:
    python scripts/diagnose_sitemaps.py                     # all configured sites
    python scripts/diagnose_sitemaps.py --lang nepali       # one language
    python scripts/diagnose_sitemaps.py https://example.com # one arbitrary site
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.sitemap_discovery import (  # noqa: E402
    COMMON_SITEMAP_PATHS,
    DEFAULT_USER_AGENT,
    _make_session,
    probe_url,
)


def diagnose(base_url: str, timeout: int = 15, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    """Probe robots.txt and every candidate sitemap path for one site.

    Args:
        base_url: Site root to diagnose.
        timeout: Per-request timeout in seconds.
        user_agent: User-Agent header to send.

    Returns:
        A dictionary summarizing what was found.
    """
    session = _make_session(user_agent)
    print("=" * 78)
    print(f"DIAGNOSING  {base_url}")
    print("=" * 78)

    # --- robots.txt --------------------------------------------------------
    robots = probe_url(urljoin(base_url, "/robots.txt"), session, timeout)
    print(f"\n  robots.txt   status={robots.status}  error={robots.error}")
    if robots.status == 200:
        declared = [
            line.strip()
            for line in robots.snippet.splitlines()
            if line.lower().startswith("sitemap:")
        ]
        if declared:
            print("    DECLARED SITEMAPS:")
            for d in declared:
                print(f"      {d}")
        else:
            print("    (no Sitemap: directive in the first 400 bytes)")
        print("    --- robots.txt head ---")
        for line in robots.snippet.splitlines()[:12]:
            print(f"      {line}")

    # --- candidate paths ---------------------------------------------------
    print("\n  CANDIDATE PATHS:")
    hits = []
    for path in COMMON_SITEMAP_PATHS:
        candidate = urljoin(base_url, path)
        r = probe_url(candidate, session, timeout)
        flag = "  <-- USABLE" if r.looks_like_sitemap else ""
        detail = r.error if r.error else f"status={r.status}"
        print(f"    {path:34s} {detail:28s} xml={r.looks_like_sitemap}{flag}")
        if r.looks_like_sitemap:
            hits.append(candidate)

    print()
    if hits:
        print(f"  RESULT: {len(hits)} usable sitemap(s). Add to the site config:")
        print(f"    sitemap_urls={hits!r}")
    else:
        statuses = {robots.status}
        if statuses == {403}:
            print("  RESULT: blocked (403). This source may not be crawlable.")
        else:
            print("  RESULT: no usable sitemap. Check the robots.txt head above,")
            print("          or open the site and look for an archive index page.")
    print()
    return {"base_url": base_url, "hits": hits, "robots_status": robots.status}


def main() -> None:
    """Parse arguments and run diagnostics."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urls", nargs="*", help="Specific site roots to diagnose.")
    p.add_argument("--lang", choices=["telugu", "nepali"], default=None,
                   help="Diagnose all configured sites for one language.")
    p.add_argument("--timeout", type=int, default=15)
    args = p.parse_args()

    targets: list[str] = list(args.urls)
    if args.lang == "telugu" or (not targets and args.lang is None):
        from telugu.collect.scrape_news import TELUGU_SITES
        targets += [s.base_url for s in TELUGU_SITES]
    if args.lang == "nepali" or (not args.urls and args.lang is None):
        from nepali.collect.scrape_news import NEPALI_SITES
        targets += [s.base_url for s in NEPALI_SITES]

    # De-duplicate while preserving order (bbc.com appears in both lists).
    seen: set[str] = set()
    ordered = [t for t in targets if not (t in seen or seen.add(t))]

    results = [diagnose(t, timeout=args.timeout) for t in ordered]

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        status = f"{len(r['hits'])} sitemap(s)" if r["hits"] else "NONE"
        print(f"  {r['base_url']:52s} {status}")


if __name__ == "__main__":
    main()

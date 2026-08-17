"""
Report crawl progress, corpus size, and a projected finish time.

Run this any time, from another terminal, while the crawlers are working. It
only opens the SQLite state databases read-only and streams the JSONL shards,
so it never interferes with a running crawl.

    python scripts/crawl_status.py                 # both languages
    python scripts/crawl_status.py --lang nepali   # one language
    python scripts/crawl_status.py --watch 300     # refresh every 5 minutes

WHY "DONE" IS NOT THE SAME AS "COLLECTED"
-----------------------------------------
A URL leaves the frontier for several reasons, and only one of them produces a
document. robots.txt disallowals in particular can retire hundreds of thousands
of URLs while yielding nothing at all, which makes a raw "done" count wildly
optimistic. This report therefore separates:

    discovered  -- URLs found in sitemaps
    processed   -- URLs the crawler has finished with, for any reason
    skipped     -- retired without a document (robots.txt, too short, too long)
    articles    -- documents actually written to disk

The token projection is built from `articles` and measured mean article length,
never from `processed`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Characters per token. Telugu and Nepali are both morphologically rich, so a
# subword vocabulary in the tens of thousands typically lands in this band.
# Reported as a range because the true value is unknown until the Phase 1
# tokenizer sweep finishes -- do not present a single number as if it were one.
CHARS_PER_TOKEN = (3.0, 3.5)

# Fraction of remaining pending URLs expected to become real articles, measured
# from the crawl so far rather than assumed.
LISTING_PAGE_CHARS = 50_000


def read_state(lang: str) -> tuple[list[tuple], dict]:
    """Read per-site crawl state from the SQLite databases.

    Args:
        lang: Language directory name, e.g. "telugu".

    Returns:
        A tuple ``(rows, totals)`` where rows are per-site tuples of
        ``(site, total, processed, pending, skipped)``.
    """
    rows = []
    totals = {"total": 0, "processed": 0, "pending": 0, "skipped": 0}
    for db in sorted(glob.glob(f"data/{lang}/state/*.db")):
        site = os.path.basename(db)[:-3]
        try:
            # Read-only URI so we never take a write lock on a live crawl.
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            counts = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM urls GROUP BY status"
                ).fetchall()
            )
            conn.close()
        except sqlite3.Error:
            counts = {}
        total = sum(v for v in counts.values() if isinstance(v, int))
        pending = counts.get("pending", 0)
        skipped = counts.get("skipped", 0)
        rows.append((site, total, total - pending, pending, skipped))
        totals["total"] += total
        totals["processed"] += total - pending
        totals["pending"] += pending
        totals["skipped"] += skipped
    return rows, totals


def read_corpus(lang: str) -> dict:
    """Stream the JSONL shards and summarize what has actually been collected.

    Args:
        lang: Language directory name.

    Returns:
        Dictionary with document counts, character totals, and per-source
        breakdown. Listing pages (over ``LISTING_PAGE_CHARS``) are excluded,
        since they are dropped by the cleaning stage.
    """
    docs = 0
    chars = 0
    listing = 0
    per_source: dict[str, list[int]] = {}
    for f in glob.glob(f"data/{lang}/raw/*.jsonl"):
        try:
            fh = open(f, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # A shard being appended to right now may end mid-line.
                    continue
                n = len(d.get("text", ""))
                if n > LISTING_PAGE_CHARS:
                    listing += 1
                    continue
                docs += 1
                chars += n
                slot = per_source.setdefault(d.get("source", "?"), [0, 0])
                slot[0] += 1
                slot[1] += n
    return {
        "docs": docs,
        "chars": chars,
        "listing_pages": listing,
        "per_source": per_source,
    }


def report(lang: str) -> None:
    """Print a full progress report for one language.

    Args:
        lang: Language directory name.
    """
    rows, totals = read_state(lang)
    corpus = read_corpus(lang)

    print("=" * 74)
    print(f"  {lang.upper()}")
    print("=" * 74)

    if not rows:
        print("  No crawl state yet. Run --discover first.\n")
        return

    print(f"\n  {'site':18s}{'found':>10s}{'processed':>11s}{'pending':>10s}"
          f"{'articles':>10s}")
    print("  " + "-" * 57)
    src = corpus["per_source"]
    rows.sort(key=lambda r: -r[3])
    for site, total, processed, pending, skipped in rows:
        arts = src.get(site, [0, 0])[0]
        # A site that retired every URL without producing a document is almost
        # always robots.txt-blocked; flag it rather than let it read as progress.
        flag = ""
        if total > 1000 and processed == total and arts == 0:
            flag = "  <-- BLOCKED (robots.txt)"
        elif total == 0:
            flag = "  <-- no sitemap"
        print(f"  {site:18s}{total:>10,}{processed:>11,}{pending:>10,}"
              f"{arts:>10,}{flag}")

    print("  " + "-" * 57)
    print(f"  {'TOTAL':18s}{totals['total']:>10,}{totals['processed']:>11,}"
          f"{totals['pending']:>10,}{corpus['docs']:>10,}")

    # ---- corpus size -----------------------------------------------------
    docs, chars = corpus["docs"], corpus["chars"]
    mean = chars // docs if docs else 0
    print(f"\n  COLLECTED SO FAR")
    print(f"    articles          {docs:>12,}")
    print(f"    characters        {chars:>12,}")
    print(f"    mean article      {mean:>12,} chars")
    if corpus["listing_pages"]:
        print(f"    listing pages     {corpus['listing_pages']:>12,}  "
              f"(excluded; dropped by cleaning)")
    if docs:
        lo = chars / CHARS_PER_TOKEN[1] / 1e6
        hi = chars / CHARS_PER_TOKEN[0] / 1e6
        print(f"    ~tokens           {lo:>9.1f}M - {hi:.1f}M")

    # ---- projection ------------------------------------------------------
    pending = totals["pending"]
    if pending and docs:
        # Yield measured from this crawl: articles per URL actually processed.
        # Excludes robots-blocked sites, whose URLs are already retired and so
        # cannot contribute to the remaining pending count.
        active_processed = sum(
            p for site, t, p, pd, sk in rows if src.get(site, [0])[0] > 0
        )
        yield_rate = docs / active_processed if active_processed else 0.75
        proj_docs = docs + pending * yield_rate
        proj_chars = proj_docs * mean
        lo = proj_chars / CHARS_PER_TOKEN[1] / 1e6
        hi = proj_chars / CHARS_PER_TOKEN[0] / 1e6
        print(f"\n  PROJECTED AT COMPLETION")
        print(f"    measured yield    {yield_rate:>12.1%}  (articles per URL processed)")
        print(f"    articles          {proj_docs:>12,.0f}")
        print(f"    ~tokens           {lo:>9.1f}M - {hi:.1f}M")
        print(f"    (before deduplication, which typically removes 15-25%)")

    print()


def main() -> None:
    """Parse arguments and print the report."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", choices=["telugu", "nepali"], default=None,
                   help="Report on one language only. Default: both.")
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                   help="Refresh continuously every N seconds.")
    args = p.parse_args()

    langs = [args.lang] if args.lang else ["telugu", "nepali"]

    while True:
        if args.watch:
            # Clear screen between refreshes so the terminal stays readable.
            print("\033[2J\033[H", end="")
            print(f"  refreshed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for lang in langs:
            report(lang)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

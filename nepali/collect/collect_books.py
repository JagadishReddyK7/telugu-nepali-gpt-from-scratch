"""Collect Nepali book text from the Internet Archive (manual collection).

    python -m nepali.collect.collect_books
    python -m nepali.collect.collect_books --max-items 300   # smaller trial

Writes chunked OCR text to ``data/nepali/raw/books-*.jsonl`` tagged
``provenance="manual_scrape"``, because the project spec counts "OCR from
books/PDFs" as manual collection.

Resumable: processed item identifiers are recorded in
``data/nepali/books_progress.json`` and skipped on re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.archive_books import BookCollectorConfig, collect
from common.run_crawl import setup_logging


def main() -> None:
    """Parse arguments and collect Nepali books."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-items", type=int, default=2000,
                   help="Maximum volumes to fetch.")
    p.add_argument("--target-chars", type=int, default=250_000_000,
                   help="Stop once this many characters are collected.")
    p.add_argument("--delay", type=float, default=1.5,
                   help="Seconds between requests. Archive.org is a nonprofit; "
                        "do not lower this much.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    cfg = BookCollectorConfig(
        language="nepali",
        archive_language="Nepali",
        language_aliases=("nep", "Nepalese", "ne"),
        script="devanagari",
        max_items=args.max_items,
        target_chars=args.target_chars,
        delay_seconds=args.delay,
    )
    stats = collect(cfg, Path("data/nepali"))
    print(f"\nNepali books: {stats['items_kept']:,} volumes -> "
          f"{stats['chunks']:,} chunks, {stats['chars']:,} characters")
    print(f"  (~{stats['chars']/3.2/1e6:.0f}M tokens manual, which unlocks "
          f"~{stats['chars']*5/3.2/1e6:.0f}M tokens of final corpus)")


if __name__ == "__main__":
    main()

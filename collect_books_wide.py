"""Re-run archive.org book collection with a much wider alias net.

    python collect_books_wide.py --lang nepali
    python collect_books_wide.py --lang telugu

WHY ALIASES MATTER MORE THAN ANY OTHER KNOB HERE
------------------------------------------------
Archive.org language metadata is contributor-supplied and inconsistent: the same
language appears as a full name, an ISO 639-1 code, an ISO 639-2 code, an older
exonym, and common misspellings, across different collections. Searching one
spelling silently misses records. On the first pass ``Nepali`` returned 240 items
while ``nep`` returned 1,627 -- a 7.8x difference for the same corpus.

This run adds every alias worth trying. Identifiers already processed are
recorded in ``data/<lang>/books_progress.json`` and skipped, so re-running costs
nothing for volumes already fetched -- it only pulls what the narrower search
never saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.archive_books import BookCollectorConfig, collect
from common.run_crawl import setup_logging

ALIASES = {
    "nepali": ("nep", "ne", "npi", "Nepalese", "Nepali language",
               "Nepalese language", "Gorkhali", "Khas Kura", "nepali",
               "Nepali (macrolanguage)", "Nepali, Nepalese"),
    "telugu": ("tel", "te", "Telugu language", "Telegu", "telugu",
               "Telugu (Andhra)", "Andhra", "Telugu language texts"),
}
SCRIPTS = {"nepali": "devanagari", "telugu": "telugu"}
PRIMARY = {"nepali": "Nepali", "telugu": "Telugu"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True, choices=("nepali", "telugu"))
    p.add_argument("--max-items", type=int, default=8000)
    p.add_argument("--target-chars", type=int, default=500_000_000)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    cfg = BookCollectorConfig(
        language=args.lang,
        archive_language=PRIMARY[args.lang],
        language_aliases=ALIASES[args.lang],
        script=SCRIPTS[args.lang],
        max_items=args.max_items,
        target_chars=args.target_chars,
        delay_seconds=args.delay,
    )
    stats = collect(cfg, Path(f"data/{args.lang}"))
    print(f"\n{args.lang} books: {stats['items_kept']:,} volumes -> "
          f"{stats['chunks']:,} chunks, {stats['chars']:,} characters")


if __name__ == "__main__":
    main()

"""
Reconstruct ``clean_stats.json`` from the cleaned splits after a crashed run.

    python scripts/rebuild_clean_stats.py --lang telugu --log telugu_clean_v2.log

WHY THIS IS NEEDED
------------------
``CorpusCleaner.run`` wrote ``clean_stats.json`` *after* its try/except/finally
block. The ``finally`` closed the split writers, so a run that died late left
train/validation/test.jsonl complete and usable on disk while the statistics
file was never created and the exception propagated into ``tee`` — whose exit
code is what the shell's ``&&`` then tested, making the failure look like a
success.

WHAT IS RECOVERABLE AND WHAT IS NOT
-----------------------------------
Everything on the KEPT side is recoverable exactly, because it is a property of
the documents that were written: document counts, character and byte totals,
the per-source and per-provenance breakdowns, and the split sizes.

The REJECTED side is not. Those counters lived only in memory and died with the
process. ``rejected_by_reason`` is therefore emitted empty and flagged, rather
than being filled with plausible-looking numbers. ``documents_read`` is taken
from the last progress line in the run log, which makes it a LOWER BOUND — the
run read at least that many documents before it stopped, and the true figure is
unknown. Both facts are recorded in the output so the report cannot quote them
as if they were measured.

A truncated final line, which is what a SIGKILL leaves behind, is counted and
reported rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
PROGRESS_RE = re.compile(r"read\s+([\d,]+)\s+kept\s+([\d,]+)")


def scan_log(path: Path) -> dict:
    """Extract the highest read/kept pair from a cleaning run's log.

    Args:
        path: Log file written by ``run_clean``.

    Returns:
        Dict with ``documents_read`` and ``documents_kept`` as seen by the last
        progress line, or an empty dict if nothing parseable was found.
    """
    if not path or not path.exists():
        return {}
    best = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = PROGRESS_RE.search(line)
        if m:
            best = {
                "documents_read": int(m.group(1).replace(",", "")),
                "documents_kept": int(m.group(2).replace(",", "")),
            }
    return best


def scan_splits(clean_dir: Path) -> dict:
    """Recompute every kept-side statistic by streaming the written splits.

    Args:
        clean_dir: Directory holding train/validation/test.jsonl.

    Returns:
        Counters and totals mirroring ``CleanStats.as_dict``'s kept-side fields.
    """
    kept = chars = nbytes = 0
    per_source: Counter = Counter()
    per_provenance: Counter = Counter()
    chars_per_provenance: Counter = Counter()
    split_docs: Counter = Counter()
    split_chars: Counter = Counter()
    malformed = 0

    for split in SPLITS:
        path = clean_dir / f"{split}.jsonl"
        if not path.exists():
            logger.warning("%s missing — skipping", path.name)
            continue
        logger.info("scanning %s (%.2f GB) ...",
                    path.name, path.stat().st_size / 1e9)
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed mid-write leaves one truncated line.
                    malformed += 1
                    continue
                text = doc.get("text") or ""
                prov = doc.get("provenance", "unknown")
                source = doc.get("source", "unknown")

                kept += 1
                chars += len(text)
                nbytes += len(text.encode("utf-8"))
                per_source[source] += 1
                per_provenance[prov] += 1
                chars_per_provenance[prov] += len(text)
                split_docs[split] += 1
                split_chars[split] += len(text)

        logger.info("  %s: %s docs", split, f"{split_docs[split]:,}")

    return {
        "documents_kept": kept,
        "chars_kept": chars,
        "bytes_kept": nbytes,
        "documents_per_source": dict(per_source.most_common()),
        "documents_per_provenance": dict(per_provenance),
        "chars_per_provenance": dict(chars_per_provenance),
        "manual_char_fraction": round(
            chars_per_provenance.get("manual_scrape", 0) / max(chars, 1), 4),
        "split_documents": dict(split_docs),
        "split_chars": dict(split_chars),
        "malformed_lines_encountered": malformed,
    }


def main() -> None:
    """Rebuild and write ``clean_stats.json`` for one language."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=["telugu", "nepali"])
    p.add_argument("--log", type=Path, default=None,
                   help="Run log, used to recover documents_read as a lower bound.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing clean_stats.json.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    clean_dir = Path(f"data/{args.lang}/clean")
    out = clean_dir / "clean_stats.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists. Use --force to overwrite.")

    stats = scan_splits(clean_dir)
    from_log = scan_log(args.log)

    stats["documents_read"] = from_log.get("documents_read", 0)
    stats["keep_rate"] = (round(stats["documents_kept"] / stats["documents_read"], 4)
                          if stats["documents_read"] else 0.0)
    stats["rejected_by_reason"] = {}
    stats["seconds"] = 0.0
    stats["reconstructed"] = True
    stats["reconstruction_note"] = (
        "Rebuilt from the written splits by scripts/rebuild_clean_stats.py after "
        "the cleaning run terminated before writing clean_stats.json. All "
        "kept-side figures (document counts, characters, bytes, per-source and "
        "per-provenance breakdowns, split sizes) are exact — they are recomputed "
        "from the documents themselves. 'documents_read' is a LOWER BOUND taken "
        "from the last progress line in the run log; the true value is unknown. "
        "'rejected_by_reason' is empty because those counters existed only in "
        "memory and were lost with the process; it is not reported as zero "
        "rejections. 'keep_rate' inherits the lower-bound caveat and is an "
        "UPPER bound on the true keep rate."
    )

    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print()
    print(f"  wrote {out}")
    print(f"    documents kept        {stats['documents_kept']:,}")
    print(f"    characters kept       {stats['chars_kept']:,}")
    print(f"    manual chars          "
          f"{stats['chars_per_provenance'].get('manual_scrape', 0):,}")
    print(f"    manual char fraction  {stats['manual_char_fraction']:.2%}")
    print(f"    split documents       {stats['split_documents']}")
    if stats["malformed_lines_encountered"]:
        print(f"    truncated/bad lines   {stats['malformed_lines_encountered']} "
              f"(expected: 1, from the kill)")
    if stats["documents_read"]:
        print(f"    documents read        >= {stats['documents_read']:,} "
              f"(lower bound, from log)")
    else:
        print("    documents read        UNKNOWN (no log supplied)")
    print()
    print("  rejected_by_reason is empty and flagged. Do not present the")
    print("  cleaning funnel for this language as if it were measured.")
    print()


if __name__ == "__main__":
    main()

"""
Shared entry point for the cleaning stage.

Both languages call this with their own YAML config, their own data directory,
and their own output splits. No document, hash, or LSH index is shared between
them -- each invocation builds its own from scratch.

    python -m telugu.clean.run_clean
    python -m nepali.clean.run_clean --limit 50000     # quick trial
    python -m telugu.clean.run_clean --no-langid       # skip fastText
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from common.clean_pipeline import CleanConfig, CorpusCleaner
from common.run_crawl import setup_logging

logger = logging.getLogger(__name__)


def load_clean_config(config_path: Path, use_langid: bool = True) -> CleanConfig:
    """Build a :class:`CleanConfig` from a language's YAML file.

    Args:
        config_path: Path to ``telugu.yaml`` or ``nepali.yaml``.
        use_langid: Whether to run fastText language identification.

    Returns:
        A populated :class:`CleanConfig`.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cleaning = raw.get("cleaning", {})
    splits = raw.get("splits", {})
    lang = raw["language"]
    return CleanConfig(
        script=lang["script"],
        iso_code=lang["iso_code"],
        min_script_ratio=cleaning.get("min_script_ratio", 0.80),
        min_langid_confidence=cleaning.get("min_langid_confidence", 0.60),
        min_chars=cleaning.get("min_chars", 200),
        max_chars=cleaning.get("max_chars", 50_000),
        min_mean_word_len=cleaning.get("min_mean_word_len", 2.0),
        max_mean_word_len=cleaning.get("max_mean_word_len", 30.0),
        max_dup_line_fraction=cleaning.get("max_dup_line_fraction", 0.30),
        minhash_threshold=cleaning.get("minhash_threshold", 0.80),
        minhash_num_perm=cleaning.get("minhash_num_perm", 128),
        shingle_size=cleaning.get("shingle_size", 5),
        train=splits.get("train", 0.98),
        validation=splits.get("validation", 0.01),
        test=splits.get("test", 0.01),
        seed=splits.get("seed", 1337),
        max_val_docs=splits.get("max_val_docs", 20_000),
        max_test_docs=splits.get("max_test_docs", 20_000),
        use_langid=use_langid,
    )


def run(language: str, config_path: Path, data_root: Path) -> None:
    """Run cleaning for one language and print a readable funnel summary.

    Args:
        language: Language name.
        config_path: Path to that language's YAML config.
        data_root: Directory containing ``raw/``.
    """
    p = argparse.ArgumentParser(description=f"Clean the {language} corpus.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N input documents (for a quick trial).")
    p.add_argument("--no-langid", action="store_true",
                   help="Skip fastText language ID. NOT recommended for Nepali: "
                        "it is the only filter that separates Nepali from Hindi.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    if args.no_langid and language == "nepali":
        logger.warning(
            "Running Nepali WITHOUT language identification. Devanagari is "
            "shared with Hindi, so the resulting corpus may be bilingual, "
            "which violates the project's monolingual constraint."
        )

    config = load_clean_config(config_path, use_langid=not args.no_langid)
    cleaner = CorpusCleaner(language, config, data_root)
    stats = cleaner.run(limit=args.limit)
    d = stats.as_dict()

    print("\n" + "=" * 66)
    print(f"  {language.upper()} -- cleaning funnel")
    print("=" * 66)
    print(f"  read            {d['documents_read']:>12,}")
    print(f"  kept            {d['documents_kept']:>12,}   "
          f"({d['keep_rate']:.1%})")
    print(f"\n  REJECTED BY REASON")
    for reason, n in d["rejected_by_reason"].items():
        pct = n / max(d["documents_read"], 1) * 100
        print(f"    {reason:34s}{n:>10,}  ({pct:5.2f}%)")

    print(f"\n  PROVENANCE (the >=20% manual requirement)")
    for prov, n in d["documents_per_provenance"].items():
        chars = d["chars_per_provenance"].get(prov, 0)
        print(f"    {prov:20s}{n:>10,} docs  {chars:>14,} chars")
    frac = d["manual_char_fraction"]
    verdict = "MEETS >=20%" if frac >= 0.20 else "BELOW 20% -- see report"
    print(f"    manual character share: {frac:.1%}   [{verdict}]")

    print(f"\n  SPLITS")
    for split in ("train", "validation", "test"):
        n = d["split_documents"].get(split, 0)
        c = d["split_chars"].get(split, 0)
        print(f"    {split:12s}{n:>10,} docs  {c:>14,} chars")

    print(f"\n  total characters {d['chars_kept']:,}")
    print(f"  total UTF-8 bytes {d['bytes_kept']:,}  (needed for Phase 2 BPB)")
    print(f"  elapsed {d['seconds']:.0f}s\n")

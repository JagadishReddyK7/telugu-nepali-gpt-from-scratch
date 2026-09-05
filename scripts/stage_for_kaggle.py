"""
Build the Kaggle upload archive from the Phase 1 tokenized tarballs.

    python scripts/stage_for_kaggle.py ^
        --telugu-tar C:\\Users\\you\\Downloads\\telugu_tokenized.tar.gz ^
        --nepali-tar C:\\Users\\you\\Downloads\\nepali_tokenized_v2.tar.gz

Produces ``kaggle_upload/lma-tokenized.zip`` -- ONE file to upload to Kaggle.

WHY A ZIP AND NOT LOOSE FILES
-----------------------------
Kaggle's browser uploader flattens directory structure, so uploading
``telugu/tokenized/train.bin`` and ``nepali/tokenized/train.bin`` as loose files
gives you two files called ``train.bin`` in the same folder, one of which
silently overwrites the other. Kaggle *does* extract an uploaded ``.zip`` and
preserve the paths inside it, so a single archive is both simpler and safer.

WHY THE SIZE CHECK MATTERS
--------------------------
There are two Nepali archives on Drive: ``nepali_tokenized.tar.gz`` (the
superseded 493.9M-token corpus with no Wikipedia) and
``nepali_tokenized_v2.tar.gz`` (511.3M tokens, all sources). They differ by one
suffix and produce a model trained on the wrong corpus with no error anywhere.

Every ``.bin`` is a flat array of uint16 token ids, so its size in bytes is
exactly ``2 x token_count``. This script cross-checks each file against the
``token_counts.json`` packed alongside it and refuses to build the archive on a
mismatch -- which is what catches the v1/v2 mixup before 2GB goes over the wire
rather than after a training run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

# What the Kaggle notebook expects to find under /kaggle/input/lma-tokenized/
LAYOUT = {
    "telugu": {
        "bins": ["train.bin", "validation.bin", "test.bin"],
        "tokenizer": ["telugu_sp_16000.model", "telugu_sp_16000.vocab"],
    },
    "nepali": {
        "bins": ["train.bin", "validation.bin", "test.bin"],
        "tokenizer": ["nepali_sp_16000.model", "nepali_sp_16000.vocab"],
    },
}


def extract(tar_path: Path, dest: Path) -> None:
    """Extract a tarball, refusing paths that escape the destination.

    Args:
        tar_path: Archive to extract.
        dest: Destination directory.

    Raises:
        SystemExit: If the archive is missing or contains an unsafe member.
    """
    if not tar_path.exists():
        raise SystemExit(f"not found: {tar_path}")
    print(f"  extracting {tar_path.name} ({tar_path.stat().st_size/1e6:,.0f} MB) ...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise SystemExit(f"unsafe path in archive: {member.name}")
        # filter="data" is the safe extraction policy and becomes the default in
        # Python 3.14; passing it explicitly silences the deprecation warning
        # and keeps behaviour identical across versions.
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)


def find_one(root: Path, name: str) -> Path | None:
    """Return the first file called ``name`` anywhere under ``root``."""
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def verify_bin(path: Path, expected_tokens: int) -> tuple[bool, str]:
    """Check that a ``.bin`` holds exactly the expected number of uint16 tokens.

    Args:
        path: The ``.bin`` file.
        expected_tokens: Token count from ``token_counts.json``.

    Returns:
        ``(ok, message)``.
    """
    size = path.stat().st_size
    actual = size // 2
    if size % 2:
        return False, f"odd byte count {size:,} -- not a uint16 array"
    if actual != expected_tokens:
        return (False, f"{actual:,} tokens, expected {expected_tokens:,} "
                       f"(difference {actual - expected_tokens:+,})")
    return True, f"{actual:,} tokens"


def main() -> None:
    """Stage, verify and zip the tokenized corpora for Kaggle."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--telugu-tar", type=Path, required=True,
                   help="Path to telugu_tokenized.tar.gz downloaded from Drive.")
    p.add_argument("--nepali-tar", type=Path, required=True,
                   help="Path to nepali_tokenized_v2.tar.gz (NOTE the _v2).")
    p.add_argument("--out", type=Path, default=Path("kaggle_upload"),
                   help="Staging directory (default: kaggle_upload/).")
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Repository root, used to source the tokenizer .model/"
                        ".vocab files when the tarball does not carry them "
                        "(default: current directory).")
    p.add_argument("--keep-staging", action="store_true",
                   help="Keep the unzipped staging tree after building the zip.")
    args = p.parse_args()

    out = args.out
    work = out / "_extract"
    stage = out / "lma-tokenized"
    for d in (work, stage):
        if d.exists():
            shutil.rmtree(d)
    stage.mkdir(parents=True)

    print("\n1. extracting archives")
    extract(args.telugu_tar, work / "telugu")
    extract(args.nepali_tar, work / "nepali")

    print("\n2. collecting files and verifying token counts")
    problems: list[str] = []
    total_bytes = 0

    for lang, spec in LAYOUT.items():
        src_root = work / lang
        counts_file = find_one(src_root, "token_counts.json")
        if counts_file is None:
            problems.append(f"{lang}: token_counts.json not found in the archive")
            continue
        counts = json.loads(counts_file.read_text(encoding="utf-8"))
        splits = counts.get("splits", {})
        print(f"\n  {lang}  (tokenizer {counts.get('tokenizer_model')}, "
              f"{counts.get('total_tokens', 0):,} tokens total)")

        (stage / lang / "tokenized").mkdir(parents=True, exist_ok=True)
        (stage / lang / "tokenizer").mkdir(parents=True, exist_ok=True)

        for b in spec["bins"]:
            src = find_one(src_root, b)
            if src is None:
                problems.append(f"{lang}/{b}: not found in the archive")
                continue
            split = b.replace(".bin", "")
            expected = splits.get(split, {}).get("tokens")
            if expected is None:
                problems.append(f"{lang}/{b}: no token count for split '{split}'")
                continue
            ok, msg = verify_bin(src, expected)
            print(f"    {'OK  ' if ok else 'FAIL'} {b:16s} {msg}")
            if not ok:
                problems.append(f"{lang}/{b}: {msg}")
                continue
            shutil.copy2(src, stage / lang / "tokenized" / b)
            total_bytes += src.stat().st_size

        shutil.copy2(counts_file, stage / lang / "tokenized" / "token_counts.json")

        # The tokenizer files may or may not be inside the tarball, depending on
        # which Colab cell built it -- the Nepali v2 archive carries only the
        # tokenized bins. Fall back to the repository copies, which are the
        # committed Phase 1 deliverables and are therefore authoritative.
        for t in spec["tokenizer"]:
            src = find_one(src_root, t)
            origin = "archive"
            if src is None:
                repo_copy = args.repo_root / lang / "tokenizer" / t
                if repo_copy.exists():
                    src, origin = repo_copy, "repo"
            if src is None:
                problems.append(
                    f"{lang}/{t}: not in the archive and not at "
                    f"{args.repo_root / lang / 'tokenizer' / t}")
                continue
            shutil.copy2(src, stage / lang / "tokenizer" / t)
            print(f"    OK   {t:28s} (from {origin})")

    if problems:
        print("\nFAILED -- archive not built:\n")
        for msg in problems:
            print(f"  * {msg}")
        print("\nIf a Nepali count is ~493.9M rather than 511.3M you extracted "
              "nepali_tokenized.tar.gz instead of nepali_tokenized_v2.tar.gz.")
        sys.exit(1)

    print(f"\n3. building the zip ({total_bytes/1e9:.2f} GB of .bin to compress, "
          f"this takes a few minutes)")
    zip_path = out / "lma-tokenized.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage))

    shutil.rmtree(work)
    if not args.keep_staging:
        shutil.rmtree(stage)

    print(f"\nDONE  {zip_path}  ({zip_path.stat().st_size/1e9:.2f} GB)")
    print("\nUpload that ONE file to Kaggle:")
    print("  kaggle.com > Datasets > New Dataset > drag lma-tokenized.zip")
    print("  Title it exactly:  lma-tokenized")
    print("\nKaggle extracts the zip, so the notebook will see:")
    for lang in LAYOUT:
        print(f"  /kaggle/input/lma-tokenized/{lang}/tokenized/train.bin")
        print(f"  /kaggle/input/lma-tokenized/{lang}/tokenizer/{lang}_sp_16000.model")


if __name__ == "__main__":
    main()

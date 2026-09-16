"""
Build ``kaggle_upload/lma-code.zip`` -- the code dataset for Kaggle.

    python scripts/stage_code_for_kaggle.py

Run this from the repository root every time the code changes, then upload the
zip as a NEW VERSION of the ``lma-code`` dataset.

WHY NOT PowerShell's Compress-Archive
-------------------------------------
``Compress-Archive`` writes Windows path separators into the archive, producing
member names like ``common\\model.py``. The ZIP specification requires forward
slashes, and Kaggle rejects such an archive with:

    'common\\model.py' contains a forbidden character in name ('\\')

Python's ``zipfile`` normalizes ``os.sep`` to ``/`` when it builds each
``ZipInfo``, so an archive written here is portable by construction.

WHAT IT EXCLUDES AND WHY
------------------------
``__pycache__`` and ``.pyc`` files are stripped. A stale ``.pyc`` can shadow the
source file it was compiled from, so shipping them risks Kaggle importing an
older version of a module you just fixed -- a failure that looks like the fix
never worked rather than like a packaging problem.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Directories the Kaggle notebook copies out of /kaggle/input/lma-code/.
INCLUDE_DIRS = ["common", "telugu", "nepali", "scripts", "tests"]

EXCLUDE_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
# The Phase 3 reasoning splits are regenerated on Kaggle in two seconds by
# `python -m <lang>.reasoning.generate` (deterministic); shipping 60MB of
# JSONL in every code upload buys nothing.
EXCLUDE_DIR_NAMES |= {"dataset"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

# Markers proving the archive carries the current code rather than an earlier
# revision. Each one is a fix that is invisible until it is needed.
REQUIRED_MARKERS = {
    "common/model.py": [
        ("attention written out, not F.scaled_dot_product_attention",
         lambda s: "scaled_dot_product_attention(" not in s),
        ("causal mask present", lambda s: "causal_mask" in s),
    ],
    "common/train.py": [
        ("CUDA RNG saved (exact resume with dropout on GPU)",
         lambda s: "cuda_rng_state" in s),
        ("--restore-from (Kaggle read-only resume)",
         lambda s: "--restore-from" in s),
        ("no per-micro-batch GPU sync", lambda s: "step_loss_t" in s),
        ("ablation gets its own log file",
         lambda s: "follows --out-dir" in s),
    ],
    "common/data.py": [
        ("chunked BPB byte count", lambda s: "CHUNK" in s),
    ],
    # Phase 3
    "common/finetune.py": [
        ("--session-steps (clean stop without touching the schedule)",
         lambda s: "session_steps" in s),
        ("explicit answer mask for answer-only loss", lambda s: "a.float()" in s),
    ],
    "common/reasoning_eval.py": [
        ("bucketed generation", lambda s: "bucket" in s.lower()),
    ],
    "common/attention_compare.py": [
        ("piece offsets without proto API", lambda s: "def piece_offsets" in s),
    ],
    "scripts/run_phase3.py": [
        ("--pretrained-root", lambda s: "--pretrained-root" in s),
    ],
    "telugu/reasoning/lexicon.py": [("present", lambda s: "LEXICON" in s)],
    "nepali/reasoning/lexicon.py": [("present", lambda s: "LEXICON" in s)],
}


def should_skip(path: Path) -> bool:
    """True if this path is build noise rather than source."""
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True
    return path.suffix in EXCLUDE_SUFFIXES


def main() -> None:
    """Verify the working tree, then zip it for upload."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("kaggle_upload/lma-code.zip"))
    args = p.parse_args()

    root = args.repo_root.resolve()

    print("1. checking the working tree is current")
    problems = []
    for rel, markers in REQUIRED_MARKERS.items():
        f = root / rel
        if not f.exists():
            problems.append(f"{rel}: missing")
            continue
        text = f.read_text(encoding="utf-8")
        for label, check in markers:
            ok = check(text)
            print(f"   {'OK  ' if ok else 'FAIL'} {rel:18s} {label}")
            if not ok:
                problems.append(f"{rel}: {label}")

    missing_dirs = [d for d in INCLUDE_DIRS if not (root / d).is_dir()]
    if missing_dirs:
        problems.append(f"missing directories: {missing_dirs}")

    if problems:
        print("\nFAILED -- archive not built:\n")
        for msg in problems:
            print(f"  * {msg}")
        raise SystemExit(1)

    print("\n2. building the archive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    n_files = 0
    per_dir: dict[str, int] = {}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for d in INCLUDE_DIRS:
            for f in sorted((root / d).rglob("*")):
                if not f.is_file() or should_skip(f.relative_to(root)):
                    continue
                # as_posix() guarantees forward slashes regardless of platform.
                z.write(f, f.relative_to(root).as_posix())
                n_files += 1
                per_dir[d] = per_dir.get(d, 0) + 1

    for d in INCLUDE_DIRS:
        print(f"   {d:10s} {per_dir.get(d, 0):>4} files")

    print("\n3. verifying the archive is portable")
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        print(f"   FAIL {len(bad)} member(s) contain backslashes")
        raise SystemExit(1)
    print(f"   OK   all {len(names)} member names use forward slashes")
    for probe in ("common/model.py", "common/train.py", "tests/test_model.py"):
        print(f"   {'OK  ' if probe in names else 'FAIL'} {probe}")

    size = args.out.stat().st_size
    print(f"\nDONE  {args.out}  ({size/1e6:.1f} MB, {n_files} files)")
    print("\nUpload as a NEW VERSION of the lma-code dataset:")
    print("  kaggle.com > Datasets > lma-code > New Version > drag the zip")
    print("\nFirst time: New Dataset, title it exactly  lma-code")


if __name__ == "__main__":
    main()

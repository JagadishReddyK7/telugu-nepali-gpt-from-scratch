"""
Identify, sort and verify downloaded Kaggle checkpoint archives.

    python scripts/sort_checkpoints.py --src "%USERPROFILE%\\Downloads" --out ckpt_staging

Kaggle wraps each large output file in its own ``.zip`` whose name says only
``best`` or ``latest``. Nothing in the filename says which MODEL it came from,
and two runs downloaded on different days land in the same folder as
``best.zip`` and ``best (1).zip``. Guessing from timestamps is how Model H's
results end up published as Model L's.

Every checkpoint written by ``common.train`` embeds the full config, and the
config carries ``language``. This script reads that and sorts on it, so the
labelling is derived from the file's own contents rather than from its name.

WHY A .pt MUST NEVER BE EXTRACTED
---------------------------------
Since PyTorch 1.6 ``torch.save`` writes a ZIP container: a pickle plus the raw
tensor storages as separate members. Windows and 7-Zip see the ``PK`` magic and
happily unpack it into a DIRECTORY containing ``data.pkl``, ``data/0``,
``data/1`` and ``version``. That directory is not loadable, and rebuilding a
working archive by re-zipping it is unreliable. This script extracts exactly
one level -- Kaggle's wrapper -- and treats the ``.pt`` inside as opaque.

IDENTIFICATION WITHOUT TORCH
----------------------------
``torch`` is used when it is importable, because it gives the exact step and
token count. When it is not, the language is still recovered by scanning the
pickle stream inside the archive for the language name, which is sufficient to
sort the files correctly. A missing ``torch`` therefore degrades the report,
not the outcome.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

LANGUAGES = ("telugu", "nepali")
KINDS = ("best", "latest")

# 25,860,480 params x 4 bytes x 3 (weights + AdamW's two moments), plus the
# pickle and RNG state. Anything far from this is not a full checkpoint.
EXPECTED_BYTES = 310_000_000
TOLERANCE = 0.08


def kind_of(name: str) -> str | None:
    """Infer ``best`` / ``latest`` from an archive's filename.

    Args:
        name: Archive filename, e.g. ``"best (1).zip"``.

    Returns:
        The kind, or None if the name matches neither.
    """
    low = name.lower()
    for k in KINDS:
        if low.startswith(k):
            return k
    return None


def extract_one_level(zip_path: Path, dest: Path) -> Path:
    """Unpack Kaggle's wrapper and return the single ``.pt`` inside.

    Args:
        zip_path: The downloaded ``.zip``.
        dest: Directory to extract into.

    Returns:
        Path to the extracted ``.pt`` file.

    Raises:
        SystemExit: If the archive holds no ``.pt``, or more than one.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".pt")]
        if len(members) != 1:
            raise SystemExit(
                f"{zip_path.name}: expected exactly one .pt inside, found "
                f"{len(members)}: {members[:5]}")
        z.extract(members[0], dest)
    return dest / members[0]


def identify(pt_path: Path) -> dict:
    """Read a checkpoint's identity from its own contents.

    Args:
        pt_path: Path to a ``.pt`` file.

    Returns:
        Dict with ``language``, ``positional``, ``step``, ``tokens_seen``,
        ``best_val_loss`` and ``bytes``. Fields other than ``language`` and
        ``bytes`` are None when torch is unavailable.

    Raises:
        SystemExit: If the file is not a torch zip archive, or its language
            cannot be determined.
    """
    size = pt_path.stat().st_size
    if not zipfile.is_zipfile(pt_path):
        raise SystemExit(
            f"{pt_path.name} is not a torch archive. If it is a FOLDER, it was "
            f"extracted one level too far -- delete it and re-extract.")

    info = {"language": None, "positional": None, "step": None,
            "tokens_seen": None, "best_val_loss": None, "bytes": size}

    try:
        import torch
        ck = torch.load(pt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        info.update(language=cfg.get("language"),
                    positional=cfg.get("model", {}).get("positional"),
                    step=ck.get("step"), tokens_seen=ck.get("tokens_seen"),
                    best_val_loss=ck.get("best_val_loss"))
    except ImportError:
        # Fall back to scanning the pickle stream for the language name. The
        # config dict is pickled verbatim, so the string is present as bytes.
        with zipfile.ZipFile(pt_path) as z:
            names = [n for n in z.namelist() if n.endswith("data.pkl")]
            blob = z.read(names[0]) if names else b""
        hits = [l for l in LANGUAGES if l.encode() in blob]
        if len(hits) == 1:
            info["language"] = hits[0]

    if info["language"] not in LANGUAGES:
        raise SystemExit(
            f"{pt_path.name}: could not determine the language from the "
            f"checkpoint (got {info['language']!r}). Install torch and retry: "
            f"pip install torch --index-url https://download.pytorch.org/whl/cpu")
    return info


def main() -> None:
    """CLI: sort every downloaded checkpoint archive into per-language folders."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="Folder holding the downloaded best*.zip / latest*.zip.")
    p.add_argument("--out", type=Path, default=Path("ckpt_staging"),
                   help="Destination; <out>/<language>/<kind>.pt is written.")
    args = p.parse_args()

    zips = sorted(q for q in args.src.glob("*.zip") if kind_of(q.name))
    if not zips:
        raise SystemExit(f"no best*.zip / latest*.zip found in {args.src}")

    print(f"\n1. found {len(zips)} archive(s) in {args.src}\n")
    tmp = Path(tempfile.mkdtemp(prefix="ckpt_"))
    resolved: dict[tuple[str, str], tuple[Path, dict]] = {}
    try:
        for z in zips:
            kind = kind_of(z.name)
            pt = extract_one_level(z, tmp / z.stem)
            info = identify(pt)
            key = (info["language"], kind)
            print(f"   {z.name:20s} -> {info['language']:>6s} / {kind:6s}  "
                  f"step {info['step'] or '?':>8}  "
                  f"tokens {info['tokens_seen'] or '?':>12}  "
                  f"pos={info['positional'] or '?'}")
            if key in resolved:
                raise SystemExit(
                    f"\nTWO archives both identify as {key}: "
                    f"{resolved[key][0].name} and {z.name}. One of them is a "
                    f"duplicate download -- delete it and rerun.")
            resolved[key] = (z, info)

        print("\n2. verifying")
        problems = []
        for (lang, kind), (z, info) in sorted(resolved.items()):
            drift = abs(info["bytes"] - EXPECTED_BYTES) / EXPECTED_BYTES
            ok = drift <= TOLERANCE
            print(f"   {'OK  ' if ok else 'FAIL'} {lang}/{kind}.pt  "
                  f"{info['bytes']:,} bytes")
            if not ok:
                problems.append(
                    f"{lang}/{kind}: {info['bytes']:,} bytes, expected about "
                    f"{EXPECTED_BYTES:,} (weights + both AdamW moments)")

        missing = [f"{l}/{k}" for l in LANGUAGES for k in KINDS
                   if (l, k) not in resolved]
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if problems:
            print("\nFAILED:\n")
            for m in problems:
                print(f"  * {m}")
            raise SystemExit(1)

        print(f"\n3. staging into {args.out}")
        for (lang, kind), (z, _) in sorted(resolved.items()):
            pt = extract_one_level(z, tmp / z.stem)   # already extracted; cheap
            target = args.out / lang / f"{kind}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pt, target)
            print(f"   {target}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nDONE. Upload to Google Drive:")
    for lang in LANGUAGES:
        for kind in KINDS:
            print(f"   {args.out / lang / (kind + '.pt')}")
    print("\nAnd zip the same tree for Kaggle:")
    print(f"   Compress-Archive -Path {args.out}\\* "
          f"-DestinationPath lma-ckpts.zip -Force")


if __name__ == "__main__":
    main()

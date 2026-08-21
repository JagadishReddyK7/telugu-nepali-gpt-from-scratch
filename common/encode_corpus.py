"""
Encode the cleaned splits into flat ``uint16`` token arrays.

This produces two things at once:

1. The **exact** token counts the project spec asks you to report. Every number
   quoted before this point has been an estimate from a characters-per-token
   assumption; these are measured.

2. The binary format Phase 2 trains from. A contiguous ``uint16`` array read
   through ``numpy.memmap`` turns sampling a training window into a pointer
   offset, with the OS page cache doing the I/O. Anything richer -- a database,
   per-document records, JSON -- would bottleneck the GPU, because training
   samples random windows millions of times.

WHY uint16
----------
Any vocabulary below 65,536 fits in two bytes, so 500M tokens is exactly 1GB.
That is what makes the whole corpus fit in a 15GB Drive alongside checkpoints.
The encoder refuses to run if the vocabulary would overflow, rather than
silently truncating token ids and corrupting the corpus.

DOCUMENT BOUNDARIES
-------------------
Each document is written as ``<s> ... </s>``. The BOS/EOS markers let the model
learn where documents begin and end instead of treating the corpus as one
undifferentiated stream, and they give Phase 2 a clean way to reset context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def encode_split(
    jsonl_path: Path,
    model_path: Path,
    out_bin: Path,
    add_bos_eos: bool = True,
    report_every: int = 100_000,
) -> dict:
    """Tokenize one split and write it as a flat ``uint16`` binary.

    Args:
        jsonl_path: Cleaned split to encode.
        model_path: Trained SentencePiece model.
        out_bin: Destination ``.bin`` file.
        add_bos_eos: Wrap each document in ``<s>`` / ``</s>``.
        report_every: Log progress every N documents.

    Returns:
        Statistics for this split, including the exact token count.
    """
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()
    if vocab_size > 65535:
        raise ValueError(
            f"Vocabulary {vocab_size:,} exceeds what uint16 can hold. "
            "Use uint32 (and double the corpus size on disk) or a smaller vocab."
        )
    bos, eos = sp.bos_id(), sp.eos_id()

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_tokens = n_chars = n_bytes = 0
    # The spec requires the manual vs downloaded split in TOKENS, not
    # characters. This is the only full tokenization pass over the corpus, so
    # the split is accumulated here rather than in a separate script that would
    # have to re-encode the whole corpus to learn the same thing.
    tokens_by_provenance: dict[str, int] = {}
    docs_by_provenance: dict[str, int] = {}

    with out_bin.open("wb") as out:
        for line in jsonl_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue

            ids = sp.encode(text, out_type=int)
            if add_bos_eos:
                ids = [bos] + ids + [eos]

            np.asarray(ids, dtype=np.uint16).tofile(out)

            # Anything not explicitly tagged manual counts as downloaded, so a
            # missing or malformed provenance tag can only ever UNDERSTATE the
            # manual share. The compliance number is never flattered by a bad
            # record.
            prov = ("manual_scrape"
                    if record.get("provenance") == "manual_scrape"
                    else "downloaded")
            tokens_by_provenance[prov] = tokens_by_provenance.get(prov, 0) + len(ids)
            docs_by_provenance[prov] = docs_by_provenance.get(prov, 0) + 1

            n_docs += 1
            n_tokens += len(ids)
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))

            if n_docs % report_every == 0:
                logger.info("  %s: %s docs, %s tokens",
                            jsonl_path.stem, f"{n_docs:,}", f"{n_tokens:,}")

    stats = {
        "split": jsonl_path.stem,
        "documents": n_docs,
        "tokens": n_tokens,
        "characters": n_chars,
        "utf8_bytes": n_bytes,
        "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
        # Bytes per token is what Phase 2's bits-per-byte metric divides by. It
        # is the honest way to compare two models whose tokenizers compress
        # differently: perplexity per token is not comparable across
        # vocabularies, but loss per byte of source text is.
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
        "tokens_per_provenance": tokens_by_provenance,
        "documents_per_provenance": docs_by_provenance,
        "manual_token_fraction": round(
            tokens_by_provenance.get("manual_scrape", 0) / max(n_tokens, 1), 4),
        "bin_file": out_bin.name,
        "bin_size_bytes": out_bin.stat().st_size,
    }
    logger.info("[%s] %s docs -> %s tokens (%.1f MB on disk)",
                jsonl_path.stem, f"{n_docs:,}", f"{n_tokens:,}",
                stats["bin_size_bytes"] / 1e6)
    return stats


def encode_corpus(language: str, data_root: Path, model_path: Path) -> dict:
    """Encode all three splits for one language.

    Args:
        language: Language name.
        data_root: Directory containing ``clean/``.
        model_path: Chosen SentencePiece model.

    Returns:
        A report with per-split statistics and the exact total token count.
    """
    data_root = Path(data_root)
    clean_dir = data_root / "clean"
    out_dir = data_root / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split = {}
    for split in SPLITS:
        src = clean_dir / f"{split}.jsonl"
        if not src.exists():
            logger.warning("[%s] %s missing; skipping", language, src.name)
            continue
        per_split[split] = encode_split(
            src, model_path, out_dir / f"{split}.bin"
        )

    total_tokens = sum(s["tokens"] for s in per_split.values())
    total_bytes = sum(s["utf8_bytes"] for s in per_split.values())

    # ---- the compliance number ------------------------------------------
    # The spec requires ">=20% of the final TRAINING tokens" to be manually
    # collected, so the gate is computed on the train split alone. The
    # corpus-wide figure is reported alongside it for completeness, but train
    # is the one that decides pass or fail.
    train = per_split.get("train", {})
    train_prov = train.get("tokens_per_provenance", {})
    train_manual = train_prov.get("manual_scrape", 0)
    train_total = train.get("tokens", 0)

    all_manual = sum(s.get("tokens_per_provenance", {}).get("manual_scrape", 0)
                     for s in per_split.values())

    report = {
        "language": language,
        "tokenizer_model": model_path.name,
        "splits": per_split,
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "overall_chars_per_token": round(
            sum(s["characters"] for s in per_split.values())
            / max(total_tokens, 1), 4),
        "train_tokens": train_total,
        "train_manual_tokens": train_manual,
        "train_downloaded_tokens": train_prov.get("downloaded", 0),
        "train_manual_token_share": round(train_manual / max(train_total, 1), 4),
        "corpus_manual_tokens": all_manual,
        "corpus_manual_token_share": round(all_manual / max(total_tokens, 1), 4),
        "meets_20pct_manual_requirement":
            (train_manual / max(train_total, 1)) >= 0.20,
        "dtype": "uint16",
    }
    out = out_dir / "token_counts.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("[%s] EXACT TOTAL: %s tokens. Wrote %s",
                language, f"{total_tokens:,}", out)
    logger.info("[%s] train manual token share: %.2f%% (%s / %s) -- %s",
                language, report["train_manual_token_share"] * 100,
                f"{train_manual:,}", f"{train_total:,}",
                "MEETS >=20%" if report["meets_20pct_manual_requirement"]
                else "BELOW 20% -- REQUIREMENT NOT MET")
    return report

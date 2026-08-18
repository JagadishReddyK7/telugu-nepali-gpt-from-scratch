"""
Train a SentencePiece BPE tokenizer from scratch and choose its vocabulary size.

The spec forbids pretrained tokenizers, so both vocabularies are built here from
the cleaned corpus. Telugu and Nepali get entirely separate tokenizers; nothing
is shared, not even the training sample.

CHOOSING THE VOCABULARY SIZE IS A REAL TRADE-OFF, NOT A DEFAULT
---------------------------------------------------------------
A larger vocabulary means fewer tokens per word (better *fertility*), so a fixed
context window holds more text and the model has fewer steps to predict. But the
embedding matrix is ``vocab_size x d_model``, and it is tied to the output head,
so vocabulary is bought directly out of the ~25M parameter budget. At d_model
384, a 32k vocabulary costs 12.3M parameters -- roughly half the entire model,
leaving too little for the Transformer blocks that do the actual work.

So the sweep reports three things per candidate and lets the trade-off be seen:

    fertility          tokens per whitespace word on held-out text (lower better)
    chars per token    the same idea in characters (higher better)
    embedding params   vocab_size x d_model, i.e. what it costs

Selection picks the smallest vocabulary whose fertility is within a tolerance of
the best observed. Marginal fertility gains are not worth millions of parameters
that could be depth instead.

WHY FERTILITY DIFFERS BETWEEN THESE TWO LANGUAGES
-------------------------------------------------
Telugu is agglutinative with a large syllabic inventory: single orthographic
words carry case, number and postpositions that other languages write
separately. Nepali is inflectional but writes more of its grammar as separate
words. Telugu should therefore show *worse* fertility at equal vocabulary size --
more tokens per word -- which matters because it means Telugu needs more tokens
to express the same content. That is a genuine finding for the Phase 3 analysis
and a confound to control for when comparing perplexity across the two models.

BYTE FALLBACK
-------------
``byte_fallback=True`` decomposes anything unseen into UTF-8 bytes, so the
unknown-token rate is structurally zero and no input is ever unrepresentable.
The UNK statistics are still computed and reported -- a nonzero value would mean
something is wrong with the configuration.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Settings for one language's tokenizer sweep.

    Attributes:
        language: Language name, used for output paths.
        vocab_sizes: Candidate vocabulary sizes to train and compare.
        character_coverage: Fraction of characters the model must cover. 0.9995
            suits large-alphabet scripts; the residual tail is handled by byte
            fallback.
        model_type: SentencePiece algorithm. "bpe" per the project spec.
        split_digits: Split numbers into individual digits. Stops the tokenizer
            memorizing specific numbers as single units, which matters for the
            Phase 3 comparative-reasoning finetune where numeric comparison is
            the task.
        byte_fallback: Guarantee zero unknown tokens.
        train_sample_bytes: How much text to feed the trainer. SentencePiece
            holds its training data in memory, so this is bounded by Colab RAM
            rather than by corpus size.
        eval_docs: Held-out documents used to measure fertility.
        d_model_for_costing: Model dimension used only to report the embedding
            parameter cost of each vocabulary.
        fertility_tolerance: Accept the smallest vocabulary whose fertility is
            within this relative margin of the best.
    """

    language: str
    vocab_sizes: list[int] = field(default_factory=lambda: [8000, 16000, 24000, 32000])
    character_coverage: float = 0.9995
    model_type: str = "bpe"
    split_digits: bool = True
    byte_fallback: bool = True
    train_sample_bytes: int = 500_000_000
    eval_docs: int = 5_000
    d_model_for_costing: int = 384
    fertility_tolerance: float = 0.03


def write_training_sample(
    train_jsonl: Path,
    out_txt: Path,
    max_bytes: int,
    seed: int = 17,
) -> tuple[int, int]:
    """Write a plain-text sample of the training split for SentencePiece.

    SentencePiece reads raw text, one sentence or document per line, and holds
    it in memory. Reservoir-style random acceptance is used rather than taking a
    prefix, because shards are grouped by source: a prefix would train the
    tokenizer almost entirely on whichever source happens to sort first.

    Args:
        train_jsonl: Cleaned training split.
        out_txt: Destination text file.
        max_bytes: Stop after writing approximately this many bytes.
        seed: RNG seed for reproducibility.

    Returns:
        ``(documents_written, bytes_written)``.
    """
    rng = random.Random(seed)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    total = train_jsonl.stat().st_size
    # Accept roughly the fraction we need, so the sample spans the whole file
    # instead of stopping partway through the first source.
    keep_prob = min(1.0, (max_bytes * 1.15) / max(total, 1))
    logger.info("Sampling ~%.1f%% of %s for tokenizer training",
                keep_prob * 100, train_jsonl.name)

    docs = written = 0
    with out_txt.open("w", encoding="utf-8") as out:
        for line in train_jsonl.open(encoding="utf-8"):
            if written >= max_bytes:
                break
            if keep_prob < 1.0 and rng.random() > keep_prob:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            # One line per paragraph keeps SentencePiece's sentence length sane.
            for para in text.split("\n"):
                para = para.strip()
                if para:
                    out.write(para + "\n")
                    written += len(para.encode("utf-8")) + 1
            docs += 1
    logger.info("Wrote %s docs / %s bytes to %s",
                f"{docs:,}", f"{written:,}", out_txt.name)
    return docs, written


def load_eval_texts(jsonl: Path, n: int, seed: int = 23) -> list[str]:
    """Sample held-out documents for tokenizer evaluation.

    Args:
        jsonl: Validation or test split.
        n: How many documents to sample.
        seed: RNG seed.

    Returns:
        A list of document texts.
    """
    rng = random.Random(seed)
    picked: list[str] = []
    seen = 0
    for line in jsonl.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
        except json.JSONDecodeError:
            continue
        if not text:
            continue
        seen += 1
        if len(picked) < n:
            picked.append(text)
        else:
            j = rng.randrange(seen)
            if j < n:
                picked[j] = text
    return picked


def train_one(
    sample_txt: Path,
    out_prefix: Path,
    vocab_size: int,
    config: TokenizerConfig,
) -> Path:
    """Train a single SentencePiece model.

    Args:
        sample_txt: Plain-text training sample.
        out_prefix: Output prefix; ``.model`` and ``.vocab`` are written.
        vocab_size: Vocabulary size to train.
        config: Sweep configuration.

    Returns:
        Path to the trained ``.model`` file.
    """
    import sentencepiece as spm

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] training vocab_size=%s ...", config.language, f"{vocab_size:,}")
    t0 = time.time()
    _train_with_fallback(spm, sample_txt, out_prefix, vocab_size, config)
    logger.info("[%s] vocab_size=%s trained in %.0fs",
                config.language, f"{vocab_size:,}", time.time() - t0)
    return out_prefix.with_suffix(".model")


def _train_with_fallback(spm, sample_txt, out_prefix, vocab_size, config) -> None:
    """Train, retrying at a smaller vocabulary if the corpus cannot support it.

    SentencePiece refuses to build a vocabulary larger than the distinct
    subword units its input actually contains, and helpfully names the maximum
    in the exception message. A small corpus plus an ambitious vocabulary
    target hits this -- a live risk for Nepali, whose corpus ended up roughly
    six times smaller than Telugu's.

    Rather than dropping the candidate entirely, retry at the largest size the
    corpus supports. That keeps the sweep informative: "32k was requested but
    only 21,318 units exist" is itself a finding about corpus size, and one
    worth reporting.
    """
    import re as _re
    try:
        _do_train(spm, sample_txt, out_prefix, vocab_size, config)
        return
    except RuntimeError as exc:
        m = _re.search(r"set it to a value <= (\d+)", str(exc))
        if not m:
            raise
        capped = int(m.group(1))
        logger.warning(
            "[%s] corpus supports at most %s subword units, not %s -- "
            "retraining at the cap. This is a signal about corpus size.",
            config.language, f"{capped:,}", f"{vocab_size:,}")
        _do_train(spm, sample_txt, out_prefix, capped, config)


def _do_train(spm, sample_txt, out_prefix, vocab_size, config) -> None:
    """Invoke the SentencePiece trainer once."""
    spm.SentencePieceTrainer.train(
        input=str(sample_txt),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        split_digits=config.split_digits,
        byte_fallback=config.byte_fallback,
        # Fixed ids so every model in the sweep is interchangeable and Phase 2
        # can hardcode them.
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<s>", eos_piece="</s>",
        normalization_rule_name="identity",  # we already applied NFC ourselves
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=vocab_size >= 24000,
        num_threads=8,
        minloglevel=2,          # silence SentencePiece's per-symbol chatter
    )


def evaluate(model_path: Path, texts: list[str]) -> dict:
    """Measure fertility, compression, and unknown rate on held-out text.

    Args:
        model_path: Trained ``.model`` file.
        texts: Held-out documents.

    Returns:
        Dictionary of evaluation metrics.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    n_tokens = n_words = n_chars = n_bytes = n_unk = 0
    for text in texts:
        ids = sp.encode(text, out_type=int)
        n_tokens += len(ids)
        n_unk += sum(1 for i in ids if i == unk_id)
        n_words += len(text.split())
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))

    return {
        "vocab_size": sp.get_piece_size(),
        "tokens": n_tokens,
        "words": n_words,
        "chars": n_chars,
        "bytes": n_bytes,
        # Fertility: tokens per whitespace-delimited word. Lower is better.
        "fertility": round(n_tokens / max(n_words, 1), 4),
        "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
        "unk_tokens": n_unk,
        "unk_rate": round(n_unk / max(n_tokens, 1), 8),
    }


def select_vocab(results: list[dict], config: TokenizerConfig) -> int:
    """Choose a vocabulary size from the sweep.

    Picks the SMALLEST vocabulary whose fertility is within
    ``fertility_tolerance`` of the best observed. Fertility improves
    monotonically with vocabulary size but with sharply diminishing returns,
    while the embedding cost grows linearly and is taken straight out of the
    ~25M parameter budget. Buying a 1% fertility gain with 4M parameters that
    could otherwise be Transformer depth is a bad trade.

    Args:
        results: Evaluation dictionaries from the sweep.
        config: Sweep configuration.

    Returns:
        The chosen vocabulary size.
    """
    best = min(r["fertility"] for r in results)
    threshold = best * (1 + config.fertility_tolerance)
    eligible = [r for r in results if r["fertility"] <= threshold]
    chosen = min(eligible, key=lambda r: r["vocab_size"])
    logger.info("[%s] best fertility %.4f; accepting <= %.4f; chose vocab %s",
                config.language, best, threshold, f"{chosen['vocab_size']:,}")
    return chosen["vocab_size"]


def run_sweep(
    clean_dir: Path,
    out_dir: Path,
    config: TokenizerConfig,
) -> dict:
    """Train every candidate vocabulary, evaluate, and select one.

    Args:
        clean_dir: Directory holding ``train.jsonl`` and ``validation.jsonl``.
        out_dir: Where tokenizer models and the sweep report are written.
        config: Sweep configuration.

    Returns:
        A report dictionary, also written to ``tokenizer_sweep.json``.
    """
    clean_dir, out_dir = Path(clean_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = clean_dir / "train.jsonl"
    val_jsonl = clean_dir / "validation.jsonl"
    for p in (train_jsonl, val_jsonl):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run the cleaning stage first."
            )

    sample_txt = out_dir / "sp_train_sample.txt"
    docs, sample_bytes = write_training_sample(
        train_jsonl, sample_txt, config.train_sample_bytes
    )

    eval_texts = load_eval_texts(val_jsonl, config.eval_docs)
    logger.info("[%s] evaluating on %s held-out documents",
                config.language, f"{len(eval_texts):,}")

    results = []
    for vocab_size in config.vocab_sizes:
        prefix = out_dir / f"{config.language}_sp_{vocab_size}"
        try:
            model = train_one(sample_txt, prefix, vocab_size, config)
        except Exception as exc:
            # A vocabulary larger than the corpus can support will fail; that is
            # informative, not fatal.
            logger.error("[%s] vocab_size=%s failed: %s: %s",
                         config.language, vocab_size, type(exc).__name__, exc)
            continue
        metrics = evaluate(model, eval_texts)
        metrics["embedding_params"] = metrics["vocab_size"] * config.d_model_for_costing
        metrics["model_file"] = model.name
        results.append(metrics)
        logger.info(
            "[%s] vocab %6s | fertility %.3f | chars/tok %.2f | "
            "unk %.2e | emb params %.1fM",
            config.language, f"{metrics['vocab_size']:,}", metrics["fertility"],
            metrics["chars_per_token"], metrics["unk_rate"],
            metrics["embedding_params"] / 1e6,
        )

    if not results:
        raise RuntimeError(f"[{config.language}] every vocabulary size failed")

    chosen = select_vocab(results, config)
    chosen_prefix = out_dir / f"{config.language}_sp_{chosen}"

    report = {
        "language": config.language,
        "training_sample": {
            "documents": docs, "bytes": sample_bytes,
            "source": str(train_jsonl),
        },
        "eval_documents": len(eval_texts),
        "settings": {
            "model_type": config.model_type,
            "character_coverage": config.character_coverage,
            "split_digits": config.split_digits,
            "byte_fallback": config.byte_fallback,
            "d_model_for_costing": config.d_model_for_costing,
            "fertility_tolerance": config.fertility_tolerance,
        },
        "sweep": results,
        "chosen_vocab_size": chosen,
        "chosen_model": f"{chosen_prefix.name}.model",
        "chosen_vocab": f"{chosen_prefix.name}.vocab",
    }
    (out_dir / "tokenizer_sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[%s] wrote %s", config.language,
                out_dir / "tokenizer_sweep.json")
    return report

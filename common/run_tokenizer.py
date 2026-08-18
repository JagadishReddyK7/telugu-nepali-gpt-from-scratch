"""
Shared entry point for tokenizer training, plus the tokenization examples and
plots the Phase 1 report requires.

    python -m telugu.tokenizer.train_tokenizer
    python -m nepali.tokenizer.train_tokenizer --quick   # 8k/16k only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from common.run_crawl import setup_logging
from common.tokenizer_train import TokenizerConfig, run_sweep

logger = logging.getLogger(__name__)


def plot_sweep(report: dict, out_png: Path) -> None:
    """Plot fertility and embedding cost against vocabulary size.

    Every plot the project grades must carry a title, axis labels and a legend;
    a figure missing them scores zero, so those are set explicitly here rather
    than left to matplotlib defaults.

    Args:
        report: Sweep report from :func:`run_sweep`.
        out_png: Destination image path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = report["sweep"]
    vocab = [r["vocab_size"] for r in sweep]
    fert = [r["fertility"] for r in sweep]
    cpt = [r["chars_per_token"] for r in sweep]
    emb = [r["embedding_params"] / 1e6 for r in sweep]
    chosen = report["chosen_vocab_size"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(vocab, fert, "o-", label="Fertility (tokens/word)")
    ax1.plot(vocab, cpt, "s--", label="Characters per token")
    ax1.axvline(chosen, color="crimson", linestyle=":",
                label=f"Chosen vocab = {chosen:,}")
    ax1.set_title(f"{report['language'].title()}: tokenizer efficiency "
                  f"vs vocabulary size")
    ax1.set_xlabel("Vocabulary size (subword units)")
    ax1.set_ylabel("Tokens per word  /  characters per token")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(vocab, emb, "^-", color="darkorange",
             label="Embedding parameters (tied input/output)")
    ax2.axvline(chosen, color="crimson", linestyle=":",
                label=f"Chosen vocab = {chosen:,}")
    ax2.axhline(25, color="grey", linestyle="--",
                label="Total model budget (25M params)")
    ax2.set_title(f"{report['language'].title()}: parameter cost "
                  f"of the vocabulary")
    ax2.set_xlabel("Vocabulary size (subword units)")
    ax2.set_ylabel("Parameters (millions)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_png)


def tokenization_examples(model_path: Path, clean_dir: Path, n: int = 5) -> list[dict]:
    """Produce worked tokenization examples for the report.

    Args:
        model_path: Chosen ``.model`` file.
        clean_dir: Directory containing ``test.jsonl``.
        n: How many examples to produce.

    Returns:
        A list of ``{text, pieces, n_tokens, n_words, fertility}`` dictionaries.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    examples = []
    test = clean_dir / "test.jsonl"
    if not test.exists():
        return examples
    for line in test.open(encoding="utf-8"):
        if len(examples) >= n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
        except json.JSONDecodeError:
            continue
        # First sentence-ish fragment, so the example fits in a report table.
        snippet = text.strip().split("\n")[0][:180]
        if len(snippet) < 40:
            continue
        pieces = sp.encode(snippet, out_type=str)
        examples.append({
            "text": snippet,
            "pieces": pieces,
            "n_tokens": len(pieces),
            "n_words": len(snippet.split()),
            "fertility": round(len(pieces) / max(len(snippet.split()), 1), 3),
        })
    return examples


def run(language: str, config_path: Path, data_root: Path,
        tokenizer_dir: Path) -> None:
    """Train, evaluate, select, plot, and report one language's tokenizer.

    Args:
        language: Language name.
        config_path: Path to the language YAML.
        data_root: Directory containing ``clean/``.
        tokenizer_dir: Repository directory for committed tokenizer artifacts.
    """
    p = argparse.ArgumentParser(description=f"Train the {language} tokenizer.")
    p.add_argument("--quick", action="store_true",
                   help="Sweep only 8k and 16k, with a smaller sample.")
    p.add_argument("--sample-bytes", type=int, default=None,
                   help="Override the tokenizer training sample size in bytes.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tk = raw.get("tokenizer", {})

    vocab_sizes = tk.get("vocab_size_sweep", [8000, 16000, 24000, 32000])
    sample_bytes = args.sample_bytes or tk.get("train_sample_bytes", 500_000_000)
    if args.quick:
        vocab_sizes = [v for v in vocab_sizes if v <= 16000] or [8000]
        sample_bytes = min(sample_bytes, 50_000_000)

    config = TokenizerConfig(
        language=language,
        vocab_sizes=vocab_sizes,
        character_coverage=tk.get("character_coverage", 0.9995),
        model_type=tk.get("algorithm", "bpe"),
        split_digits=tk.get("split_digits", True),
        byte_fallback=tk.get("byte_fallback", True),
        train_sample_bytes=sample_bytes,
    )

    clean_dir = Path(data_root) / "clean"
    out_dir = Path(data_root) / "tokenizer"
    report = run_sweep(clean_dir, out_dir, config)

    chosen_model = out_dir / report["chosen_model"]
    report["examples"] = tokenization_examples(chosen_model, clean_dir)

    # Copy the chosen model and vocabulary into the repository. They are small
    # and are explicit Phase 1 deliverables, so unlike the corpora they belong
    # in git rather than on Drive.
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".model", ".vocab"):
        src = chosen_model.with_suffix(suffix)
        if src.exists():
            (tokenizer_dir / src.name).write_bytes(src.read_bytes())
            logger.info("copied %s -> %s", src.name, tokenizer_dir)

    plot_sweep(report, Path("report/phase1/figures")
               / f"{language}_tokenizer_sweep.png")
    (Path("report/phase1") / f"{language}_tokenizer_sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  {language.upper()} TOKENIZER SWEEP")
    print("=" * 74)
    print(f"  {'vocab':>8s}{'fertility':>12s}{'chars/tok':>12s}"
          f"{'unk rate':>12s}{'emb params':>13s}")
    print("  " + "-" * 55)
    for r in report["sweep"]:
        mark = "  <-- chosen" if r["vocab_size"] == report["chosen_vocab_size"] else ""
        print(f"  {r['vocab_size']:>8,}{r['fertility']:>12.4f}"
              f"{r['chars_per_token']:>12.3f}{r['unk_rate']:>12.2e}"
              f"{r['embedding_params']/1e6:>11.2f}M{mark}")
    print(f"\n  chosen vocabulary: {report['chosen_vocab_size']:,}")
    print(f"  model: {report['chosen_model']}")
    if report["examples"]:
        ex = report["examples"][0]
        print(f"\n  example ({ex['n_words']} words -> {ex['n_tokens']} tokens, "
              f"fertility {ex['fertility']}):")
        print(f"    {ex['text'][:100]}")
        print(f"    {' '.join(ex['pieces'][:20])} ...")
    print()

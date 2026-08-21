"""
Override the tokenizer sweep's vocabulary choice and re-point the artifacts.

    python scripts/reselect.py --lang telugu --vocab 16000
    python scripts/reselect.py --lang nepali --vocab 16000 --yes

WHY THIS SCRIPT EXISTS
----------------------
The sweep in ``common/tokenizer_train.py`` selects on fertility and unknown-token
rate, as the spec's Section 1.3 suggests. Applied literally, that criterion can
only ever pick the LARGEST vocabulary offered, for two reasons:

* Fertility (tokens per word) falls monotonically with vocabulary size. There
  is no interior optimum for a tolerance-based rule to find.
* Unknown-token rate is identically 0.00 at every size, because the tokenizers
  are trained with ``byte_fallback: true``. A byte-fallback BPE model cannot
  emit UNK. So the second half of the selection criterion carries no
  information at all and cannot break the tie.

The sweep therefore chose 32,000 for Telugu. Measured on the held-out set:

    vocab    fertility   chars/tok   unk        emb params
     8,000     2.2855      3.470     0.00e+00      3.07M
    16,000     2.0347      3.898     0.00e+00      6.14M
    24,000     1.9194      4.132     0.00e+00      9.22M
    32,000     1.8490      4.289     0.00e+00     12.29M   <- sweep's pick

THE ARGUMENT FOR OVERRIDING
---------------------------
Phase 2 fixes the model at ~25M parameters, and the input embedding is tied to
the output head, so the embedding matrix is paid for exactly once but is scored
against that same budget. At d_model=384:

    vocab 32,000 -> 12.29M embedding params -> ~49% of the entire model
    vocab 16,000 ->  6.14M embedding params -> ~25% of the entire model

Spending half of a small model's capacity on a lookup table leaves ~12.7M
parameters for the layers that do the actual modelling — roughly 7 transformer
blocks — against ~18.9M, roughly 11 blocks, at 16,000. The price is a 10%
fertility increase (2.035 vs 1.849), which also means 16,000 yields MORE
training tokens from the same text, not fewer.

This is still "choosing using fertility", as the spec asks. It is choosing on
fertility *per parameter spent* rather than on fertility alone, and the
reasoning is recorded in ``tokenizer_sweep.json`` so the decision is auditable
rather than silent.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not re-encode the corpus. Run the encoder afterwards -- the .bin files
and token_counts.json are tokenizer-specific and MUST be regenerated, or Phase 2
will train on token ids that mean nothing in the selected vocabulary:

    python scripts/build_phase1_report.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SELECTION_NOTE = (
    "Vocabulary was set to {vocab:,} by scripts/reselect.py, overriding the "
    "sweep's selection of {swept:,}. The sweep selects on fertility and "
    "unknown-token rate; with byte_fallback enabled the UNK rate is identically "
    "0.00 at every vocabulary size and carries no information, and fertility "
    "falls monotonically with vocabulary size, so that criterion applied "
    "literally always returns the largest candidate. The override selects on "
    "fertility per parameter spent: at d_model=384 a {swept:,} vocabulary costs "
    "{swept_emb:.2f}M tied embedding parameters against a ~25M total budget for "
    "Phase 2, versus {vocab_emb:.2f}M at {vocab:,}. The fertility cost of the "
    "override is ~10%, and the smaller vocabulary yields MORE training tokens "
    "from the same corpus. See report/phase1 for the full argument."
)


def find_model(tok_dir: Path, lang: str, vocab: int) -> tuple[Path, Path]:
    """Locate the trained SentencePiece artifacts for one vocabulary size.

    Args:
        tok_dir: Directory the sweep wrote its models into.
        lang: Language name.
        vocab: Vocabulary size to select.

    Returns:
        Paths to the ``.model`` and ``.vocab`` files.

    Raises:
        SystemExit: If either artifact is missing.
    """
    model = tok_dir / f"{lang}_sp_{vocab}.model"
    vocab_file = tok_dir / f"{lang}_sp_{vocab}.vocab"
    missing = [p for p in (model, vocab_file) if not p.exists()]
    if missing:
        available = sorted(p.name for p in tok_dir.glob(f"{lang}_sp_*.model"))
        raise SystemExit(
            f"Missing {', '.join(p.name for p in missing)} in {tok_dir}.\n"
            f"Available models: {available or 'none'}\n"
            f"The sweep must be run before reselecting."
        )
    return model, vocab_file


def main() -> None:
    """Re-point the chosen tokenizer to a different swept vocabulary."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", required=True, choices=["telugu", "nepali"])
    p.add_argument("--vocab", type=int, default=16000,
                   help="Vocabulary size to select. Must have been swept.")
    p.add_argument("--d-model", type=int, default=384,
                   help="Phase 2 model width, used to state the embedding cost.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt.")
    args = p.parse_args()

    lang = args.lang
    tok_dir = Path(f"data/{lang}/tokenizer")
    repo_tok_dir = Path(f"{lang}/tokenizer")
    sweep_path = tok_dir / "tokenizer_sweep.json"

    if not sweep_path.exists():
        raise SystemExit(f"No sweep report at {sweep_path}. Train the tokenizer first.")

    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    swept_choice = sweep.get("chosen_vocab_size")

    model, vocab_file = find_model(tok_dir, lang, args.vocab)

    swept_emb = (swept_choice or 0) * args.d_model / 1e6
    vocab_emb = args.vocab * args.d_model / 1e6

    print(f"\n  {lang.upper()} tokenizer reselection")
    print(f"    sweep chose      : {swept_choice:,}"
          f"  ({swept_emb:.2f}M embedding params at d_model={args.d_model})")
    print(f"    overriding to    : {args.vocab:,}"
          f"  ({vocab_emb:.2f}M embedding params)")
    print(f"    model            : {model.name}")

    if swept_choice == args.vocab:
        print("\n  Sweep already selected this vocabulary; refreshing artifacts anyway.")

    if not args.yes:
        reply = input("\n  Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            raise SystemExit("Aborted.")

    repo_tok_dir.mkdir(parents=True, exist_ok=True)
    for src in (model, vocab_file):
        shutil.copy2(src, repo_tok_dir / src.name)
        print(f"    copied {src.name} -> {repo_tok_dir}")

    # Remove the superseded artifacts so the committed directory cannot contain
    # two candidate models with no indication of which one Phase 2 loads.
    for stale in repo_tok_dir.glob(f"{lang}_sp_*"):
        if not stale.name.startswith(f"{lang}_sp_{args.vocab}."):
            stale.unlink()
            print(f"    removed superseded {stale.name}")

    sweep["sweep_selected_vocab_size"] = swept_choice
    sweep["chosen_vocab_size"] = args.vocab
    sweep["chosen_model"] = model.name
    sweep["chosen_vocab_file"] = vocab_file.name
    sweep["selection_note"] = SELECTION_NOTE.format(
        vocab=args.vocab, swept=swept_choice or 0,
        swept_emb=swept_emb, vocab_emb=vocab_emb,
    )
    sweep_path.write_text(json.dumps(sweep, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"    updated {sweep_path}")

    print("\n  NEXT: re-encode. The .bin files and token_counts.json are")
    print("  tokenizer-specific and are now stale:")
    print("      python scripts/build_phase1_report.py\n")


if __name__ == "__main__":
    main()

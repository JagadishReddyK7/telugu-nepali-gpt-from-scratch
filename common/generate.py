"""
Generation quality and diversity for a trained model.

    python -m common.generate --config telugu/configs/model.yaml --n 200

Produces, for greedy decoding and temperatures 0.5 / 1.0 / 1.5:
BLEU-4, chrF++, ROUGE-L against held-out reference continuations, plus
repetition rate and Distinct-1/2, plus sample generations for the report.

WHY THESE METRICS ARE WEAK HERE, STATED UP FRONT
------------------------------------------------
The spec asks for BLEU, chrF and ROUGE-L "and briefly explain why each metric is
or is not informative for open-ended LM generation". The honest answer is that
all three are overlap metrics designed for tasks with a *correct* output --
translation, summarisation -- and free-form continuation has no such thing. A
model can produce fluent, entirely reasonable Telugu that shares almost no
4-grams with the one continuation that happened to follow in the corpus, and
score near zero. These numbers are reported because they are asked for and
because they are comparable *between* Model H and Model L under identical
conditions; they should not be read as measuring quality in absolute terms.
Perplexity and bits-per-byte are the meaningful intrinsic measures.

Among the three, **chrF++ is the most trustworthy for these languages** because
it works on character n-grams and needs no notion of a word. That matters
concretely: Telugu and Nepali are agglutinative, so a single morphological
difference changes a whole word token and destroys a word-level n-gram match
while barely moving character overlap.

TOKENIZATION OF THE METRICS THEMSELVES
--------------------------------------
sacreBLEU's default '13a' tokenizer is built around English punctuation and
treats a run of Devanagari or Telugu script as one token, which would make BLEU
meaningless rather than merely weak. Word-level metrics here therefore use
whitespace tokenization, which is the honest minimum for both scripts, and this
choice is recorded in the output so the numbers are interpretable.

ROUGE-L is implemented directly as a longest-common-subsequence F-measure over
whitespace tokens rather than via ``rouge_score``, whose tokenizer strips
non-ASCII alphanumerics -- it would reduce Telugu and Devanagari text to empty
strings and silently report zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from common.data import TokenDataset
from common.evaluate import load_model_from_checkpoint

logger = logging.getLogger(__name__)

DECODINGS = [
    ("greedy", dict(greedy=True)),
    ("t0.5", dict(greedy=False, temperature=0.5)),
    ("t1.0", dict(greedy=False, temperature=1.0)),
    ("t1.5", dict(greedy=False, temperature=1.5)),
]


# ------------------------------------------------------------------- metrics

def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Uses a rolling two-row table: O(len(a) x len(b)) time, O(len(b)) memory.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        Length of the LCS.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 over whitespace tokens.

    Args:
        hypothesis: Generated text.
        reference: Reference text.

    Returns:
        F-measure in [0, 1].
    """
    h, r = hypothesis.split(), reference.split()
    if not h or not r:
        return 0.0
    l = lcs_length(h, r)
    if l == 0:
        return 0.0
    precision, recall = l / len(h), l / len(r)
    return 2 * precision * recall / (precision + recall)


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats of an earlier n-gram.

    Degenerate decoding -- especially greedy -- collapses into loops, and this
    is the cheapest way to see it. A healthy sample sits low; a looping one
    approaches 1.

    Args:
        text: Generated text.
        n: n-gram order.

    Returns:
        Fraction in [0, 1]; 0.0 if the text is too short to contain an n-gram.
    """
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: unique n-grams divided by total n-grams across a corpus.

    Measured over the whole set of generations rather than per sample, so it
    captures the model producing the *same* output for different prompts --
    which per-sample diversity would miss entirely.

    Args:
        texts: Generated texts.
        n: n-gram order (1 or 2).

    Returns:
        Ratio in [0, 1].
    """
    total, unique = 0, set()
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def corpus_metrics(hyps: list[str], refs: list[str]) -> dict:
    """BLEU-4 and chrF++ over a corpus, with an explicit tokenizer choice.

    Args:
        hyps: Generated continuations.
        refs: Reference continuations.

    Returns:
        Dict with the scores and the settings used to produce them.
    """
    out: dict = {}
    try:
        import sacrebleu
        # tokenize="none" with whitespace-split input: see the module docstring
        # for why the default '13a' tokenizer is wrong for these scripts.
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True)
        out["bleu4"] = round(bleu.score, 4)
        out["bleu_signature"] = str(bleu.format(signature=""))[:80]
        out["bleu_tokenizer"] = "none (whitespace); 13a is English-oriented"
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)   # chrF++
        out["chrf_pp"] = round(chrf.score, 4)
    except ImportError:
        out["bleu4"] = None
        out["chrf_pp"] = None
        out["metrics_note"] = "sacrebleu not installed; pip install sacrebleu"
    out["rouge_l"] = round(
        float(np.mean([rouge_l(h, r) for h, r in zip(hyps, refs)])), 4)
    return out


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate_samples(model, dataset: TokenDataset, sp, n_samples: int,
                     prefix_tokens: int, continue_tokens: int, ctx,
                     batch_size: int = 16, seed: int = 1234) -> dict:
    """Generate continuations of held-out prefixes under each decoding setting.

    Prompts are drawn from the held-out split, so the model has never seen the
    prefix or the reference during training.

    Args:
        model: Trained model in eval mode.
        dataset: Held-out dataset.
        sp: SentencePieceProcessor for this language.
        n_samples: Number of prompts.
        prefix_tokens: Tokens of context given to the model.
        continue_tokens: Tokens to generate, and length of the reference.
        ctx: Autocast context.
        batch_size: Prompts per generation call.
        seed: Seed for prompt selection and for sampling.

    Returns:
        Dict keyed by decoding name, each holding hypotheses, references and
        per-setting metrics.
    """
    rng = np.random.default_rng(seed)
    need = prefix_tokens + continue_tokens
    starts = rng.integers(0, len(dataset) - need, size=n_samples)

    prompts = np.stack([dataset.data[s:s + prefix_tokens].astype(np.int64)
                        for s in starts])
    refs_ids = [dataset.data[s + prefix_tokens:s + need].astype(np.int64).tolist()
                for s in starts]
    references = [sp.decode(r) for r in refs_ids]
    prompt_texts = [sp.decode(p.tolist()) for p in prompts]

    results = {}
    for name, kw in DECODINGS:
        torch.manual_seed(seed)          # same seed per setting: differences
                                         # come from the decoding rule, not luck
        hyps: list[str] = []
        for i in range(0, n_samples, batch_size):
            chunk = torch.from_numpy(prompts[i:i + batch_size]).to(dataset.device)
            with ctx:
                out = model.generate(chunk, max_new_tokens=continue_tokens, **kw)
            gen = out[:, prefix_tokens:]
            hyps.extend(sp.decode(row.tolist()) for row in gen.cpu())

        m = corpus_metrics(hyps, references)
        m["repetition_rate_4gram"] = round(
            float(np.mean([repetition_rate(h, 4) for h in hyps])), 4)
        m["distinct_1"] = round(distinct_n(hyps, 1), 4)
        m["distinct_2"] = round(distinct_n(hyps, 2), 4)
        m["mean_chars_generated"] = round(float(np.mean([len(h) for h in hyps])), 1)
        results[name] = {
            "metrics": m,
            "samples": [
                {"prompt": prompt_texts[j][-200:],
                 "generated": hyps[j][:300],
                 "reference": references[j][:300]}
                for j in range(min(3, n_samples))
            ],
        }
        logger.info("  %-7s BLEU %s | chrF++ %s | ROUGE-L %.4f | rep %.3f | "
                    "d1 %.3f d2 %.3f", name, m.get("bleu4"), m.get("chrf_pp"),
                    m["rouge_l"], m["repetition_rate_4gram"],
                    m["distinct_1"], m["distinct_2"])
    return results


def main() -> None:
    """CLI: generate from a checkpoint and score the output."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--n", type=int, default=200, help="Number of prompts.")
    p.add_argument("--prefix-tokens", type=int, default=64)
    p.add_argument("--continue-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = args.checkpoint or Path(config["paths"]["out_dir"]) / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = config["train"].get("dtype", "float16")
    if device_type == "cpu":
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    model, ck = load_model_from_checkpoint(ckpt, device)
    bin_path = config["paths"][f"{args.split}_bin"]
    ds = TokenDataset(bin_path, config["model"]["block_size"], device,
                      seed=config["train"]["seed"])

    logger.info("[%s] generating %d continuations (%d-token prefix -> %d tokens)",
                config.get("language"), args.n, args.prefix_tokens,
                args.continue_tokens)
    res = generate_samples(model, ds, sp, args.n, args.prefix_tokens,
                           args.continue_tokens, ctx, args.batch_size,
                           seed=config["train"]["seed"])

    payload = {
        "language": config.get("language"),
        "checkpoint": str(ckpt),
        "checkpoint_step": ck.get("step"),
        "split": args.split,
        "n_prompts": args.n,
        "prefix_tokens": args.prefix_tokens,
        "continue_tokens": args.continue_tokens,
        "decodings": res,
    }
    out = args.out or Path(f"report/phase2/stats/{config['language']}_generation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()

"""
Reasoning evaluation: exact match by greedy generation, plus a scored-choice
accuracy that gives the *pretrained* model a fair baseline.

    python -m common.reasoning_eval --config telugu/configs/finetune.yaml \\
        --checkpoint checkpoints/telugu/best.pt --name telugu_pretrained

TWO ACCURACIES, AND WHY BOTH ARE REPORTED
-----------------------------------------
**Exact match (generative).** The model is given the prompt up to and including
the answer prefix ("జవాబు:" / "उत्तर:") and decodes greedily until ``</s>``.
The decoded string, after Unicode normalization and stripping a trailing
terminator, must equal the gold answer exactly. This is the number the spec
asks for, and it is the honest one: a model that answers "రాము కంటే" instead
of "రాము" is wrong.

The pretrained model has never seen the answer-prefix convention and will
mostly continue the prompt as prose. Its exact match is therefore expected to
be near zero -- which measures format ignorance, not reasoning ability.

**Scored choice (multiple-choice by log-likelihood).** Every example carries a
small candidate set (the entity names; yes/no; the possible relation
statements; the permutations of an ordering). Each candidate is appended to the
prompt and scored by its total log-probability under the model; the argmax is
the prediction. Chance is 1/|choices|. This isolates *whether the model prefers
the right answer* from *whether it can produce the answer in the right
format*, and is the only way to say anything about the pretrained model beyond
"it did not learn the format."

BATCHING WITHOUT PADDING
------------------------
The model has no padding-aware attention mask, and left-padding would shift the
learned absolute positions. Prompts are therefore grouped into buckets of
identical token length and generated batch-by-bucket -- exact, no padding.
Choice scoring right-pads instead, which is safe under a causal mask: a pad
token after the sequence cannot influence any earlier position.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from common.reasoning_gen import normalize_answer, read_jsonl

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ encoding

def encode_prompt(sp, prompt: str) -> list[int]:
    """``<s>`` + prompt tokens. The model generates from here."""
    return [sp.bos_id()] + sp.encode(prompt)


def encode_example(sp, prompt: str, answer: str) -> tuple[list[int], list[bool]]:
    """Tokenize one training example and mark which tokens are the answer.

    Prompt and answer are encoded separately and concatenated, so the boundary
    is exact and matches what :func:`encode_prompt` produces at test time.
    SentencePiece prepends a word-boundary marker to each ``encode`` call, so
    the join is equivalent to encoding ``prompt + " " + answer``.

    Args:
        sp: SentencePieceProcessor.
        prompt: Premises + question + answer prefix.
        answer: Gold answer.

    Returns:
        ``(ids, is_answer)`` where ``ids`` is ``<s> prompt answer </s>`` and
        ``is_answer[i]`` is True for the answer tokens and the closing ``</s>``.
    """
    p = encode_prompt(sp, prompt)
    a = sp.encode(answer) + [sp.eos_id()]
    return p + a, [False] * len(p) + [True] * len(a)


def make_ctx(device: str, dtype: str):
    """Autocast context for ``device`` (fp32 on CPU regardless of ``dtype``)."""
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cpu":
        return nullcontext()
    pt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type=device_type, dtype=pt)


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate_answers(model, sp, prompts: list[str], device: str, ctx,
                     max_new_tokens: int = 24, batch_size: int = 64) -> list[str]:
    """Greedy-decode an answer for every prompt, bucketed by prompt length.

    Args:
        model: Model in eval mode.
        sp: SentencePieceProcessor.
        prompts: Prompt strings (each ending in the answer prefix).
        device: Device string.
        ctx: Autocast context.
        max_new_tokens: Generation cap; answers here are at most ~15 tokens.
        batch_size: Maximum prompts per generate call.

    Returns:
        Decoded answers, in the same order as ``prompts``, cut at ``</s>``.
    """
    was_training = model.training
    model.eval()
    eos = sp.eos_id()
    enc = [encode_prompt(sp, p) for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out: list[str | None] = [None] * len(enc)

    i = 0
    while i < len(order):
        L = len(enc[order[i]])
        j = i
        while j < len(order) and len(enc[order[j]]) == L and j - i < batch_size:
            j += 1
        idx = torch.tensor([enc[k] for k in order[i:j]], dtype=torch.long,
                           device=device)
        with ctx:
            gen = model.generate(idx, max_new_tokens=max_new_tokens,
                                 greedy=True, eos_id=eos)
        for row, k in zip(gen[:, L:].tolist(), order[i:j]):
            if eos in row:
                row = row[:row.index(eos)]
            out[k] = sp.decode(row)
        i = j
    if was_training:
        model.train()
    return [o if o is not None else "" for o in out]


@torch.no_grad()
def score_choices(model, sp, examples: list[dict], device: str, ctx,
                  max_seqs: int = 96) -> list[int]:
    """Pick, for each example, the candidate with the highest log-likelihood.

    Args:
        model: Model in eval mode.
        sp: SentencePieceProcessor.
        examples: Dicts with ``prompt`` and ``choices``.
        device: Device string.
        ctx: Autocast context.
        max_seqs: Sequences per forward pass.

    Returns:
        Index into ``choices`` of the argmax candidate, per example.
    """
    was_training = model.training
    model.eval()
    eos, pad = sp.eos_id(), sp.pad_id() if sp.pad_id() >= 0 else 0

    # Flatten (example, choice) pairs, then process in chunks.
    seqs: list[tuple[int, int, list[int], int]] = []   # (ex, choice, ids, n_prompt)
    for ei, ex in enumerate(examples):
        p = encode_prompt(sp, ex["prompt"])
        for ci, c in enumerate(ex["choices"]):
            seqs.append((ei, ci, p + sp.encode(c) + [eos], len(p)))

    scores: dict[tuple[int, int], float] = {}
    for s in range(0, len(seqs), max_seqs):
        chunk = seqs[s:s + max_seqs]
        T = max(len(ids) for _, _, ids, _ in chunk) - 1
        x = torch.full((len(chunk), T), pad, dtype=torch.long)
        y = torch.full((len(chunk), T), -1, dtype=torch.long)
        for r, (_, _, ids, n_p) in enumerate(chunk):
            x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
            # Targets only over the candidate tokens (and </s>): position t
            # predicts ids[t+1], so candidate tokens are targets at t >= n_p-1.
            y[r, n_p - 1:len(ids) - 1] = torch.tensor(ids[n_p:])
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _, _ = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        tok = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        tok = tok.masked_fill(y < 0, 0.0).sum(dim=-1)
        for r, (ei, ci, _, _) in enumerate(chunk):
            scores[(ei, ci)] = float(tok[r])

    picks = []
    for ei, ex in enumerate(examples):
        best = max(range(len(ex["choices"])), key=lambda ci: scores[(ei, ci)])
        picks.append(best)
    if was_training:
        model.train()
    return picks


# ----------------------------------------------------------------- scoring

def _acc(flags: list[bool]) -> float:
    """Fraction of True, 0.0 for an empty list."""
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def breakdown(examples: list[dict], flags: list[bool], key: str) -> dict:
    """Accuracy grouped by one example attribute (``task``, ``hops`` ...)."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for ex, ok in zip(examples, flags):
        groups[str(ex[key])].append(ok)
    return {k: {"n": len(v), "accuracy": _acc(v)} for k, v in sorted(groups.items())}


def evaluate_reasoning(model, sp, examples: list[dict], device: str, ctx,
                       max_new_tokens: int = 24, with_choices: bool = True,
                       n_examples_per_task: int = 4) -> dict:
    """Score a checkpoint on a list of reasoning examples.

    Args:
        model: Model (any mode; restored afterwards).
        sp: SentencePieceProcessor.
        examples: Loaded JSONL rows.
        device: Device string.
        ctx: Autocast context.
        max_new_tokens: Generation cap.
        with_choices: Also compute scored-choice accuracy.
        n_examples_per_task: Successes and failures to keep per task.

    Returns:
        JSON-serializable results: overall accuracies, breakdowns, and
        qualitative samples.
    """
    t0 = time.time()
    prompts = [ex["prompt"] for ex in examples]
    preds = generate_answers(model, sp, prompts, device, ctx, max_new_tokens)
    em = [normalize_answer(p) == normalize_answer(ex["answer"])
          for p, ex in zip(preds, examples)]

    res: dict = {
        "n": len(examples),
        "exact_match": _acc(em),
        "exact_match_by_task": breakdown(examples, em, "task"),
        "exact_match_by_hops": breakdown(examples, em, "hops"),
        "exact_match_by_n_entities": breakdown(examples, em, "n_entities"),
        "exact_match_by_attribute": breakdown(examples, em, "attribute"),
        "exact_match_by_template_held_out": breakdown(examples, em, "template_held_out"),
    }
    # "Format" accuracy: did the model at least emit one of the candidates?
    in_choices = [normalize_answer(p) in {normalize_answer(c) for c in ex["choices"]}
                  for p, ex in zip(preds, examples)]
    res["prediction_is_a_valid_choice"] = _acc(in_choices)

    if with_choices:
        picks = score_choices(model, sp, examples, device, ctx)
        ch = [ex["choices"][k] == ex["answer"] for ex, k in zip(examples, picks)]
        chance = [1.0 / len(ex["choices"]) for ex in examples]
        res["choice_accuracy"] = _acc(ch)
        res["choice_chance"] = round(sum(chance) / len(chance), 4)
        res["choice_accuracy_by_task"] = breakdown(examples, ch, "task")
        res["choice_accuracy_by_hops"] = breakdown(examples, ch, "hops")
        res["choice_accuracy_by_template_held_out"] = breakdown(
            examples, ch, "template_held_out")
        res["choice_chance_by_task"] = {
            t: round(sum(c for c, e in zip(chance, examples) if e["task"] == t)
                     / max(1, sum(1 for e in examples if e["task"] == t)), 4)
            for t in sorted({e["task"] for e in examples})}

    # Qualitative samples: a few successes and failures per task.
    samples: dict[str, dict[str, list]] = defaultdict(lambda: {"correct": [], "wrong": []})
    for ex, p, ok in zip(examples, preds, em):
        bucket = samples[ex["task"]]["correct" if ok else "wrong"]
        if len(bucket) < n_examples_per_task:
            bucket.append({"id": ex["id"], "prompt": ex["prompt"],
                           "gold": ex["answer"], "predicted": p,
                           "hops": ex["hops"],
                           "template_held_out": ex["template_held_out"]})
    res["samples"] = dict(samples)
    res["predictions"] = preds
    res["eval_seconds"] = round(time.time() - t0, 1)
    return res


def write_examples_md(res: dict, out_path: Path, title: str) -> None:
    """Render the qualitative successes / failures as Markdown for the report."""
    L = [f"# {title}", "",
         f"Exact match **{res['exact_match']:.4f}** on {res['n']} test examples"
         + (f"; scored-choice accuracy **{res['choice_accuracy']:.4f}** "
            f"(chance {res['choice_chance']:.4f})" if "choice_accuracy" in res else ""),
         ""]
    for task, buckets in res["samples"].items():
        L += [f"## {task}", ""]
        for kind in ("correct", "wrong"):
            L += [f"### {kind}", ""]
            if not buckets[kind]:
                L += ["*(none)*", ""]
            for s in buckets[kind]:
                flag = " *(held-out template)*" if s["template_held_out"] else ""
                L += [f"- **{s['id']}**{flag} · {s['hops']} hops",
                      f"  - prompt: {s['prompt']}",
                      f"  - gold: **{s['gold']}**",
                      f"  - predicted: **{s['predicted']}**", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- cli

def main() -> None:
    """CLI: evaluate one checkpoint on the reasoning test split."""
    from common.evaluate import load_model_from_checkpoint
    import sentencepiece as spm

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="A finetune YAML (for the data paths and tokenizer).")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--name", required=True,
                   help="Output stem, e.g. telugu_pretrained / telugu_finetuned.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stats-dir", type=Path, default=Path("report/phase3/stats"))
    p.add_argument("--no-choices", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model_from_checkpoint(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=config["paths"]["tokenizer"])
    ctx = make_ctx(device, config["train"].get("dtype", "float16"))

    examples = read_jsonl(Path(config["data"][args.split]))
    if args.limit:
        examples = examples[:args.limit]
    logger.info("[%s] %d %s examples, checkpoint step %s", args.name,
                len(examples), args.split, ck.get("step"))
    res = evaluate_reasoning(model, sp, examples, device, ctx,
                             with_choices=not args.no_choices)
    res.update({"name": args.name, "language": config.get("language"),
                "checkpoint": str(args.checkpoint),
                "checkpoint_step": ck.get("step"), "split": args.split})

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    out = args.stats_dir / f"{args.name}_reasoning_{args.split}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_examples_md(res, args.stats_dir / f"{args.name}_reasoning_examples.md",
                      f"Reasoning examples — {args.name}")
    print(f"\n  exact match      {res['exact_match']:.4f}")
    if "choice_accuracy" in res:
        print(f"  choice accuracy  {res['choice_accuracy']:.4f}  "
              f"(chance {res['choice_chance']:.4f})")
    for t, v in res["exact_match_by_task"].items():
        print(f"    {t:20s} {v['accuracy']:.4f}  (n={v['n']})")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

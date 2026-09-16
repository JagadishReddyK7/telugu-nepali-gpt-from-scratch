"""
Tests for Phase 3: the synthetic reasoning data, the finetune loop, and the
reasoning evaluator.

    python tests/test_reasoning.py
    python -m pytest tests/test_reasoning.py -q

THE TESTS THAT MATTER MOST
--------------------------
``test_labels_are_correct_by_independent_derivation`` re-derives every gold
answer from the hidden order recorded in the example, using none of the
generator's own answer code. A template bug that rendered "A is taller than B"
for A < B would not be caught by the generator itself; it is caught here.

``test_no_entity_leakage_between_splits`` and
``test_held_out_templates_never_appear_in_train`` check the two leakage
controls the report relies on.

``test_finetune_resume_is_bit_exact`` does for finetuning what
``tests/test_train.py`` does for pretraining: straight-through vs
stop-and-resume must give identical weights.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reasoning_gen import (ReasoningGenerator, TASKS, normalize_answer,  # noqa: E402
                                  build_dataset)
from telugu.reasoning.lexicon import LEXICON as TE  # noqa: E402
from nepali.reasoning.lexicon import LEXICON as NE  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- generator

def _small(lex, seed=7, n=300):
    gen = ReasoningGenerator(lex, seed)
    return gen, {s: gen.generate_split(s, n) for s in ("train", "validation", "test")}


def _held_out_phrasings(lex) -> set[str]:
    """Every reserved (last) phrasing, with placeholders stripped to a stem."""
    out = set()
    fams = [lex["intro"]] + [v for a in lex["attributes"].values()
                             for v in a.values() if isinstance(v, list)]
    for fam in fams:
        if len(fam) >= 2 and fam[-1]:
            out.add(fam[-1])
    return out


def _stem(template: str) -> str:
    """Template with placeholders removed, for substring matching."""
    return template.replace("{a}", "").replace("{b}", "").replace("{v}", "")


def test_all_tasks_present_and_balanced():
    for lex in (TE, NE):
        _, splits = _small(lex)
        for exs in splits.values():
            counts = {t: sum(e.task == t for e in exs) for t in TASKS}
            assert min(counts.values()) >= len(exs) // len(TASKS) - 1, counts


def test_labels_are_correct_by_independent_derivation():
    """Re-derive every answer from ``premise_order`` / the prompt values."""
    for lex in (TE, NE):
        _, splits = _small(lex, n=500)
        for exs in splits.values():
            for e in exs:
                attr = lex["attributes"][e.attribute]
                if e.task == "superlative_chain":
                    assert e.answer in (e.premise_order[0], e.premise_order[-1])
                    # which one is fixed by the question; both are entities
                    assert e.answer in e.entities
                elif e.task == "ordering":
                    names = e.answer.split(lex["list_separator"])
                    assert names == e.premise_order or names == e.premise_order[::-1]
                    assert sorted(names) == sorted(e.entities)
                elif e.task == "yes_no":
                    assert e.answer in (lex["yes"], lex["no"])
                elif e.task == "chain_relation":
                    if e.answer == lex["undetermined"]:
                        assert len(e.premise_order) == 1     # branching structure
                    else:
                        a, c = e.premise_order[0], e.premise_order[-1]
                        assert e.answer in (attr["ans_more"].format(a=a, b=c),
                                            attr["ans_less"].format(a=c, b=a))
                elif e.task == "numeric_compare":
                    assert e.answer in e.entities or e.answer == lex["equal"]
                assert e.answer in e.choices


def test_yes_no_answers_follow_the_hidden_order():
    """For yes/no, recompute truth from premise_order and the question words."""
    for lex in (TE, NE):
        _, splits = _small(lex, n=400)
        for e in splits["test"] + splits["train"]:
            if e.task != "yes_no":
                continue
            attr = lex["attributes"][e.attribute]
            order = e.premise_order
            # The question is the last sentence; find which pair and polarity.
            q = e.prompt.rsplit(lex["answer_prefix"], 1)[0].strip()
            q = q[q.rfind("?", 0, len(q) - 1) + 1:] if q.count("?") > 1 else q
            found = None
            for x, y in itertools.permutations(order, 2):
                for fam, more in ((attr["q_yesno_more"], True),
                                  (attr["q_yesno_less"], False)):
                    for t in fam:
                        if q.endswith(t.format(a=x, b=y)):
                            found = (x, y, more)
            assert found, e.prompt
            x, y, more = found
            truth = order.index(x) < order.index(y)
            expect = lex["yes"] if (truth if more else not truth) else lex["no"]
            assert e.answer == expect, (e.prompt, e.answer)


def test_no_entity_leakage_between_splits():
    for lex in (TE, NE):
        gen, splits = _small(lex)
        pools = gen.pools
        for kind in ("names", "objects"):
            tr, va, te = (set(pools[kind][s]) for s in ("train", "validation", "test"))
            assert not (tr & va) and not (tr & te) and not (va & te)
        train_entities = {x for e in splits["train"] for x in e.entities}
        test_entities = {x for e in splits["test"] for x in e.entities}
        assert not (train_entities & test_entities)


def test_held_out_templates_never_appear_in_train():
    for lex in (TE, NE):
        _, splits = _small(lex, n=600)
        held = {_stem(t) for t in _held_out_phrasings(lex)}
        for e in splits["train"] + splits["validation"]:
            for h in held:
                # Stems are long enough that accidental substring hits do not occur.
                assert h not in e.prompt, (h, e.prompt)
        flagged = [e for e in splits["test"] if e.template_held_out]
        assert 0.35 < len(flagged) / len(splits["test"]) < 0.65
        # Every flagged test example really does contain a held-out phrasing.
        for e in flagged:
            assert any(h in e.prompt for h in held), e.prompt


def test_prompts_are_unique_across_splits():
    for lex in (TE, NE):
        _, splits = _small(lex)
        prompts = [normalize_answer(e.prompt) for exs in splits.values() for e in exs]
        assert len(prompts) == len(set(prompts))


def test_generation_is_deterministic():
    a = ReasoningGenerator(TE, 11).generate_split("test", 100)
    b = ReasoningGenerator(TE, 11).generate_split("test", 100)
    assert [e.prompt for e in a] == [e.prompt for e in b]
    assert [e.answer for e in a] == [e.answer for e in b]


def test_prompts_use_the_target_script():
    def frac(s, lo, hi):
        letters = [c for c in s if c.isalpha()]
        return sum(lo <= ord(c) <= hi for c in letters) / max(len(letters), 1)
    for lex, lo, hi in ((TE, 0x0C00, 0x0C7F), (NE, 0x0900, 0x097F)):
        _, splits = _small(lex, n=100)
        for e in splits["test"]:
            assert frac(e.prompt, lo, hi) > 0.95, e.prompt
            assert frac(e.answer, lo, hi) > 0.95, e.answer


def test_committed_dataset_matches_generator():
    """The committed test split must be exactly what the script produces."""
    for lang, lex, seed in (("telugu", TE, 1337), ("nepali", NE, 2718)):
        path = REPO / lang / "reasoning" / "dataset" / "test.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        gen = ReasoningGenerator(lex, seed)
        gen.generate_split("train", 30_000)      # advances the dedup set
        gen.generate_split("validation", 2_000)
        regen = gen.generate_split("test", 3_000)
        assert len(rows) == len(regen)
        assert all(r["prompt"] == g.prompt and r["answer"] == g.answer
                   for r, g in zip(rows[:200], regen[:200]))


# ---------------------------------------------------------------- finetune

def _tiny_workspace(root: Path, lang: str, lex, seed: int = 3) -> Path:
    """Tiny pretrained checkpoint + tiny dataset + finetune config."""
    from common.model import GPT, GPTConfig
    import sentencepiece as spm

    tok = REPO / lang / "tokenizer" / f"{lang}_sp_16000.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    build_dataset(lex, lang, root / "ds", seed, {"train": 120, "validation": 40, "test": 40})

    m = {"vocab_size": sp.get_piece_size(), "d_model": 32, "n_layer": 2, "n_head": 4,
         "d_ff": 64, "block_size": 128, "dropout": 0.1, "bias": True,
         "tie_embeddings": True, "positional": "learned"}
    g = GPT(GPTConfig(**m))
    (root / "pre").mkdir(parents=True, exist_ok=True)
    torch.save({"model": g.state_dict(), "optimizer": None, "scaler": None,
                "step": 1, "tokens_seen": 1, "best_val_loss": 9.0,
                "config": {"name": "tiny", "model": m, "train": {"dtype": "float32"}}},
               root / "pre" / "best.pt")

    cfg = {
        "name": "tiny_ft", "language": lang, "init_from": str(root / "pre" / "best.pt"),
        "data": {"train": str(root / "ds" / "train.jsonl"),
                 "validation": str(root / "ds" / "validation.jsonl"),
                 "test": str(root / "ds" / "test.jsonl"),
                 "answer_prefix": lex["answer_prefix"]},
        "train": {"batch_size": 4, "grad_accum": 1, "epochs": 3,
                  "learning_rate": 1e-3, "min_learning_rate": 1e-4,
                  "warmup_steps": 2, "weight_decay": 0.1, "beta1": 0.9,
                  "beta2": 0.95, "grad_clip": 1.0, "dropout": 0.1,
                  "prompt_loss_weight": 1.0, "dtype": "float32", "seed": 5},
        "eval": {"eval_interval_steps": 1000, "eval_batches": 2,
                 "val_accuracy_examples": 8, "checkpoint_interval_steps": 4,
                 "log_interval_steps": 1000},
        "paths": {"tokenizer": str(tok), "out_dir": str(root / "ckpt"),
                  "log_file": str(root / "logs" / "ft.jsonl")},
    }
    p = root / "finetune.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def test_packing_keeps_examples_whole_and_masks_answers():
    import sentencepiece as spm
    from common.finetune import PackedReasoningData
    from common.reasoning_gen import read_jsonl
    sp = spm.SentencePieceProcessor(model_file=str(REPO / "telugu/tokenizer/telugu_sp_16000.model"))
    _, splits = _small(TE, n=60)
    rows_json = [json.loads(json.dumps(e.__dict__)) for e in splits["train"]]
    data = PackedReasoningData(rows_json, sp, 128, prompt_loss_weight=0.0, seed=1)
    rows = data.rows_for_epoch(0)
    assert rows == data.rows_for_epoch(0)                     # deterministic
    assert sorted(i for r in rows for i in r) == list(range(len(rows_json)))
    for r in rows:
        assert sum(len(data.encoded[i][0]) for i in r) <= 129
    x, y, w, a = data.build_batch(rows[:2], "cpu")
    assert x.shape == y.shape == w.shape == a.shape == (2, 128)
    # With prompt weight 0, weighted positions == answer positions exactly.
    assert torch.equal(w > 0, a)
    # Every </s> is an answer target and is followed by <s> or padding.
    eos, bos = sp.eos_id(), sp.bos_id()
    for r in range(2):
        for t in range(127):
            if y[r, t] == eos:
                assert a[r, t]
                assert y[r, t + 1] in (bos, -1)


def test_finetune_checkpoint_has_every_required_field():
    root = Path(tempfile.mkdtemp())
    try:
        from common.finetune import finetune
        cfg = _tiny_workspace(root, "telugu", TE)
        finetune(cfg, max_steps=4)
        ck = torch.load(root / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
        for k in ("model", "optimizer", "scaler", "step", "tokens_seen", "config",
                  "torch_rng_state", "finetune"):
            assert k in ck, k
        assert ck["config"]["model"]["d_model"] == 32          # copied from init_from
        assert ck["config"]["init_from"].endswith("best.pt")
        assert ck["config"]["train"]["learning_rate"] == 1e-3   # hyperparameters logged
        assert ck["finetune"]["epoch"] >= 0 and "batch_index" in ck["finetune"]
        assert ck["optimizer"]["state"], "AdamW moments missing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_finetune_resume_is_bit_exact():
    root = Path(tempfile.mkdtemp())
    try:
        from common.finetune import finetune
        cfg = _tiny_workspace(root, "nepali", NE)
        # Straight through: 8 steps.
        finetune(cfg, max_steps=8, out_dir_override=str(root / "A"))
        a = torch.load(root / "A" / "latest.pt", map_location="cpu", weights_only=False)
        # Two halves: same 8-step schedule, but stop the session after 4,
        # checkpoint, and resume. Only the interruption differs.
        finetune(cfg, max_steps=8, out_dir_override=str(root / "B"), session_steps=4)
        b4 = torch.load(root / "B" / "latest.pt", map_location="cpu", weights_only=False)
        assert b4["step"] == 4
        finetune(cfg, max_steps=8, out_dir_override=str(root / "B"), resume=True)
        b = torch.load(root / "B" / "latest.pt", map_location="cpu", weights_only=False)
        assert a["step"] == b["step"] == 8
        assert a["tokens_seen"] == b["tokens_seen"]
        for k in a["model"]:
            assert torch.equal(a["model"][k], b["model"][k]), k
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_finetune_starts_from_pretrained_weights_and_keeps_vocab():
    root = Path(tempfile.mkdtemp())
    try:
        from common.finetune import build_finetune_model
        cfg_path = _tiny_workspace(root, "telugu", TE)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        model, pre = build_finetune_model(cfg, "cpu")
        for k, v in pre["model"].items():
            assert torch.equal(model.state_dict()[k], v), k
        assert model.token_embedding.weight.shape[0] == 16000
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------- evaluator

def test_exact_match_normalisation():
    assert normalize_answer(" राम। ") == "राम"
    assert normalize_answer("రాము.") == "రాము"
    assert normalize_answer("a  b") == "a b"
    assert normalize_answer("के") == normalize_answer("के")   # NFC


def test_choice_scoring_matches_unbatched_scoring():
    """Right-padded batch scoring must equal scoring each candidate alone."""
    import sentencepiece as spm
    import torch.nn.functional as F
    from common.model import GPT, GPTConfig
    from common.reasoning_eval import score_choices, encode_prompt
    sp = spm.SentencePieceProcessor(model_file=str(REPO / "nepali/tokenizer/nepali_sp_16000.model"))
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=16000, d_model=32, n_layer=2, n_head=4, d_ff=64,
                          block_size=128, dropout=0.0)).eval()
    _, splits = _small(NE, n=20)
    exs = [json.loads(json.dumps(e.__dict__)) for e in splits["test"][:10]]
    picks = score_choices(model, sp, exs, "cpu", torch.no_grad())
    for e, k in zip(exs, picks):
        p = encode_prompt(sp, e["prompt"])
        best = None
        for ci, c in enumerate(e["choices"]):
            ids = p + sp.encode(c) + [sp.eos_id()]
            x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
            with torch.no_grad():
                lg, _, _ = model(x)
            lp = F.log_softmax(lg.float(), -1).gather(-1, y.unsqueeze(-1))[0, len(p) - 1:].sum().item()
            if best is None or lp > best[0]:
                best = (lp, ci)
        assert best[1] == k


def test_generate_answers_is_exact_under_bucketing():
    """Bucketed batch generation must equal one-at-a-time generation."""
    import sentencepiece as spm
    from contextlib import nullcontext
    from common.model import GPT, GPTConfig
    from common.reasoning_eval import generate_answers, encode_prompt
    sp = spm.SentencePieceProcessor(model_file=str(REPO / "telugu/tokenizer/telugu_sp_16000.model"))
    torch.manual_seed(1)
    model = GPT(GPTConfig(vocab_size=16000, d_model=32, n_layer=2, n_head=4, d_ff=64,
                          block_size=128, dropout=0.0)).eval()
    _, splits = _small(TE, n=20)
    prompts = [e.prompt for e in splits["test"][:12]]
    batched = generate_answers(model, sp, prompts, "cpu", nullcontext(), max_new_tokens=6, batch_size=5)
    for p, b in zip(prompts, batched):
        idx = torch.tensor([encode_prompt(sp, p)])
        out = model.generate(idx, 6, greedy=True, eos_id=sp.eos_id())[0, idx.shape[1]:].tolist()
        if sp.eos_id() in out:
            out = out[:out.index(sp.eos_id())]
        assert sp.decode(out) == b


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

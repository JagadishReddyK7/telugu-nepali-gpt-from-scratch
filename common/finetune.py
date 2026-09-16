"""
Reasoning finetuning for one model, resumable, starting from its own
pretrained checkpoint (Phase 3.1).

    python -m common.finetune --config telugu/configs/finetune.yaml
    python -m common.finetune --config telugu/configs/finetune.yaml --resume

PROTOCOL (spec 3.1)
-------------------
* ``init_from`` is that language's OWN pretrained ``best.pt``. Model H is never
  initialised from Model L or vice versa; the config is per language.
* The tokenizer and vocabulary are fixed: the embedding matrix keeps its shape
  and the SentencePiece model is the Phase 1 file. No token is added -- the
  answer prefix is ordinary text.
* Every hyperparameter is in the YAML and is written into every checkpoint.
* Checkpoints use exactly the pretraining format (``common/train.py``):
  weights, AdamW moments, GradScaler, step, tokens, config, and NumPy / CPU /
  CUDA RNG state, written atomically. Two extra keys record the finetuning
  position (epoch, batch index) so a resume continues at the right example.

HOW EXAMPLES BECOME BATCHES
---------------------------
Examples are short (20-60 prompt tokens, 1-15 answer tokens). Padding each one
to the 512-token context would waste >90% of every batch, so examples are
*packed*: each epoch shuffles the examples with a seed derived from the epoch
number, then fills rows of ``block_size + 1`` tokens greedily with whole
examples, ``<s> prompt answer </s>`` back to back. Leftover positions carry
target ``-1`` and are ignored. Rows never split an example, and the row set of
epoch ``e`` is a pure function of ``(seed, e)`` -- which is what makes resume
exact: the loader is rebuilt for the saved epoch and fast-forwarded to the
saved batch index.

This is the same shape the model saw in pretraining, where documents were
concatenated with ``<s>``/``</s>`` markers and windows crossed boundaries. The
Phase 2 attention analysis showed both models learned to gate attention at
those markers, so an example can be expected not to read its neighbour.

THE LOSS
--------
Per-token cross-entropy, weighted: answer tokens (and the closing ``</s>``)
weigh 1.0, prompt tokens weigh ``prompt_loss_weight``. At 1.0 this is plain
language modelling over the packed stream; at 0.0 it is answer-only
supervision.

Run 1 used 1.0 and failed in an instructive way: validation exact match
plateaued at ~0.15 within 50 steps, and on the held-out-name test split the
model answered name questions with *training* names (superlative accuracy
0.5%, far below the 33% of guessing a name from the prompt). Full-sequence
loss asks the model to predict the first mention of every random name inside
the premises, which is possible only through a prior over names -- so it
trains precisely the behaviour that is wrong at the answer position, where
the name must be *copied* from context. Run 2 sets the weight to 0.0 and
enlarges the name pool ~6x with compound names, so memorising the answer
distribution stops being the cheapest solution.

The **learning rate is a pure function of the step** (warmup then cosine),
exactly as in pretraining, so no scheduler object needs to be serialized.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common.evaluate import load_model_from_checkpoint
from common.model import GPT, GPTConfig
from common.reasoning_eval import (encode_example, evaluate_reasoning,
                                   make_ctx)
from common.reasoning_gen import read_jsonl

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ schedule

def lr_at_step(step: int, total_steps: int, cfg: dict) -> float:
    """Linear warmup over ``warmup_steps`` then cosine decay to the minimum.

    Args:
        step: Optimizer steps completed so far.
        total_steps: Planned length of the run.
        cfg: The ``train`` block of the finetune config.

    Returns:
        Learning rate for the next step.
    """
    warmup = cfg["warmup_steps"]
    lr, lr_min = cfg["learning_rate"], cfg["min_learning_rate"]
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= total_steps:
        return lr_min
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------- data

class PackedReasoningData:
    """Tokenizes reasoning examples and packs them into fixed-length rows.

    Args:
        examples: Loaded JSONL rows with ``prompt`` and ``answer``.
        sp: SentencePieceProcessor (the language's fixed Phase 1 tokenizer).
        block_size: Model context length.
        prompt_loss_weight: Loss weight on prompt tokens (answer tokens = 1).
        seed: Base seed; epoch ``e`` shuffles with ``seed + e``.
    """

    def __init__(self, examples: list[dict], sp, block_size: int,
                 prompt_loss_weight: float, seed: int):
        self.block_size = block_size
        self.seed = seed
        self.encoded: list[tuple[list[int], list[float], list[bool]]] = []
        for ex in examples:
            ids, is_ans = encode_example(sp, ex["prompt"], ex["answer"])
            if len(ids) > block_size + 1:
                raise ValueError(f"example {ex.get('id')} has {len(ids)} tokens "
                                 f"> block_size + 1 = {block_size + 1}")
            w = [1.0 if a else prompt_loss_weight for a in is_ans]
            self.encoded.append((ids, w, is_ans))
        self.n_tokens = sum(len(ids) for ids, _, _ in self.encoded)

    def rows_for_epoch(self, epoch: int) -> list[list[int]]:
        """Return the example indices packed into each row for one epoch.

        Pure function of ``(seed, epoch)``: the same call always yields the
        same rows, which is what resume relies on.
        """
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.encoded))
        rows, cur, used = [], [], 0
        cap = self.block_size + 1
        for i in order:
            n = len(self.encoded[i][0])
            if used + n > cap:
                rows.append(cur)
                cur, used = [], 0
            cur.append(int(i))
            used += n
        if cur:
            rows.append(cur)
        return rows

    def build_batch(self, rows: list[list[int]], device: str):
        """Materialize rows as ``(x, y, w, a)`` tensors.

        ``x`` and ``y`` are the usual shifted pair; ``w`` is the per-target
        loss weight (0 on padding); ``a`` marks targets that are answer tokens
        (or the closing ``</s>``), used to report an answer-only loss whatever
        the prompt weight is.
        """
        B, T = len(rows), self.block_size
        x = torch.zeros((B, T), dtype=torch.long)
        y = torch.full((B, T), -1, dtype=torch.long)
        w = torch.zeros((B, T), dtype=torch.float32)
        a = torch.zeros((B, T), dtype=torch.bool)
        for r, row in enumerate(rows):
            ids, wts, ans = [], [], []
            for i in row:
                ids.extend(self.encoded[i][0])
                wts.extend(self.encoded[i][1])
                ans.extend(self.encoded[i][2])
            n = len(ids)
            x[r, :n - 1] = torch.tensor(ids[:-1])
            y[r, :n - 1] = torch.tensor(ids[1:])
            w[r, :n - 1] = torch.tensor(wts[1:])      # weight follows the TARGET
            a[r, :n - 1] = torch.tensor(ans[1:])
        return x.to(device), y.to(device), w.to(device), a.to(device)


def weighted_loss(logits: torch.Tensor, y: torch.Tensor,
                  w: torch.Tensor) -> torch.Tensor:
    """Weighted mean cross-entropy over the targets.

    Args:
        logits: ``(B, T, V)``.
        y: Targets ``(B, T)``, ``-1`` = ignore.
        w: Per-target weights ``(B, T)``; pass the answer mask as a float
            tensor to get an answer-only loss.

    Returns:
        Scalar loss.
    """
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1),
                         ignore_index=-1, reduction="none").view_as(y)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


# --------------------------------------------------------------- checkpoint

def save_checkpoint(path: Path, model, optimizer, scaler, step: int,
                    tokens_seen: int, epoch: int, batch_index: int,
                    best_metric: float, config: dict) -> None:
    """Write a complete, resumable finetune checkpoint atomically.

    Same key layout as pretraining (``common/train.py``) so every Phase 2 tool
    -- ``load_model_from_checkpoint``, the attention toolkit, the LM evaluator
    -- reads it unchanged. ``finetune`` adds the position within the epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_metric,      # here: best validation exact match
        "config": config,
        "numpy_rng_state": None,           # the packer is a pure function of epoch
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "finetune": {"epoch": epoch, "batch_index": batch_index,
                     "init_from": config.get("init_from"),
                     "best_val_exact_match": best_metric},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, scaler, device: str) -> dict:
    """Restore weights, optimizer, scaler and RNG state in place."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if ck.get("torch_rng_state") is not None:
        s = ck["torch_rng_state"]
        torch.set_rng_state(s.cpu() if hasattr(s, "cpu") else s)
    if ck.get("cuda_rng_state") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                          for s in ck["cuda_rng_state"]])
        except (RuntimeError, ValueError) as e:
            logger.warning("could not restore CUDA RNG state (%s)", e)
    return ck


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate_loss(model, data: PackedReasoningData, batch_size: int,
                  max_batches: int, device: str, ctx) -> tuple[float, float]:
    """Mean weighted loss and answer-only loss over deterministic val rows."""
    model.eval()
    rows = data.rows_for_epoch(0)
    tot, tot_ans, n = 0.0, 0.0, 0
    for b in range(0, min(len(rows), max_batches * batch_size), batch_size):
        x, y, w, a = data.build_batch(rows[b:b + batch_size], device)
        with ctx:
            logits, _, _ = model(x)
        tot += weighted_loss(logits, y, w).item()
        tot_ans += weighted_loss(logits, y, a.float()).item()
        n += 1
    model.train()
    return tot / max(n, 1), tot_ans / max(n, 1)


# ------------------------------------------------------------------ training

def build_finetune_model(config: dict, device: str) -> tuple[GPT, dict]:
    """Instantiate the model from the pretrained checkpoint's own config.

    The architecture is read from ``init_from`` rather than from the finetune
    YAML, so the finetune can never silently disagree with the weights it is
    loading. The resulting ``model`` block is copied into the finetune config
    so the finetune checkpoint is self-describing too.
    """
    ck = torch.load(config["init_from"], map_location="cpu", weights_only=False)
    m = dict(ck["config"]["model"])
    m["dropout"] = config["train"].get("dropout", m.get("dropout", 0.1))
    config["model"] = m
    config["pretrained"] = {"checkpoint": config["init_from"],
                            "step": ck.get("step"),
                            "tokens_seen": ck.get("tokens_seen"),
                            "name": ck["config"].get("name")}
    gcfg = GPTConfig(vocab_size=m["vocab_size"], d_model=m["d_model"],
                     n_layer=m["n_layer"], n_head=m["n_head"], d_ff=m["d_ff"],
                     block_size=m["block_size"], dropout=m["dropout"],
                     bias=m["bias"], tie_embeddings=m["tie_embeddings"],
                     positional=m["positional"])
    model = GPT(gcfg)
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def finetune(config_path: Path, resume: bool = False,
             max_steps: int | None = None, out_dir_override: str | None = None,
             init_from_override: str | None = None,
             restore_from: str | None = None,
             session_steps: int | None = None) -> dict:
    """Run (or resume) reasoning finetuning for one model.

    Args:
        config_path: Finetune YAML.
        resume: Continue from ``latest.pt`` in the output directory.
        max_steps: Cap the TOTAL run (smoke tests). This also shortens the
            cosine schedule, so it must be the same across resumes of one run.
        out_dir_override: Write checkpoints elsewhere.
        init_from_override: Use a different pretrained checkpoint.
        restore_from: Read-only directory to seed ``latest.pt`` from (Kaggle).
        session_steps: Stop after this many steps IN THIS SESSION, checkpoint,
            and exit cleanly without altering the schedule -- the finetune
            analogue of pretraining's ``--session-tokens``.

    Returns:
        Final summary dict (also appended to the log).
    """
    import sentencepiece as spm

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if init_from_override:
        config["init_from"] = init_from_override
    tcfg, ecfg, pcfg, dcfg = (config["train"], config["eval"],
                              config["paths"], config["data"])
    out_dir = Path(out_dir_override or pcfg["out_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    dtype = tcfg.get("dtype", "float16") if device_type == "cuda" else "float32"
    ctx = make_ctx(device, dtype)
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    model, pre_ck = build_finetune_model(config, device)
    logger.info("[%s] init from %s (pretrain step %s), %s parameters",
                config["name"], config["init_from"], pre_ck.get("step"),
                f"{model.num_parameters():,}")

    # Fresh AdamW: the pretraining moments describe a different objective and
    # a learning rate ten times larger; carrying them over would make the first
    # finetune steps far too aggressive.
    optimizer = model.configure_optimizers(
        tcfg["weight_decay"], tcfg["learning_rate"],
        (tcfg["beta1"], tcfg["beta2"]), device_type)

    sp = spm.SentencePieceProcessor(model_file=pcfg["tokenizer"])
    block = config["model"]["block_size"]
    train_ex = read_jsonl(Path(dcfg["train"]))
    val_ex = read_jsonl(Path(dcfg["validation"]))
    train_data = PackedReasoningData(train_ex, sp, block,
                                     tcfg["prompt_loss_weight"], tcfg["seed"])
    val_data = PackedReasoningData(val_ex, sp, block,
                                   tcfg["prompt_loss_weight"], tcfg["seed"] + 1)
    val_acc_ex = val_ex[:ecfg["val_accuracy_examples"]]

    B, accum = tcfg["batch_size"], tcfg["grad_accum"]
    rows0 = train_data.rows_for_epoch(0)
    steps_per_epoch = math.ceil(len(rows0) / (B * accum))
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    logger.info("%s train examples -> %s packed rows/epoch (%.1f examples/row, "
                "%.1f%% of positions used) -> %d steps/epoch, %d total",
                f"{len(train_ex):,}", f"{len(rows0):,}",
                len(train_ex) / len(rows0),
                100 * train_data.n_tokens / (len(rows0) * (block + 1)),
                steps_per_epoch, total_steps)

    # ---------------------------------------------------------------- resume
    step, tokens_seen, epoch, batch_index = 0, 0, 0, 0
    best_metric = -1.0
    latest = out_dir / "latest.pt"
    if resume and not latest.exists() and restore_from:
        import shutil
        src = Path(restore_from) / "latest.pt"
        if src.exists():
            latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, latest)
            if (Path(restore_from) / "best.pt").exists():
                shutil.copy2(Path(restore_from) / "best.pt", out_dir / "best.pt")
            logger.info("seeded %s from %s", latest, src)
    if resume and latest.exists():
        ck = load_checkpoint(latest, model, optimizer, scaler, device)
        step, tokens_seen = ck["step"], ck["tokens_seen"]
        epoch, batch_index = ck["finetune"]["epoch"], ck["finetune"]["batch_index"]
        best_metric = ck["finetune"].get("best_val_exact_match", -1.0)
        logger.info("resumed at step %d (epoch %d, batch %d), best val EM %.4f",
                    step, epoch, batch_index, best_metric)
    elif resume:
        logger.warning("--resume given but %s does not exist; starting fresh", latest)

    log_path = Path(pcfg["log_file"])
    if out_dir_override:
        log_path = log_path.with_name(f"{out_dir.name}_finetune.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    def run_eval(tag: str) -> dict:
        """Validation loss, answer loss and exact match; logged and returned."""
        vl, va = evaluate_loss(model, val_data, B, ecfg["eval_batches"], device, ctx)
        r = evaluate_reasoning(model, sp, val_acc_ex, device, ctx,
                               with_choices=False, n_examples_per_task=0)
        rec = {"step": step, "epoch": epoch, "tokens": tokens_seen,
               "val_loss": round(vl, 5), "val_answer_loss": round(va, 5),
               "val_exact_match": r["exact_match"],
               "val_exact_match_by_task": {k: v["accuracy"] for k, v in
                                           r["exact_match_by_task"].items()},
               "tag": tag}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        logger.info("  eval %-6s step %5d | val loss %.4f | answer loss %.4f | "
                    "exact match %.4f", tag, step, vl, va, r["exact_match"])
        return rec

    # A step-0 evaluation puts the pretrained model on the same curve as the
    # finetuned one, so the figure starts from the true baseline.
    if step == 0 and not (resume and latest.exists()):
        run_eval("init")

    # ------------------------------------------------------------- main loop
    model.train()
    t_start, t_last, tokens_last = time.time(), time.time(), tokens_seen
    session_start = step
    stopped_early = False
    done = step >= total_steps
    while not done and epoch < tcfg["epochs"]:
        rows = train_data.rows_for_epoch(epoch)
        n_batches = math.ceil(len(rows) / B)
        while batch_index < n_batches and not done:
            lr = lr_at_step(step, total_steps, tcfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            step_loss_t = torch.zeros((), device=device)
            for _ in range(accum):
                if batch_index >= n_batches:
                    break
                x, y, w, _ = train_data.build_batch(
                    rows[batch_index * B:(batch_index + 1) * B], device)
                batch_index += 1
                with ctx:
                    logits, _, _ = model(x)
                loss = weighted_loss(logits, y, w)
                step_loss_t += loss.detach() / accum
                scaler.scale(loss / accum).backward()
                tokens_seen += int((y >= 0).sum())

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % ecfg["log_interval_steps"] == 0:
                now = time.time()
                tps = (tokens_seen - tokens_last) / max(now - t_last, 1e-9)
                t_last, tokens_last = now, tokens_seen
                rec = {"step": step, "epoch": epoch, "tokens": tokens_seen,
                       "train_loss": round(step_loss_t.item(), 5),
                       "lr": round(lr, 8), "grad_norm": round(float(grad_norm), 4),
                       "tokens_per_sec": round(tps, 1),
                       "elapsed_s": round(now - t_start, 1)}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                logger.info("step %4d/%d | epoch %d | loss %.4f | lr %.2e | "
                            "%s tok/s", step, total_steps, epoch,
                            rec["train_loss"], lr, f"{tps:,.0f}")

            if step % ecfg["eval_interval_steps"] == 0:
                rec = run_eval("train")
                if rec["val_exact_match"] > best_metric:
                    best_metric = rec["val_exact_match"]
                    save_checkpoint(out_dir / "best.pt", model, optimizer, scaler,
                                    step, tokens_seen, epoch, batch_index,
                                    best_metric, config)
                    logger.info("  new best -> %s", out_dir / "best.pt")
            if step % ecfg["checkpoint_interval_steps"] == 0:
                save_checkpoint(latest, model, optimizer, scaler, step,
                                tokens_seen, epoch, batch_index, best_metric,
                                config)
            done = step >= total_steps
            if session_steps is not None and step - session_start >= session_steps and not done:
                stopped_early = True
                break
        if stopped_early:
            break
        if not done:
            epoch += 1
            batch_index = 0

    if stopped_early:
        # Checkpoint the exact position (epoch + batch index) and leave the
        # schedule alone; a --resume continues as if never interrupted.
        save_checkpoint(latest, model, optimizer, scaler, step, tokens_seen,
                        epoch, batch_index, best_metric, config)
        log_f.close()
        logger.info("PAUSED at step %d/%d (epoch %d, batch %d) -> %s; "
                    "resume with the SAME config and --resume",
                    step, total_steps, epoch, batch_index, latest)
        return {"step": step, "paused": True}

    # ------------------------------------------------------------------ final
    rec = run_eval("final")
    if rec["val_exact_match"] >= best_metric:
        best_metric = rec["val_exact_match"]
        save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, step,
                        tokens_seen, epoch, batch_index, best_metric, config)
    save_checkpoint(latest, model, optimizer, scaler, step, tokens_seen,
                    epoch, batch_index, best_metric, config)
    summary = {"step": step, "tokens": tokens_seen, "epochs_completed": epoch,
               "final_val_loss": rec["val_loss"],
               "final_val_answer_loss": rec["val_answer_loss"],
               "final_val_exact_match": rec["val_exact_match"],
               "best_val_exact_match": best_metric,
               "wall_clock_s": round(time.time() - t_start, 1)}
    log_f.write(json.dumps(summary) + "\n")
    log_f.close()
    logger.info("FINAL | %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap the run (smoke tests).")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--init-from", default=None,
                   help="Override the pretrained checkpoint path.")
    p.add_argument("--restore-from", default=None,
                   help="Read-only dir holding a latest.pt to resume from.")
    p.add_argument("--session-steps", type=int, default=None,
                   help="Stop after this many steps this session (schedule "
                        "unchanged), checkpoint, exit.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    finetune(args.config, resume=args.resume, max_steps=args.max_steps,
             out_dir_override=args.out_dir, init_from_override=args.init_from,
             restore_from=args.restore_from, session_steps=args.session_steps)


if __name__ == "__main__":
    main()

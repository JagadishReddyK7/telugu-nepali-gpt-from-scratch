"""
Pretraining loop for one model, resumable across Colab preemptions.

    python -m common.train --config telugu/configs/model.yaml
    python -m common.train --config telugu/configs/model.yaml --resume
    python -m common.train --config telugu/configs/model.yaml --benchmark

CHECKPOINTING IS THE POINT
--------------------------
Free Colab preempts sessions without warning, and a full run here is several
hours. Every checkpoint therefore carries everything needed to continue as if
nothing happened:

    model weights        state of the network
    optimizer state      AdamW's first and second moment estimates -- discarding
                         these restarts the moment estimates from zero, which
                         produces a visible loss spike on every resume
    scaler state         fp16 loss-scale, for the same reason
    step, tokens_seen    position in the run AND the scheduler state, because
                         the learning rate here is a pure function of
                         tokens_seen (see `lr_at`) rather than a stateful
                         object -- a design choice made precisely so that
                         resume is exact rather than approximate
    config               the architecture, so a checkpoint can be rebuilt into
                         the right model without consulting any other file
    rng states           numpy (data sampler) and torch, so a resumed run does
                         not re-draw the same training windows it already saw

Checkpoints are written to a temporary file and then atomically renamed. A
preemption during a 300MB write would otherwise leave a truncated file where
the last good checkpoint used to be -- turning a recoverable interruption into
a lost run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from common.data import TokenDataset
from common.model import GPT, GPTConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- scheduling

def lr_at(tokens_seen: int, cfg: dict) -> float:
    """Learning rate as a pure function of tokens consumed.

    Linear warmup then cosine decay to ``min_learning_rate``. Being a pure
    function rather than a stateful ``torch.optim.lr_scheduler`` object is
    deliberate: on resume the schedule is recomputed from ``tokens_seen``, so it
    is exactly right without needing to replay steps or trust a serialized
    scheduler.

    Warmup exists because AdamW's second-moment estimate is unreliable in the
    first few hundred steps -- it has seen too few gradients to estimate their
    scale -- so a full-size step early on can move weights far enough to
    destabilize the run.

    Args:
        tokens_seen: Tokens consumed so far.
        cfg: The ``train`` block of the model config.

    Returns:
        The learning rate for this point in the run.
    """
    warmup = cfg["warmup_tokens"]
    total = cfg["max_tokens"]
    lr, lr_min = cfg["learning_rate"], cfg["min_learning_rate"]

    if tokens_seen < warmup:
        return lr * tokens_seen / max(warmup, 1)
    if tokens_seen > total:
        return lr_min
    progress = (tokens_seen - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


# -------------------------------------------------------------- checkpointing

def save_checkpoint(path: Path, model, optimizer, scaler, step: int,
                    tokens_seen: int, best_val: float, config: dict,
                    data_rng: np.random.Generator) -> None:
    """Write a complete, resumable checkpoint atomically.

    Args:
        path: Destination ``.pt`` file.
        model: The model.
        optimizer: AdamW instance.
        scaler: GradScaler (may be disabled).
        step: Optimizer steps completed.
        tokens_seen: Training tokens consumed.
        best_val: Best validation loss seen so far.
        config: The full parsed config dict.
        data_rng: The training sampler's RNG, so windows are not re-drawn.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "config": config,
        "numpy_rng_state": data_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        # Dropout draws from the CUDA generator when the model is on GPU, NOT
        # from the CPU generator above. Saving only the CPU state makes resume
        # look correct on a CPU smoke test and silently diverge on the actual
        # GPU run, because every dropout mask after the resume differs.
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "learning_rate_at_save": lr_at(tokens_seen, config["train"]),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic on POSIX and on Windows for same-volume


def sync_to_durable(src: Path, dst_dir: str | Path) -> None:
    """Mirror a checkpoint to durable storage (e.g. a mounted Drive folder).

    Colab writes checkpoints to the instance's local disk, which is fast but is
    destroyed when the session ends. Drive survives, but writing 300MB to a FUSE
    mount every few minutes is slow enough to matter -- so the sync is periodic
    rather than on every save, and the training loop never blocks on Drive for
    its own correctness. A lost sync costs the interval; a lost local disk
    without any sync costs the whole session.

    Copies via a temporary name and renames, so a session that dies mid-copy
    cannot leave a truncated file where the last good checkpoint was.

    Args:
        src: Local checkpoint file.
        dst_dir: Durable destination directory.
    """
    import shutil
    dst_dir = Path(dst_dir)
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst_dir / (src.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst_dir / src.name)
        logger.info("  synced %s -> %s", src.name, dst_dir)
    except OSError as e:
        # A Drive hiccup must not kill a multi-hour run. Warn and continue --
        # the local checkpoint is still valid and the next sync will retry.
        logger.warning("  sync to %s failed (%s); continuing", dst_dir, e)


def load_checkpoint(path: Path, model, optimizer, scaler, device: str) -> dict:
    """Restore a checkpoint in place.

    Args:
        path: Checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore moments into (may be None).
        scaler: GradScaler to restore (may be None).
        device: Map location.

    Returns:
        Dict with ``step``, ``tokens_seen``, ``best_val_loss`` and the saved
        RNG states.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if ck.get("torch_rng_state") is not None:
        state = ck["torch_rng_state"]
        torch.set_rng_state(state.cpu() if hasattr(state, "cpu") else state)
    if ck.get("cuda_rng_state") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                          for s in ck["cuda_rng_state"]])
        except (RuntimeError, ValueError) as e:
            # Resuming on a machine with a different GPU count. Dropout masks
            # will differ from the original run; say so rather than pretending
            # the resume was exact.
            logger.warning("could not restore CUDA RNG state (%s); dropout "
                           "masks will differ from the original run", e)
    return ck


def build_model_from_config(config: dict, device: str) -> GPT:
    """Instantiate the model described by a parsed config dict."""
    m = config["model"]
    gcfg = GPTConfig(
        vocab_size=m["vocab_size"], d_model=m["d_model"], n_layer=m["n_layer"],
        n_head=m["n_head"], d_ff=m["d_ff"], block_size=m["block_size"],
        dropout=m["dropout"], bias=m["bias"],
        tie_embeddings=m["tie_embeddings"], positional=m["positional"],
    )
    return GPT(gcfg).to(device)


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batch_size: int,
             max_batches: int, ctx) -> float:
    """Mean cross-entropy over deterministic, non-overlapping held-out windows.

    Uses ``sequential_batches`` rather than random draws so the number is
    reproducible across runs and across resumes.

    Args:
        model: The model.
        dataset: Held-out dataset.
        batch_size: Sequences per batch.
        max_batches: Cap on batches, to keep evaluation fast during training.
        ctx: Autocast context manager.

    Returns:
        Mean loss.
    """
    model.eval()
    losses, n = 0.0, 0
    for x, y in dataset.sequential_batches(batch_size, max_batches):
        with ctx:
            _, loss, _ = model(x, y)
        losses += loss.item()
        n += 1
    model.train()
    return losses / max(n, 1)


# -------------------------------------------------------------------- training

def train(config_path: Path, resume: bool = False, benchmark: bool = False,
          benchmark_steps: int = 50, override_max_tokens: int | None = None,
          positional_override: str | None = None,
          out_dir_override: str | None = None,
          session_tokens: int | None = None,
          max_hours: float | None = None,
          sync_dir: str | None = None,
          sync_every: int = 4,
          restore_from: str | None = None) -> None:
    """Run (or resume, or benchmark) pretraining for one model.

    Args:
        config_path: Path to a model YAML.
        resume: Continue from ``latest.pt`` in the output directory.
        benchmark: Measure throughput and exit without training.
        benchmark_steps: Steps to time when benchmarking.
        override_max_tokens: Change the TOTAL token budget for the run. This
            also changes the learning-rate schedule, since the cosine decays to
            ``max_tokens`` -- so it must stay constant across resumes of the
            same run. Use it for smoke tests, not for stopping early.
        positional_override: Force ``"learned"`` or ``"none"`` -- used by the
            positional-embedding ablation so it needs no separate config file.
        out_dir_override: Write checkpoints somewhere other than the config's
            ``out_dir``, so an ablation cannot overwrite the main run.
        session_tokens: Stop after consuming this many tokens IN THIS SESSION,
            checkpoint, and exit cleanly. Does not touch the schedule.
        max_hours: Stop after this many hours in this session, checkpoint, and
            exit cleanly. Intended for Colab's session cap -- finishing under
            your own control beats being preempted mid-write.
        sync_dir: Durable directory (a mounted Drive folder) to mirror
            checkpoints into, so a destroyed Colab instance does not take the
            run with it.
        sync_every: Mirror once every this many checkpoint saves.
        restore_from: READ-ONLY directory to seed ``latest.pt`` from when the
            output directory is empty. This is the Kaggle pattern: a previous
            notebook version's output is attached at ``/kaggle/input/...``,
            which cannot be written to, so it is a restore source only and
            never a sync target.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tcfg, mcfg, pcfg, ecfg = (config["train"], config["model"],
                              config["paths"], config["eval"])

    if positional_override:
        mcfg["positional"] = positional_override
        config["ablation"] = {"positional": positional_override}
    if override_max_tokens:
        tcfg["max_tokens"] = override_max_tokens
    out_dir = Path(out_dir_override or pcfg["out_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # fp16 needs a GradScaler; bf16 does not. On a T4 (no bf16) fp16 is correct.
    dtype = tcfg.get("dtype", "float16")
    if device_type == "cpu":
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))
    # torch.amp.GradScaler(device, ...) landed in torch 2.4. Kaggle images move
    # independently of Colab's, so fall back rather than crash three hours into
    # a queued run on an older image.
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    model = build_model_from_config(config, device)
    breakdown = model.parameter_breakdown()
    logger.info("[%s] %s parameters (%s non-embedding, %.1f%% embedding)",
                config["name"], f"{breakdown['total']:,}",
                f"{breakdown['total_non_embedding']:,}",
                breakdown["embedding_share"] * 100)

    optimizer = model.configure_optimizers(
        tcfg["weight_decay"], tcfg["learning_rate"],
        (tcfg["beta1"], tcfg["beta2"]), device_type)

    train_ds = TokenDataset(pcfg["train_bin"], mcfg["block_size"], device,
                            seed=tcfg["seed"])
    val_ds = TokenDataset(pcfg["val_bin"], mcfg["block_size"], device,
                          seed=tcfg["seed"] + 1)

    step, tokens_seen, best_val = 0, 0, float("inf")
    latest = out_dir / "latest.pt"
    # A fresh instance has no local checkpoint, but a durable copy may exist:
    # on Colab a synced folder in Drive, on Kaggle the previous notebook
    # version's output mounted read-only under /kaggle/input. Check both before
    # concluding there is nothing to resume from -- silently restarting a
    # half-finished run at step 0 is the expensive failure here.
    if resume and not latest.exists():
        import shutil
        for candidate, label in ((restore_from, "restore-from"),
                                 (sync_dir, "sync-dir")):
            if not candidate:
                continue
            src = Path(candidate) / "latest.pt"
            if src.exists():
                latest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, latest)
                logger.info("seeded %s from %s (%s)", latest.name, src, label)
                best_src = Path(candidate) / "best.pt"
                if best_src.exists():
                    shutil.copy2(best_src, out_dir / "best.pt")
                break
        else:
            if restore_from or sync_dir:
                logger.info("no checkpoint found in restore sources; "
                            "starting from step 0")

    if resume and latest.exists():
        ck = load_checkpoint(latest, model, optimizer, scaler, device)
        step, tokens_seen = ck["step"], ck["tokens_seen"]
        best_val = ck.get("best_val_loss", float("inf"))
        if ck.get("numpy_rng_state"):
            train_ds.rng.bit_generator.state = ck["numpy_rng_state"]
        logger.info("resumed from %s at step %s, %s tokens (best val %.4f)",
                    latest, f"{step:,}", f"{tokens_seen:,}", best_val)
    elif resume:
        logger.warning("--resume given but %s does not exist; starting fresh",
                       latest)

    B, accum, T = tcfg["batch_size"], tcfg["grad_accum"], mcfg["block_size"]
    tokens_per_step = B * accum * T

    # ------------------------------------------------------------- benchmark
    if benchmark:
        logger.info("benchmarking %d steps at batch=%d accum=%d block=%d ...",
                    benchmark_steps, B, accum, T)
        model.train()
        for _ in range(3):                       # warmup, excluded from timing
            x, y = train_ds.get_batch(B)
            with ctx:
                _, loss, _ = model(x, y)
            scaler.scale(loss).backward()
            optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(benchmark_steps):
            for _ in range(accum):
                x, y = train_ds.get_batch(B)
                with ctx:
                    _, loss, _ = model(x, y)
                scaler.scale(loss / accum).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        tok = benchmark_steps * tokens_per_step
        tps = tok / elapsed
        flops = model.estimate_flops_per_token() * tok
        target = tcfg["max_tokens"]
        print()
        print(f"  device                {torch.cuda.get_device_name(0) if device_type=='cuda' else 'cpu'}")
        print(f"  precision             {dtype}")
        print(f"  batch x accum x block {B} x {accum} x {T} = {tokens_per_step:,} tokens/step")
        print(f"  measured              {tps:,.0f} tokens/sec  ({elapsed/benchmark_steps*1000:.0f} ms/step)")
        print(f"  effective throughput  {flops/elapsed/1e12:.2f} TFLOPS")
        if device_type == "cuda":
            print(f"  peak VRAM             {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print()
        print(f"  {target:,} tokens would take {target/tps/3600:.2f} h "
              f"({target/tokens_per_step:,.0f} steps)")
        print()
        return

    # -------------------------------------------------------------- main loop
    # The log path follows out_dir. Without this an ablation launched with
    # --out-dir would append its loss curve into the MAIN run's jsonl, because
    # the file is opened in append mode -- silently interleaving two runs into
    # one curve.
    log_path = Path(pcfg["log_file"])
    if out_dir_override:
        log_path = log_path.with_name(f"{out_dir.name}_train.jsonl")
        logger.info("log -> %s (follows --out-dir)", log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    logger.info("training %s tokens, %s per step -> %s steps",
                f"{tcfg['max_tokens']:,}", f"{tokens_per_step:,}",
                f"{tcfg['max_tokens'] // tokens_per_step:,}")

    model.train()
    t_start, t_last, tokens_last = time.time(), time.time(), tokens_seen
    session_start_tokens = tokens_seen
    stopped_early = False
    saves_since_sync = 0

    while tokens_seen < tcfg["max_tokens"]:
        # Session limits stop THIS invocation without altering the schedule,
        # which stays a function of the run's total max_tokens. Checkpoint and
        # exit cleanly rather than waiting to be preempted mid-write.
        if session_tokens is not None and (tokens_seen - session_start_tokens) >= session_tokens:
            logger.info("session token budget reached (%s this session)",
                        f"{tokens_seen - session_start_tokens:,}")
            stopped_early = True
            break
        if max_hours is not None and (time.time() - t_start) >= max_hours * 3600:
            logger.info("session time limit reached (%.2f h)",
                        (time.time() - t_start) / 3600)
            stopped_early = True
            break

        lr = lr_at(tokens_seen, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: `accum` micro-batches contribute to one
        # optimizer step, so the effective batch is B*accum without needing the
        # VRAM for it. Each micro-batch's loss is divided by accum so the
        # accumulated gradient is the mean, not the sum.
        # Accumulate the loss across micro-batches so the logged value is the
        # mean over the whole effective batch. Logging only the last
        # micro-batch reports a quantity `accum` times noisier than the signal
        # the optimizer actually stepped on.
        #
        # Accumulated as a DETACHED TENSOR, not via .item(). Calling .item()
        # here would force a host-device synchronization on every micro-batch --
        # `accum` stalls per step, every step, for a number that is only read
        # once every log_interval_steps. The sync happens in the logging branch
        # instead, where it is paid for once.
        step_loss_t = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = train_ds.get_batch(B)
            with ctx:
                _, loss, _ = model(x, y)
            step_loss_t += loss.detach() / accum
            scaler.scale(loss / accum).backward()

        # Unscale before clipping, or the clip threshold would apply to the
        # fp16-scaled gradients and be meaningless.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += tokens_per_step

        if step % ecfg["log_interval_steps"] == 0:
            now = time.time()
            tps = (tokens_seen - tokens_last) / max(now - t_last, 1e-9)
            t_last, tokens_last = now, tokens_seen
            step_loss = step_loss_t.item()   # single sync, once per log interval
            rec = {"step": step, "tokens": tokens_seen,
                   "train_loss": round(step_loss, 5), "lr": round(lr, 8),
                   "grad_norm": round(float(grad_norm), 4),
                   "tokens_per_sec": round(tps, 1),
                   "elapsed_s": round(now - t_start, 1)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            pct = 100 * tokens_seen / tcfg["max_tokens"]
            eta = (tcfg["max_tokens"] - tokens_seen) / max(tps, 1) / 3600
            logger.info("step %s | %s tok (%.1f%%) | loss %.4f | lr %.2e | "
                        "%s tok/s | eta %.1fh", f"{step:,}", f"{tokens_seen:,}",
                        pct, step_loss, lr, f"{tps:,.0f}", eta)

        if step % ecfg["eval_interval_steps"] == 0:
            val = evaluate(model, val_ds, B, ecfg["eval_batches"], ctx)
            rec = {"step": step, "tokens": tokens_seen,
                   "val_loss": round(val, 5),
                   "val_ppl": round(math.exp(min(val, 20)), 3)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            logger.info("  eval step %s | val loss %.4f | ppl %.2f",
                        f"{step:,}", val, math.exp(min(val, 20)))
            if val < best_val:
                best_val = val
                save_checkpoint(out_dir / "best.pt", model, optimizer, scaler,
                                step, tokens_seen, best_val, config, train_ds.rng)
                logger.info("  new best -> %s", out_dir / "best.pt")

        if step % ecfg["checkpoint_interval_steps"] == 0:
            save_checkpoint(latest, model, optimizer, scaler, step,
                            tokens_seen, best_val, config, train_ds.rng)
            saves_since_sync += 1
            if sync_dir and saves_since_sync >= sync_every:
                sync_to_durable(latest, sync_dir)
                if (out_dir / "best.pt").exists():
                    sync_to_durable(out_dir / "best.pt", sync_dir)
                saves_since_sync = 0

    if stopped_early:
        save_checkpoint(latest, model, optimizer, scaler, step, tokens_seen,
                        best_val, config, train_ds.rng)
        if sync_dir:
            sync_to_durable(latest, sync_dir)
            if (out_dir / "best.pt").exists():
                sync_to_durable(out_dir / "best.pt", sync_dir)
        pct = 100 * tokens_seen / tcfg["max_tokens"]
        logger.info("PAUSED at step %s | %s / %s tokens (%.1f%%) | checkpoint %s",
                    f"{step:,}", f"{tokens_seen:,}", f"{tcfg['max_tokens']:,}",
                    pct, latest)
        logger.info("resume with the SAME config and --resume")
        log_f.close()
        return

    # Final evaluation and checkpoint.
    val = evaluate(model, val_ds, B, ecfg["eval_batches"] * 4, ctx)
    logger.info("FINAL | step %s | %s tokens | val loss %.4f | ppl %.2f",
                f"{step:,}", f"{tokens_seen:,}", val, math.exp(min(val, 20)))
    save_checkpoint(latest, model, optimizer, scaler, step, tokens_seen,
                    best_val, config, train_ds.rng)
    if val < best_val:
        save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, step,
                        tokens_seen, val, config, train_ds.rng)
    log_f.write(json.dumps({"step": step, "tokens": tokens_seen,
                            "final_val_loss": round(val, 5),
                            "wall_clock_s": round(time.time() - t_start, 1)}) + "\n")
    log_f.close()
    if sync_dir:
        sync_to_durable(latest, sync_dir)
        if (out_dir / "best.pt").exists():
            sync_to_durable(out_dir / "best.pt", sync_dir)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true",
                   help="Continue from latest.pt. Safe to pass always.")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure tokens/sec and exit without training.")
    p.add_argument("--benchmark-steps", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Override the token budget, for smoke tests.")
    p.add_argument("--positional", choices=["learned", "none"], default=None,
                   help="Override positional encoding (the ablation).")
    p.add_argument("--out-dir", default=None,
                   help="Override checkpoint directory (use for the ablation).")
    p.add_argument("--session-tokens", type=int, default=None,
                   help="Stop after this many tokens THIS session, then exit "
                        "cleanly. Does not change the LR schedule.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Stop after this many hours this session, then exit "
                        "cleanly. Use just under Colab's session cap.")
    p.add_argument("--sync-dir", default=None,
                   help="Mirror checkpoints to this durable directory (a "
                        "mounted Drive folder). Also searched on --resume when "
                        "the local checkpoint is missing.")
    p.add_argument("--sync-every", type=int, default=4,
                   help="Checkpoint saves between durable syncs (default 4).")
    p.add_argument("--restore-from", default=None,
                   help="Read-only directory to seed latest.pt from when the "
                        "output dir is empty (Kaggle: a previous version's "
                        "output under /kaggle/input/...).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    train(args.config, resume=args.resume, benchmark=args.benchmark,
          benchmark_steps=args.benchmark_steps,
          override_max_tokens=args.max_tokens,
          positional_override=args.positional,
          out_dir_override=args.out_dir,
          session_tokens=args.session_tokens,
          max_hours=args.max_hours,
          sync_dir=args.sync_dir,
          sync_every=args.sync_every,
          restore_from=args.restore_from)


if __name__ == "__main__":
    main()

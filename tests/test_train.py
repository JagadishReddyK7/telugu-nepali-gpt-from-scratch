"""
Tests for the training loop: schedule, checkpoint completeness, exact resume.

    python tests/test_train.py

THE TEST THAT MATTERS MOST
--------------------------
``test_resume_is_bit_exact`` trains a model straight through, then trains an
identical model in two halves with a checkpoint in between, and asserts the
weights are *identical*. The spec says a run that cannot resume after
interruption loses marks, and on free Colab interruption is the normal case
rather than the exception.

A resume that merely "looks fine" -- loss continues downward, nothing crashes --
can still be quietly wrong: dropping AdamW's moment estimates, or re-drawing
training windows already seen, or recomputing the learning rate from the wrong
denominator all produce a run that trains but does not train *the same*. Only a
weight-level comparison catches those.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.train import lr_at, save_checkpoint, load_checkpoint, train  # noqa: E402
from common.model import GPT, GPTConfig  # noqa: E402


TINY_CONFIG = {
    "name": "model_test", "language": "test",
    "model": {"vocab_size": 200, "d_model": 64, "n_layer": 2, "n_head": 4,
              "d_ff": 256, "block_size": 32, "dropout": 0.1, "bias": True,
              "tie_embeddings": True, "positional": "learned"},
    "train": {"batch_size": 8, "grad_accum": 2, "max_tokens": 100_000,
              "learning_rate": 1.0e-3, "min_learning_rate": 1.0e-4,
              "warmup_tokens": 10_000, "weight_decay": 0.1, "beta1": 0.9,
              "beta2": 0.95, "grad_clip": 1.0, "lr_schedule": "cosine",
              "dtype": "float32", "compile": False, "seed": 1337},
    "eval": {"eval_interval_steps": 50, "eval_batches": 5,
             "checkpoint_interval_steps": 50, "keep_last_n_checkpoints": 2,
             "log_interval_steps": 1000},
    "paths": {},
}


def make_workspace(root: Path) -> Path:
    """Write a tiny corpus and config into ``root``; return the config path."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    seq = (np.arange(400_000) % 97 + rng.integers(0, 3, 400_000)) % 200
    seq.astype(np.uint16).tofile(root / "data" / "train.bin")
    seq[:40_000].astype(np.uint16).tofile(root / "data" / "val.bin")

    cfg = {k: (v.copy() if isinstance(v, dict) else v)
           for k, v in TINY_CONFIG.items()}
    cfg["paths"] = {
        "tokenizer": "none",
        "train_bin": str(root / "data" / "train.bin"),
        "val_bin": str(root / "data" / "val.bin"),
        "test_bin": str(root / "data" / "val.bin"),
        "out_dir": str(root / "ckpt"),
        "log_file": str(root / "logs" / "train.jsonl"),
    }
    path = root / "model.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ------------------------------------------------------------------- schedule

def test_lr_schedule_shape():
    """Warmup rises linearly from 0; cosine decays to exactly min_lr."""
    cfg = TINY_CONFIG["train"]
    peak, floor = cfg["learning_rate"], cfg["min_learning_rate"]
    warm, total = cfg["warmup_tokens"], cfg["max_tokens"]

    assert lr_at(0, cfg) == 0.0
    assert abs(lr_at(warm // 2, cfg) - peak / 2) < 1e-12
    assert abs(lr_at(warm, cfg) - peak) < 1e-9
    assert abs(lr_at(total, cfg) - floor) < 1e-9
    assert abs(lr_at(total * 5, cfg) - floor) < 1e-12, "must clamp past the end"

    mid = lr_at((warm + total) // 2, cfg)
    assert floor < mid < peak

    # Monotone decreasing after warmup.
    xs = np.linspace(warm, total, 50).astype(int)
    ys = [lr_at(int(x), cfg) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:])), "cosine not monotone"
    print(f"  warmup 0 -> {peak:.1e}, cosine -> {floor:.1e}, monotone OK")


def test_lr_schedule_is_pure_function_of_tokens():
    """The same token count must always give the same LR -- this is what makes
    resume exact without serializing a scheduler object."""
    cfg = TINY_CONFIG["train"]
    for t in (0, 5_000, 10_000, 50_000, 99_999):
        assert lr_at(t, cfg) == lr_at(t, cfg)
    print("  schedule is stateless OK")


# --------------------------------------------------------------- checkpoints

def test_checkpoint_contains_everything_the_spec_requires():
    """weights + optimizer + scheduler state + step + config, per the spec."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = GPT(GPTConfig(vocab_size=200, n_layer=2, n_head=4, d_model=64,
                              d_ff=256, block_size=32, dropout=0.0))
        opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # One step so the optimizer actually has moment estimates to save.
        x = torch.randint(0, 200, (2, 16))
        _, loss, _ = model(x, x)
        loss.backward()
        opt.step()

        ck_path = root / "latest.pt"
        rng = np.random.default_rng(0)
        save_checkpoint(ck_path, model, opt, scaler, 42, 12345, 1.23,
                        TINY_CONFIG, rng)

        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "step", "tokens_seen", "config",
                    "best_val_loss", "numpy_rng_state", "torch_rng_state",
                    "learning_rate_at_save"):
            assert key in ck and ck[key] is not None, f"missing {key}"
        assert ck["step"] == 42 and ck["tokens_seen"] == 12345
        assert len(ck["optimizer"]["state"]) > 0, "optimizer moments not saved"
        assert ck["config"]["model"]["d_model"] == 64
        print(f"  checkpoint carries all required state "
              f"({len(ck['optimizer']['state'])} optimizer tensors) OK")


def test_checkpoint_write_is_atomic():
    """No .tmp file may survive a completed save."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = GPT(GPTConfig(vocab_size=200, n_layer=1, n_head=4, d_model=64,
                              d_ff=256, block_size=32, dropout=0.0))
        opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = root / "latest.pt"
        save_checkpoint(path, model, opt, scaler, 1, 1, 1.0, TINY_CONFIG,
                        np.random.default_rng(0))
        leftovers = list(root.glob("*.tmp"))
        assert not leftovers, f"temporary files left behind: {leftovers}"
        assert path.exists()
        print("  atomic write leaves no .tmp OK")


def test_checkpoint_round_trip_restores_weights():
    """Loading a checkpoint must reproduce the saved weights exactly."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = GPTConfig(vocab_size=200, n_layer=2, n_head=4, d_model=64,
                        d_ff=256, block_size=32, dropout=0.0)
        a = GPT(cfg)
        opt = a.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = root / "c.pt"
        save_checkpoint(path, a, opt, scaler, 7, 700, 2.0, TINY_CONFIG,
                        np.random.default_rng(0))

        b = GPT(cfg)
        before = max((a.state_dict()[k].float() - b.state_dict()[k].float())
                     .abs().max().item() for k in a.state_dict())
        assert before > 0, "fresh models should differ before loading"

        opt_b = b.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        load_checkpoint(path, b, opt_b, scaler, "cpu")
        after = max((a.state_dict()[k].float() - b.state_dict()[k].float())
                    .abs().max().item() for k in a.state_dict())
        assert after == 0.0, f"weights differ after load: {after}"
        print(f"  round trip: {before:.3e} before -> {after:.1e} after OK")


# ------------------------------------------------------------ THE BIG ONE

def test_resume_is_bit_exact():
    """Training in two halves must give byte-identical weights to one run.

    Run A:  0 ----------------> 100k tokens
    Run B:  0 ----> 50k | resume ----> 100k

    Dropout is deliberately NON-ZERO here. With dropout disabled the test
    passes even if no RNG state is saved at all, because there is no
    stochasticity left to diverge -- which is exactly how a missing CUDA RNG
    state hides on a CPU smoke test and then silently breaks the GPU run.

    Both must land on the same weights. Note that run B stops via
    ``session_tokens``, NOT by lowering ``max_tokens`` -- lowering max_tokens
    would compress the cosine schedule into the first half and the two runs
    would legitimately diverge. That distinction is the whole reason
    ``--session-tokens`` exists.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        a_root = root / "a"
        a_root.mkdir()
        cfg_a = make_workspace(a_root)
        train(cfg_a, resume=False)
        w_a = torch.load(a_root / "ckpt" / "latest.pt", map_location="cpu",
                         weights_only=False)

        b_root = root / "b"
        b_root.mkdir()
        cfg_b = make_workspace(b_root)
        train(cfg_b, resume=False, session_tokens=50_000)
        paused = torch.load(b_root / "ckpt" / "latest.pt", map_location="cpu",
                            weights_only=False)
        assert paused["tokens_seen"] < w_a["tokens_seen"], "did not pause early"
        train(cfg_b, resume=True)
        w_b = torch.load(b_root / "ckpt" / "latest.pt", map_location="cpu",
                         weights_only=False)

        assert w_a["step"] == w_b["step"], (w_a["step"], w_b["step"])
        assert w_a["tokens_seen"] == w_b["tokens_seen"]

        diff = max((w_a["model"][k].float() - w_b["model"][k].float())
                   .abs().max().item() for k in w_a["model"])
        assert diff == 0.0, f"resume diverged: max weight difference {diff:.3e}"
        print(f"  paused at {paused['tokens_seen']:,}, resumed to "
              f"{w_b['tokens_seen']:,}; max weight diff {diff:.1e} -> BIT-EXACT OK")


def test_session_tokens_does_not_alter_the_schedule():
    """Pausing must not change the learning rate at any token count."""
    cfg = dict(TINY_CONFIG["train"])
    full = [lr_at(t, cfg) for t in range(0, 100_000, 5_000)]
    # A "paused" run recomputes from the same config; nothing about the pause
    # enters lr_at, so the curve must be identical.
    again = [lr_at(t, cfg) for t in range(0, 100_000, 5_000)]
    assert full == again
    # Whereas lowering max_tokens genuinely changes it -- the failure mode the
    # session-token flag exists to avoid.
    short = dict(cfg, max_tokens=50_000)
    assert lr_at(40_000, short) != lr_at(40_000, cfg)
    print("  pausing leaves the schedule untouched; shortening does not OK")


def test_checkpoint_saves_cuda_rng_when_available():
    """CUDA RNG state must be captured, because dropout uses it on GPU.

    Saving only ``torch.get_rng_state()`` (the CPU generator) makes resume look
    exact on CPU and diverge on GPU: every dropout mask drawn after the resume
    differs from the uninterrupted run. On a CPU-only machine this test asserts
    the key exists and is None; on a GPU machine it asserts real state is saved.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model = GPT(GPTConfig(vocab_size=200, n_layer=1, n_head=4, d_model=64,
                              d_ff=256, block_size=32, dropout=0.1))
        opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = root / "c.pt"
        save_checkpoint(path, model, opt, scaler, 1, 1, 1.0, TINY_CONFIG,
                        np.random.default_rng(0))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert "cuda_rng_state" in ck, "cuda_rng_state key missing entirely"
        if torch.cuda.is_available():
            assert ck["cuda_rng_state"] is not None
            print(f"  CUDA RNG saved for {len(ck['cuda_rng_state'])} device(s) OK")
        else:
            assert ck["cuda_rng_state"] is None
            print("  cuda_rng_state key present (None on CPU-only host) OK")


def test_ablation_writes_a_separate_log_file():
    """--out-dir must redirect the log, or the ablation corrupts the main curve.

    The log is opened in append mode. Without redirection an ablation run would
    interleave its loss curve into the main run's jsonl, and the report's loss
    plot would silently mix two different models.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg_path = make_workspace(root)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        # Log often enough that the main run actually writes lines -- otherwise
        # "the main log did not grow" compares zero to zero and proves nothing.
        cfg["eval"]["log_interval_steps"] = 5
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        main_log = Path(cfg["paths"]["log_file"])

        train(cfg_path, resume=False, session_tokens=20_000)
        assert main_log.exists()
        main_lines = len(main_log.read_text(encoding="utf-8").splitlines())
        assert main_lines > 0, "main log empty; the test would prove nothing"

        train(cfg_path, resume=False, session_tokens=20_000,
              positional_override="none",
              out_dir_override=str(root / "ckpt_nopos"))

        after = len(main_log.read_text(encoding="utf-8").splitlines())
        assert after == main_lines, (
            f"ablation appended {after - main_lines} lines to the main log")
        abl_log = main_log.with_name("ckpt_nopos_train.jsonl")
        assert abl_log.exists(), f"ablation log not written to {abl_log}"
        print(f"  main log untouched ({main_lines} lines); ablation wrote "
              f"{abl_log.name} OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests\n")
    failed = 0
    for t in tests:
        try:
            print(t.__name__)
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

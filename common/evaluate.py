"""
Intrinsic language-model evaluation: cross-entropy, perplexity, bits-per-byte.

    python -m common.evaluate --config telugu/configs/model.yaml --split validation

WHY BITS-PER-BYTE EXISTS
------------------------
Perplexity is *per token*, and Model H and Model L have different tokenizers.
Telugu encodes at 3.97 characters per token, Nepali at 4.34 -- so a Nepali token
carries more text than a Telugu one, and a lower Nepali perplexity would not
mean the model is better at Nepali. It would partly mean Nepali tokens are
bigger targets.

Bits-per-byte removes the tokenizer from the comparison by dividing the model's
total information content by the size of the underlying *text*:

    BPB = (loss_in_nats x n_tokens) / (ln 2 x n_utf8_bytes)
        = loss_in_nats / (ln 2 x bytes_per_token)

Both models are then measured in bits needed per byte of their own language's
UTF-8 text, which is a quantity a tokenizer cannot inflate or deflate. This is
the number the Phase 2 report should lead with when comparing the two resource
tiers; perplexity is reported alongside it because it is the conventional
figure, not because it is the comparable one.

A NOTE ON WHICH BYTES
---------------------
``bytes_per_token`` is taken from the split's entry in ``token_counts.json``,
written during Phase 1 encoding by counting the UTF-8 length of exactly the
documents that were encoded. Recomputing it here by decoding would risk
measuring something subtly different -- a detokenization artifact, say -- and
the two numbers must describe the same text for the ratio to mean anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from common.data import TokenDataset
from common.model import GPT, GPTConfig

logger = logging.getLogger(__name__)

LN2 = math.log(2.0)


def load_model_from_checkpoint(ckpt_path: Path, device: str) -> tuple[GPT, dict]:
    """Rebuild a model from a checkpoint alone.

    The checkpoint carries its own config, so nothing external is consulted --
    a checkpoint and this function are together sufficient to reproduce the
    exact model that produced it.

    Args:
        ckpt_path: Path to a ``.pt`` written by ``common.train``.
        device: Device to place the model on.

    Returns:
        ``(model, checkpoint_dict)``.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    cfg = GPTConfig(
        vocab_size=m["vocab_size"], d_model=m["d_model"], n_layer=m["n_layer"],
        n_head=m["n_head"], d_ff=m["d_ff"], block_size=m["block_size"],
        dropout=0.0,                      # never dropout at evaluation time
        bias=m["bias"], tie_embeddings=m["tie_embeddings"],
        positional=m["positional"],
    )
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def cross_entropy_over_split(model: GPT, dataset: TokenDataset, batch_size: int,
                             ctx, max_batches: int | None = None) -> tuple[float, int]:
    """Mean cross-entropy in nats per token over a whole split.

    Uses non-overlapping sequential windows so the number is a property of the
    data rather than of a random draw, and is therefore reproducible and
    comparable between the two models.

    Args:
        model: Model in eval mode.
        dataset: Held-out dataset.
        batch_size: Sequences per batch.
        ctx: Autocast context.
        max_batches: Optional cap, for a quick estimate.

    Returns:
        ``(mean_loss_nats_per_token, n_tokens_scored)``.
    """
    total_loss, total_tokens = 0.0, 0
    for x, y in dataset.sequential_batches(batch_size, max_batches):
        with ctx:
            _, loss, _ = model(x, y)
        n = y.numel()
        # Weight by token count so a short final batch cannot skew the mean.
        total_loss += loss.item() * n
        total_tokens += n
    if total_tokens == 0:
        raise ValueError("no tokens scored -- split too small for this batch size")
    return total_loss / total_tokens, total_tokens


def bits_per_byte(loss_nats: float, bytes_per_token: float) -> float:
    """Convert per-token cross-entropy to bits per UTF-8 byte.

    Args:
        loss_nats: Mean cross-entropy in nats per token.
        bytes_per_token: UTF-8 bytes per token for the same text.

    Returns:
        Bits per byte.
    """
    return loss_nats / (LN2 * bytes_per_token)


def evaluate_split(config: dict, ckpt_path: Path, split: str, device: str,
                   batch_size: int | None = None,
                   max_batches: int | None = None) -> dict:
    """Score one split and return every intrinsic metric the report needs.

    Args:
        config: Parsed model config (for paths and batch size).
        ckpt_path: Checkpoint to evaluate.
        split: ``"validation"`` or ``"test"``.
        device: Device string.
        batch_size: Override the config's batch size.
        max_batches: Cap batches, for a quick check.

    Returns:
        A dict of metrics, JSON-serializable.
    """
    model, ck = load_model_from_checkpoint(ckpt_path, device)
    paths = config["paths"]
    bin_path = paths[f"{split}_bin"] if f"{split}_bin" in paths else paths["val_bin"]

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = config["train"].get("dtype", "float16")
    if device_type == "cpu":
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    bs = batch_size or config["train"]["batch_size"]
    ds = TokenDataset(bin_path, config["model"]["block_size"], device,
                      seed=config["train"]["seed"])

    loss, n_tokens = cross_entropy_over_split(model, ds, bs, ctx, max_batches)

    # bytes_per_token comes from Phase 1's encoding pass, measured on exactly
    # these documents. See the module docstring for why it is not recomputed.
    bpt = None
    counts_path = Path(bin_path).parent / "token_counts.json"
    if counts_path.exists():
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        bpt = counts.get("splits", {}).get(split, {}).get("bytes_per_token")

    result = {
        "language": config.get("language"),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ck.get("step"),
        "checkpoint_tokens_seen": ck.get("tokens_seen"),
        "split": split,
        "tokens_scored": n_tokens,
        "cross_entropy_nats": round(loss, 6),
        "perplexity": round(math.exp(min(loss, 20)), 4),
        "bytes_per_token": bpt,
        "bits_per_byte": round(bits_per_byte(loss, bpt), 6) if bpt else None,
        "vocab_size": config["model"]["vocab_size"],
        "positional": config["model"]["positional"],
    }
    if bpt is None:
        result["bits_per_byte_note"] = (
            "token_counts.json not found beside the .bin, so bytes_per_token is "
            "unknown and BPB could not be computed. BPB is the tokenizer-neutral "
            "metric; without it the two models are not directly comparable.")
    return result


def main() -> None:
    """CLI: evaluate one checkpoint on one split."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Defaults to <out_dir>/best.pt.")
    p.add_argument("--split", default="validation",
                   choices=["validation", "test"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="Write the metrics as JSON here as well as printing.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = args.checkpoint or Path(config["paths"]["out_dir"]) / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = evaluate_split(config, ckpt, args.split, device,
                         args.batch_size, args.max_batches)

    print()
    for k, v in res.items():
        print(f"  {k:28s} {v:,}" if isinstance(v, int) else f"  {k:28s} {v}")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

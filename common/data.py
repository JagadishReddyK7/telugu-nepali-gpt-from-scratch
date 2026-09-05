"""
Batch sampling from the tokenized ``uint16`` corpora produced in Phase 1.

WHY memmap AND NOT A DataLoader
-------------------------------
``train.bin`` is a flat array of token ids -- 1.3GB for Telugu, 1.0GB for
Nepali -- with no per-record structure. Sampling a training window is therefore
just picking an offset and slicing, which ``numpy.memmap`` turns into a page
fault handled by the OS cache rather than a read. There is nothing to collate,
nothing to shuffle in memory, and no Python-level per-example work, so a
``DataLoader`` with worker processes would add overhead and buy nothing.

The corpus does not fit in a Colab instance's RAM. memmap means it never has
to: only the touched pages are resident.

WINDOWS CROSS DOCUMENT BOUNDARIES, DELIBERATELY
-----------------------------------------------
Documents were written as ``<s> ... </s>`` during encoding, concatenated into
one stream. A random window may therefore span the end of one document and the
start of the next. This is standard practice and is not a bug: the model sees
the BOS/EOS tokens and learns that they mark boundaries, which is what lets it
terminate its own generations later. Padding every document to a fixed length
instead would waste a large fraction of each batch on padding for a corpus whose
documents average ~215 tokens against a 512-token context.

TRAIN/VAL SEPARATION
--------------------
train, validation and test are three separate ``.bin`` files written from three
separate ``.jsonl`` splits, which were themselves assigned at the document level
*after* deduplication in Phase 1. There is no window in this loader that can
straddle the split boundary, because the splits are different files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TokenDataset:
    """Random-window sampler over a flat ``uint16`` token array.

    Args:
        bin_path: Path to a ``.bin`` file of little-endian ``uint16`` token ids.
        block_size: Context length. Each sample is ``block_size`` inputs and
            ``block_size`` targets, offset by one.
        device: Device to move batches to.
        seed: Seed for the offset sampler, so a run is reproducible.

    Raises:
        FileNotFoundError: If ``bin_path`` does not exist.
        ValueError: If the file holds fewer than ``block_size + 1`` tokens.
    """

    def __init__(self, bin_path: str | Path, block_size: int,
                 device: str = "cpu", seed: int = 1337):
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run the Phase 1 encoding step "
                f"(scripts/build_phase1_report.py) or restore the tokenized "
                f"archive from Drive."
            )
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)

        # mode="r" is a read-only mapping; the array is never modified, so
        # several runs can share the same pages.
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self.data.shape[0])
        if self.n_tokens < block_size + 1:
            raise ValueError(
                f"{self.path.name} holds {self.n_tokens:,} tokens, fewer than "
                f"block_size + 1 = {block_size + 1}"
            )
        logger.info("%s: %s tokens, %s windows available",
                    self.path.name, f"{self.n_tokens:,}",
                    f"{self.n_tokens - block_size:,}")

    def __len__(self) -> int:
        """Number of distinct start offsets."""
        return self.n_tokens - self.block_size

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of (inputs, targets).

        Targets are the inputs shifted left by one, which is the causal
        language-modelling objective: the logit at position t is trained to
        predict the token at t+1.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            ``(x, y)``, both int64 tensors of shape (batch_size, block_size).
        """
        ix = self.rng.integers(0, len(self), size=batch_size)

        # astype(np.int64) copies out of the memmap; without the copy the
        # tensors would alias pages that numpy may later evict.
        x = np.stack([self.data[i:i + self.block_size].astype(np.int64)
                      for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size].astype(np.int64)
                      for i in ix])

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.device.startswith("cuda"):
            # pin_memory + non_blocking overlaps the host-to-device copy with
            # compute on the previous batch.
            x_t = x_t.pin_memory().to(self.device, non_blocking=True)
            y_t = y_t.pin_memory().to(self.device, non_blocking=True)
        else:
            x_t, y_t = x_t.to(self.device), y_t.to(self.device)

        return x_t, y_t

    def sequential_batches(self, batch_size: int, max_batches: int | None = None):
        """Yield non-overlapping windows in order, for deterministic evaluation.

        Random sampling is right for training but wrong for reporting a
        validation number: two runs would evaluate on different windows and the
        losses would not be comparable. This walks the split start to end with
        no overlap, so the metric is a property of the data rather than of the
        draw.

        Args:
            batch_size: Sequences per batch.
            max_batches: Stop after this many batches. None walks the whole split.

        Yields:
            ``(x, y)`` tensors of shape (batch_size, block_size).
        """
        stride = self.block_size
        starts = np.arange(0, len(self) - stride, stride)
        n_batches = len(starts) // batch_size
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)

        for b in range(n_batches):
            ix = starts[b * batch_size:(b + 1) * batch_size]
            x = np.stack([self.data[i:i + self.block_size].astype(np.int64)
                          for i in ix])
            y = np.stack([self.data[i + 1:i + 1 + self.block_size].astype(np.int64)
                          for i in ix])
            x_t = torch.from_numpy(x).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)
            yield x_t, y_t

    def total_utf8_bytes(self, tokenizer) -> int:
        """Decode the whole split and count UTF-8 bytes, for bits-per-byte.

        BPB needs the byte length of the *text*, not of the token array. This
        is only used at evaluation time on the small held-out splits.

        Args:
            tokenizer: A loaded SentencePieceProcessor.

        Returns:
            Total UTF-8 byte count of the decoded split.
        """
        # Decode in chunks. Decoding the whole split in one call builds a
        # multi-million-element Python list and a single enormous string --
        # roughly 5.8GB of text for the Telugu train split -- which is a
        # needless way to OOM a Colab instance during evaluation.
        total, CHUNK = 0, 1_000_000
        for start in range(0, self.n_tokens, CHUNK):
            ids = self.data[start:start + CHUNK].astype(np.int64).tolist()
            total += len(tokenizer.decode(ids).encode("utf-8"))
        return total

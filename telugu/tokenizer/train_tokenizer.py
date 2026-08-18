"""Train the Telugu tokenizer from scratch (SentencePiece BPE).

    python -m telugu.tokenizer.train_tokenizer

Sweeps the vocabulary sizes in ``telugu/configs/telugu.yaml``, evaluates each on
held-out text, selects one, and writes the chosen .model/.vocab into this
directory for committing. No pretrained tokenizer is used, and nothing is
shared with the other language.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_tokenizer import run

if __name__ == "__main__":
    run(
        language="telugu",
        config_path=Path("telugu/configs/telugu.yaml"),
        data_root=Path("data/telugu"),
        tokenizer_dir=Path("telugu/tokenizer"),
    )

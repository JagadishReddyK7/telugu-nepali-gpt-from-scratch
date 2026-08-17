"""Clean the Telugu corpus: filter, deduplicate, and split.

Run on Colab after both the crawl and the corpus download have finished:

    python -m telugu.clean.run_clean

Reads every shard in ``data/telugu/raw/``, applies the filters configured in
``telugu/configs/telugu.yaml``, removes exact and near duplicates, and writes
``data/telugu/clean/{train,validation,test}.jsonl`` plus ``clean_stats.json``.

Nothing is shared with the other language: separate config, separate input
directory, separate deduplication index, separate output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.run_clean import run

if __name__ == "__main__":
    run(
        language="telugu",
        config_path=Path("telugu/configs/telugu.yaml"),
        data_root=Path("data/telugu"),
    )

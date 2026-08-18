"""
Filter, deduplicate, and split a raw corpus into train/validation/test.

This is the stage that turns ~1M heterogeneous documents into a defensible
training corpus, and it is where the project's hard constraints are actually
enforced rather than merely intended.

THE FIVE STAGES, AND WHY EACH EXISTS
------------------------------------
1. *Normalization.* Unicode NFC plus whitespace and invisible-character cleanup.
   Scraped documents have not been normalized yet; downloaded ones already were
   at ingest. Normalizing twice is harmless because the transform is idempotent.

2. *Script filtering.* Rejects documents whose letters are mostly not in the
   target script. Cheap, and for Telugu it catches nearly all contamination
   because the script is unique to the language.

3. *Language identification.* THE critical filter for Nepali. Devanagari is
   shared with Hindi, so script filtering passes Hindi documents unchanged. If
   Hindi survives into the corpus, Model L is covertly bilingual -- a direct
   violation of the spec's "do not train a multilingual model" constraint, and
   a confound that would corrupt the Phase 3 resource-tier analysis. The
   rejection counts are reported, not hidden.

4. *Quality heuristics.* Mean word length, repeated-line fraction, short-line
   fraction. These catch extraction failures that produce well-formed but
   useless text: navigation menus, photo galleries, markup stripped without
   spaces.

5. *Deduplication.* Two kinds, because they catch different things:
   - EXACT (64-bit hash of normalized text): the same article syndicated
     verbatim across outlets, or the same document appearing in two corpora.
   - NEAR (MinHash + LSH): the same story with a different headline, a
     boilerplate footer swapped, a date changed. Duplicates inflate a corpus
     without adding information, and worse, near-duplicates that straddle the
     train/test boundary leak test data into training.

SPLITTING IS DOCUMENT-LEVEL AND DEDUPLICATION COMES FIRST
---------------------------------------------------------
Both matter. Splitting by line would put sentences from one article in both
train and test. And deduplicating *after* splitting would leave a document in
train and its near-duplicate in test, which inflates validation performance for
no real reason. Dedup, then split.

Split assignment is by hash of the document id rather than a random shuffle, so
it is deterministic, requires no second pass, and stays stable if the corpus
grows.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from common.text_norm import NormalizerConfig, TextNormalizer, count_bytes

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


@dataclass
class CleanConfig:
    """Thresholds controlling the cleaning pass.

    Values come from the per-language YAML so Telugu and Nepali can differ --
    notably ``min_langid_confidence``, which is deliberately stricter for
    Nepali. See the config files for the reasoning.
    """

    script: str
    iso_code: str
    min_script_ratio: float = 0.80
    min_langid_confidence: float = 0.60
    min_chars: int = 200
    max_chars: int = 50_000
    min_mean_word_len: float = 2.0
    max_mean_word_len: float = 30.0
    max_dup_line_fraction: float = 0.30
    max_short_line_fraction: float = 0.80
    minhash_threshold: float = 0.80
    minhash_num_perm: int = 128
    shingle_size: int = 5
    train: float = 0.98
    validation: float = 0.01
    test: float = 0.01
    seed: int = 1337
    max_val_docs: int = 20_000
    max_test_docs: int = 20_000
    use_langid: bool = True
    # Cap downloaded text at this multiple of the manually collected text.
    # 4.0 means downloaded <= 4x manual, i.e. manual >= 20% of the corpus --
    # exactly the spec's requirement, enforced mechanically instead of hoped for.
    #
    # This matters because the two shares are collected independently: the
    # download hits a fixed character budget while the crawl yields whatever
    # the sites allow. For Nepali the crawl came up short, so using every
    # downloaded document would put the manual share at 7%. Discarding surplus
    # downloaded text is the correct remedy -- a smaller corpus that satisfies
    # the constraint beats a larger one that violates it.
    max_downloaded_to_manual_ratio: float | None = 4.0


@dataclass
class CleanStats:
    """Counters for the cleaning funnel, reported in the Phase 1 statistics."""

    read: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    chars_kept: int = 0
    bytes_kept: int = 0
    per_source: Counter = field(default_factory=Counter)
    per_provenance: Counter = field(default_factory=Counter)
    chars_per_provenance: Counter = field(default_factory=Counter)
    split_docs: Counter = field(default_factory=Counter)
    split_chars: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "documents_read": self.read,
            "documents_kept": self.kept,
            "keep_rate": round(self.kept / self.read, 4) if self.read else 0.0,
            "rejected_by_reason": dict(self.rejected.most_common()),
            "chars_kept": self.chars_kept,
            "bytes_kept": self.bytes_kept,
            "documents_per_source": dict(self.per_source.most_common()),
            "documents_per_provenance": dict(self.per_provenance),
            "chars_per_provenance": dict(self.chars_per_provenance),
            "manual_char_fraction": round(
                self.chars_per_provenance.get("manual_scrape", 0)
                / max(self.chars_kept, 1), 4),
            "split_documents": dict(self.split_docs),
            "split_chars": dict(self.split_chars),
            "seconds": round(self.seconds, 1),
        }


def _quality_metrics(text: str) -> dict:
    """Compute cheap structural quality signals for one document."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    words = _WORD_RE.findall(text)
    if not lines or not words:
        return {"mean_word_len": 0.0, "dup_line_frac": 0.0,
                "short_line_frac": 1.0}
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "mean_word_len": sum(len(w) for w in words) / len(words),
        "dup_line_frac": dup / len(lines),
        "short_line_frac": sum(1 for ln in lines if len(ln) < 30) / len(lines),
    }


def _shingles(text: str, k: int) -> set[bytes]:
    """Return the set of k-word shingles, as hashed bytes.

    Word-level shingles rather than character-level: they are far cheaper to
    compute and more robust to the small punctuation differences that distinguish
    two copies of the same wire story.
    """
    words = _WORD_RE.findall(text)
    if len(words) < k:
        return {" ".join(words).encode("utf-8")} if words else set()
    return {
        " ".join(words[i:i + k]).encode("utf-8")
        for i in range(len(words) - k + 1)
    }


class CorpusCleaner:
    """Run the full cleaning and splitting pass for one language.

    Args:
        language: Language name, used for paths and logging.
        config: Thresholds for this language.
        data_root: Directory containing ``raw/``; ``clean/`` is written beside it.
    """

    def __init__(self, language: str, config: CleanConfig, data_root: Path):
        self.language = language
        self.config = config
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.clean_dir = self.data_root / "clean"
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = TextNormalizer(NormalizerConfig(script=config.script))
        self._lang_filter = None

    # ------------------------------------------------------------- language

    def _get_language_filter(self):
        """Lazily construct the fastText language filter."""
        if self._lang_filter is None and self.config.use_langid:
            from common.langid import LanguageFilter
            self._lang_filter = LanguageFilter(
                target_lang=self.config.iso_code,
                min_confidence=self.config.min_langid_confidence,
            )
        return self._lang_filter

    # ---------------------------------------------------------------- split

    def _assign_split(self, doc_id: str) -> str:
        """Deterministically assign a document to train, validation, or test.

        Hashing the id (salted with the language's seed) gives a stable,
        uniform assignment without shuffling the corpus or holding it in memory.

        Args:
            doc_id: Stable document identifier.

        Returns:
            One of ``"train"``, ``"validation"``, ``"test"``.
        """
        h = hashlib.md5(f"{self.config.seed}:{doc_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        if bucket < self.config.validation:
            return "validation"
        if bucket < self.config.validation + self.config.test:
            return "test"
        return "train"

    # ---------------------------------------------------------------- clean

    def run(self, limit: int | None = None) -> CleanStats:
        """Stream every raw shard, filter, deduplicate, and write the splits.

        Args:
            limit: Stop after this many input documents. For quick trials.

        Returns:
            A :class:`CleanStats` describing the whole funnel.
        """
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "datasketch is required for near-duplicate detection. "
                "Install with: pip install datasketch"
            ) from exc

        cfg = self.config
        stats = CleanStats()
        start = time.time()

        # datasketch derives its band count from (threshold, num_perm) and
        # raises if that yields fewer than 2 bands -- which a very high
        # threshold does. Clamp rather than crash a multi-hour job on a config
        # typo.
        threshold = min(max(cfg.minhash_threshold, 0.30), 0.95)
        if threshold != cfg.minhash_threshold:
            logger.warning("minhash_threshold %.2f out of usable range; using %.2f",
                           cfg.minhash_threshold, threshold)
        lsh = MinHashLSH(threshold=threshold, num_perm=cfg.minhash_num_perm)
        seen_exact: set[int] = set()
        lang_filter = self._get_language_filter()

        writers = {
            name: (self.clean_dir / f"{name}.jsonl").open("w", encoding="utf-8")
            for name in ("train", "validation", "test")
        }
        # Held-out sets are capped so Phase 2 evaluation stays fast on a T4;
        # overflow goes to train rather than being discarded.
        held_out_counts = Counter()

        # Classify each shard by provenance from its first record, then process
        # MANUAL shards first. We cannot know the downloaded budget until the
        # manual total is known, and manual text is never discarded, so this
        # ordering lets the whole job run in a single pass over the data.
        manual_shards, downloaded_shards = [], []
        for path in sorted(glob.glob(str(self.raw_dir / "*.jsonl"))):
            prov = "unknown"
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            prov = json.loads(line).get("provenance", "unknown")
                            break
            except (OSError, json.JSONDecodeError):
                pass
            (manual_shards if prov == "manual_scrape" else downloaded_shards).append(path)

        shards = manual_shards + downloaded_shards
        logger.info("[%s] cleaning %d shard(s): %d manual, %d downloaded",
                    self.language, len(shards), len(manual_shards),
                    len(downloaded_shards))
        downloaded_budget: float | None = None
        downloaded_chars = 0

        try:
            for shard in shards:
                # Entering the downloaded shards: manual total is now final, so
                # the budget can be computed.
                if (shard == (downloaded_shards[0] if downloaded_shards else None)
                        and cfg.max_downloaded_to_manual_ratio):
                    manual_chars = stats.chars_per_provenance.get("manual_scrape", 0)
                    downloaded_budget = manual_chars * cfg.max_downloaded_to_manual_ratio
                    logger.info(
                        "[%s] manual share final: %s chars. Capping downloaded at "
                        "%s chars (%.1fx) to hold manual at >=%.0f%% of the corpus.",
                        self.language, f"{manual_chars:,}",
                        f"{downloaded_budget:,.0f}",
                        cfg.max_downloaded_to_manual_ratio,
                        100 / (1 + cfg.max_downloaded_to_manual_ratio),
                    )
                    if manual_chars == 0:
                        logger.error("[%s] NO manual documents survived cleaning. "
                                     "The >=20%% requirement cannot be met.",
                                     self.language)
                for line in open(shard, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        stats.rejected["malformed_json"] += 1
                        continue

                    stats.read += 1
                    if limit and stats.read > limit:
                        raise StopIteration

                    if (downloaded_budget is not None
                            and downloaded_chars >= downloaded_budget):
                        stats.rejected["downloaded_over_ratio_budget"] += 1
                        continue

                    if stats.read % 100_000 == 0:
                        logger.info("  [%s] read %s kept %s (%.1f%%)",
                                    self.language, f"{stats.read:,}",
                                    f"{stats.kept:,}",
                                    stats.kept / stats.read * 100)

                    text = self.normalizer.normalize(doc.get("text") or "")

                    # --- stage 1: length -------------------------------------
                    if not text:
                        stats.rejected["empty_after_normalization"] += 1
                        continue
                    if len(text) < cfg.min_chars:
                        stats.rejected["too_short"] += 1
                        continue
                    if len(text) > cfg.max_chars:
                        stats.rejected["too_long_listing_page"] += 1
                        continue

                    # --- stage 2: script -------------------------------------
                    if self.normalizer.script_ratio(text) < cfg.min_script_ratio:
                        stats.rejected["wrong_script"] += 1
                        continue

                    # --- stage 3: quality ------------------------------------
                    q = _quality_metrics(text)
                    if not (cfg.min_mean_word_len <= q["mean_word_len"]
                            <= cfg.max_mean_word_len):
                        stats.rejected["bad_mean_word_length"] += 1
                        continue
                    if q["dup_line_frac"] > cfg.max_dup_line_fraction:
                        stats.rejected["repeated_lines_boilerplate"] += 1
                        continue
                    if q["short_line_frac"] > cfg.max_short_line_fraction:
                        stats.rejected["mostly_short_lines"] += 1
                        continue

                    # --- stage 4: exact duplicate ----------------------------
                    # 64 bits of BLAKE2b: collision probability across 1M
                    # documents is ~3e-8, which is negligible, and it costs an
                    # eighth of the memory of storing full digests.
                    h = int.from_bytes(
                        hashlib.blake2b(text.encode("utf-8"),
                                        digest_size=8).digest(), "big")
                    if h in seen_exact:
                        stats.rejected["exact_duplicate"] += 1
                        continue
                    seen_exact.add(h)

                    # --- stage 5: language identification --------------------
                    # Runs late because it is the most expensive filter; the
                    # cheap checks above have already removed most junk.
                    if lang_filter is not None:
                        result = lang_filter.identify(text)
                        if not result.accepted:
                            stats.rejected[f"langid_{result.reason}"] += 1
                            continue

                    # --- stage 6: near duplicate -----------------------------
                    mh = MinHash(num_perm=cfg.minhash_num_perm)
                    for sh in _shingles(text, cfg.shingle_size):
                        mh.update(sh)
                    if lsh.query(mh):
                        stats.rejected["near_duplicate"] += 1
                        continue
                    # Integer keys keep the LSH index compact at corpus scale.
                    lsh.insert(stats.kept, mh)

                    # --- accepted --------------------------------------------
                    doc_id = doc.get("id") or f"{stats.read}"
                    split = self._assign_split(doc_id)
                    if split == "validation" and held_out_counts["validation"] >= cfg.max_val_docs:
                        split = "train"
                    elif split == "test" and held_out_counts["test"] >= cfg.max_test_docs:
                        split = "train"
                    if split != "train":
                        held_out_counts[split] += 1

                    provenance = doc.get("provenance", "unknown")
                    source = doc.get("source", "unknown")
                    n_bytes = count_bytes(text)

                    writers[split].write(json.dumps({
                        "id": doc_id,
                        "source": source,
                        "provenance": provenance,
                        "text": text,
                    }, ensure_ascii=False) + "\n")

                    stats.kept += 1
                    stats.chars_kept += len(text)
                    stats.bytes_kept += n_bytes
                    stats.per_source[source] += 1
                    stats.per_provenance[provenance] += 1
                    stats.chars_per_provenance[provenance] += len(text)
                    if provenance != "manual_scrape":
                        downloaded_chars += len(text)
                    stats.split_docs[split] += 1
                    stats.split_chars[split] += len(text)

        except StopIteration:
            logger.info("[%s] stopped at limit=%s", self.language, limit)
        except KeyboardInterrupt:
            logger.info("[%s] interrupted", self.language)
        finally:
            for w in writers.values():
                w.close()
            stats.seconds = time.time() - start

        out = self.clean_dir / "clean_stats.json"
        out.write_text(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        logger.info("[%s] wrote %s", self.language, out)
        return stats

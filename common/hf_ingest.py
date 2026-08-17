"""
Stream public text corpora into normalized JSONL shards.

This produces the "downloaded" share of each corpus -- roughly 80% of the
tokens, complementing the manually scraped news. It is the counterpart to
``common/scraper.py``, and writes into the same shard format so the cleaning
stage cannot tell the two apart except by the ``provenance`` field.

WHY STREAMING
-------------
IndicCorpV2's Telugu split and IRIISNEPAL's Nepali corpus are tens of
gigabytes. Downloading them in full would exhaust Colab's disk before we ever
got to filtering, and we only need a fraction of each. HuggingFace's
``streaming=True`` mode fetches shards lazily over HTTP and yields documents one
at a time, so peak disk usage is bounded by what we choose to keep rather than
by the dataset's size.

The tradeoff is that streaming datasets are sequential-only -- you cannot seek.
Resumption therefore works by counting documents and skipping, which re-reads
(but does not re-store) the prefix. That is acceptable because the bottleneck is
disk, not bandwidth.

WHAT COUNTS AS "DOWNLOADED"
---------------------------
Everything here. The project spec requires at least 20% of final training tokens
to come from *manual* collection, so every record written by this module is
tagged ``provenance="downloaded"`` and every record written by the scraper is
tagged ``provenance="manual_scrape"``. The dataset statistics step computes the
split from those tags rather than trusting an assertion.

A NOTE ON THE HARD CONSTRAINTS
------------------------------
``datasets`` is used ONLY to stream raw text. No pretrained model, tokenizer, or
HuggingFace Transformer class is loaded anywhere in this project. Wikipedia
arrives through ``wikimedia/wikipedia``, which is a text dump, not a model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.text_norm import NormalizerConfig, TextNormalizer

logger = logging.getLogger(__name__)


@dataclass
class SourceSpec:
    """One downloadable corpus.

    Attributes:
        name: Short slug; becomes the ``source`` field and the shard prefix.
        hf_dataset: HuggingFace dataset repository id.
        hf_config: Dataset configuration name, or None.
        split: Which split to stream. Almost always "train".
        text_field: Field holding the document text.
        data_dir: Optional subdirectory within the dataset repo.
        data_files: Explicit file path(s) within the repo. Use this when the
            language is encoded in the FILENAME rather than a directory --
            IndicCorpV2 stores each language as ``data/<iso>.txt``, so asking
            for a directory would select every language at once and silently
            build a multilingual corpus.
        load_kwargs: Extra keyword arguments forwarded to ``load_dataset``.
            For plain-text repos, ``{"sample_by": "paragraph"}`` makes the text
            builder emit blank-line-separated paragraphs instead of one row per
            LINE. Without it, documents arrive shredded into single lines and
            most get rejected by the minimum-length filter.
        target_chars: Stop after collecting this many characters from this
            source. Budgeting per source rather than globally is deliberate --
            it stops one enormous corpus from crowding out the others and
            flattening the register diversity we are trying to build.
        min_chars: Reject documents shorter than this.
        max_chars: Reject documents longer than this (see the listing-page
            reasoning in ``common/scraper.py``).
        notes: Free text quoted in the dataset statistics report.
    """

    name: str
    hf_dataset: str
    hf_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    data_dir: str | None = None
    data_files: str | list[str] | None = None
    load_kwargs: dict = field(default_factory=dict)
    target_chars: int = 200_000_000
    min_chars: int = 200
    max_chars: int = 50_000
    notes: str = ""


def resolve_dataset_location(
    repo_id: str,
    config_hint: str | None = None,
    dir_hint: str | None = None,
) -> dict:
    """Work out the right ``load_dataset`` arguments for a repository.

    Dataset repositories organize languages inconsistently. Some expose one
    builder config per language; others expose a single config and separate
    languages by directory. Guessing wrong fails with a confusing error --
    IndicCorpV2 advertises per-language *subsets* in its dataset viewer but
    exposes exactly one builder config named ``indiccorp_v2``, so asking for
    config ``tel_Telu`` fails with "BuilderConfig not found".

    This asks the Hub what actually exists and adapts, so the code does not
    depend on a layout that may change.

    Args:
        repo_id: HuggingFace dataset repository id.
        config_hint: Preferred builder config name, if any.
        dir_hint: Substring identifying the language directory, e.g. "tel_Telu".

    Returns:
        Keyword arguments to splat into ``load_dataset``.
    """
    kwargs: dict = {}
    configs: list[str] = []
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(repo_id)
    except Exception as exc:
        logger.warning("%s: could not list configs (%s); using defaults",
                       repo_id, type(exc).__name__)

    if config_hint and config_hint in configs:
        kwargs["name"] = config_hint
    elif configs:
        # Exactly one config means languages are separated some other way --
        # almost always by directory.
        kwargs["name"] = configs[0]
        if config_hint:
            logger.info("%s: config %r unavailable; using %r and selecting the "
                        "language by directory instead",
                        repo_id, config_hint, configs[0])

    if dir_hint and "name" in kwargs and kwargs.get("name") != config_hint:
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id, repo_type="dataset")
            matching = [
                f for f in files
                if dir_hint in f and not f.endswith((".md", ".json", ".py",
                                                     ".gitattributes"))
            ]
            if not matching:
                logger.error(
                    "%s: NO files matched language hint %r. This source will "
                    "yield nothing. Sample of repo contents: %s",
                    repo_id, dir_hint, files[:8],
                )
                return kwargs

            # Prefer a directory. Fall back to explicit data_files when the
            # language is encoded in the FILENAME rather than a folder --
            # e.g. `tel_Telu.parquet` at the repo root, where taking the
            # directory prefix would yield "" and silently select everything.
            dirs = sorted({
                "/".join(f.split("/")[:-1]) for f in matching if "/" in f
            })
            if dirs:
                chosen = min(dirs, key=lambda d: d.count("/"))
                kwargs["data_dir"] = chosen
                logger.info("%s: selecting language via data_dir=%r (%d files)",
                            repo_id, chosen, len(matching))
            else:
                kwargs["data_files"] = matching
                logger.info("%s: language is in the filenames, not a folder; "
                            "selecting %d file(s) explicitly",
                            repo_id, len(matching))
        except Exception as exc:
            logger.warning("%s: could not list repo files (%s: %s)",
                           repo_id, type(exc).__name__, exc)
    return kwargs


def detect_text_field(row: dict, configured: str) -> str | None:
    """Find the field holding document text, falling back if the guess is wrong.

    A wrong field name is the most dangerous failure mode in this pipeline,
    because it fails *silently*: ``row.get("text")`` returns None for every
    document, everything normalizes to the empty string, and the source reports
    "read 650,000, kept 0" without ever saying why. That is exactly what
    IRIISNEPAL/Nepali-Text-Corpus did -- its text lives under a different key.

    Args:
        row: One dataset row.
        configured: The field name we expected.

    Returns:
        A usable field name, or None if the row has no substantial string field.
    """
    value = row.get(configured)
    if isinstance(value, str) and value.strip():
        return configured

    # Prefer conventional names, then fall back to the longest string field.
    for candidate in ("text", "content", "article", "body", "raw_content",
                      "Article", "Content", "sentence", "document"):
        v = row.get(candidate)
        if isinstance(v, str) and len(v.strip()) > 50:
            return candidate

    strings = [(k, v) for k, v in row.items()
               if isinstance(v, str) and len(v.strip()) > 50]
    if strings:
        return max(strings, key=lambda kv: len(kv[1]))[0]
    return None


def resolve_wikipedia_config(lang_code: str) -> str | None:
    """Find the newest ``wikimedia/wikipedia`` config for a language.

    Wikipedia dumps are published as dated snapshots, so the config name looks
    like ``20231101.te``. Hardcoding a date rots: snapshots are added and old
    ones eventually disappear. This asks the Hub which configs actually exist
    and picks the most recent one for the requested language.

    Args:
        lang_code: Wikipedia language code, e.g. "te" or "ne".

    Returns:
        The newest matching config name, or None if the language is absent or
        the Hub is unreachable.
    """
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("wikimedia/wikipedia")
    except Exception as exc:
        logger.error("Could not list Wikipedia configs: %s: %s",
                     type(exc).__name__, exc)
        return None

    matches = [c for c in configs if c.endswith(f".{lang_code}")]
    if not matches:
        logger.error("No Wikipedia snapshot found for language %r", lang_code)
        return None
    # Config names are `YYYYMMDD.lang`, so lexical sort is chronological.
    newest = sorted(matches)[-1]
    logger.info("Wikipedia %r: using snapshot %s (of %d available)",
                lang_code, newest, len(matches))
    return newest


@dataclass
class IngestStats:
    """Counters describing one source's ingestion run."""

    read: int = 0
    kept: int = 0
    too_short: int = 0
    too_long: int = 0
    wrong_script: int = 0
    empty_after_norm: int = 0
    chars_kept: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        """Return the counters as a plain dictionary."""
        return {
            "read": self.read, "kept": self.kept,
            "too_short": self.too_short, "too_long": self.too_long,
            "wrong_script": self.wrong_script,
            "empty_after_norm": self.empty_after_norm,
            "chars_kept": self.chars_kept,
            "seconds": round(self.seconds, 1),
        }


class ShardWriter:
    """Append records to rolling JSONL shards.

    Mirrors the scraper's writer so both provenance classes produce
    byte-compatible output. Opens lazily so a no-op run creates no files.
    """

    def __init__(self, output_dir: Path, prefix: str, docs_per_shard: int = 50_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self._count = 0
        self._shard_idx = len(list(self.output_dir.glob(f"{prefix}-*.jsonl")))
        self._fh = None

    def _open(self) -> None:
        """Open the next shard file."""
        if self._fh is not None:
            self._fh.close()
        path = self.output_dir / f"{self.prefix}-{self._shard_idx:05d}.jsonl"
        self._fh = path.open("a", encoding="utf-8")
        logger.info("  writing shard %s", path.name)

    def write(self, record: dict) -> None:
        """Write one record as a JSON line."""
        if self._fh is None:
            self._open()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1
        if self._count % 1000 == 0:
            self._fh.flush()
        if self._count % self.docs_per_shard == 0:
            self._shard_idx += 1
            self._open()

    def close(self) -> None:
        """Flush and close."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class CorpusIngestor:
    """Stream, normalize, filter, and shard one language's downloaded sources.

    Args:
        language: Language name, e.g. "telugu".
        script: Script key for :class:`TextNormalizer`.
        output_root: Directory containing ``raw/`` and the progress file.
    """

    def __init__(self, language: str, script: str, output_root: Path) -> None:
        self.language = language
        self.output_root = Path(output_root)
        self.raw_dir = self.output_root / "raw"
        self.progress_path = self.output_root / "download_progress.json"
        self.normalizer = TextNormalizer(NormalizerConfig(script=script))
        # Script-ratio floor applied at ingest. Kept loose here (0.5) because
        # the cleaning stage applies the real threshold with language ID; this
        # is only to discard obvious garbage before it costs us disk.
        self.min_script_ratio = 0.5

    # ------------------------------------------------------------- progress

    def _load_progress(self) -> dict:
        """Read the resume state, or an empty dict if none exists."""
        if self.progress_path.exists():
            try:
                return json.loads(self.progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("download_progress.json corrupt; starting fresh")
        return {}

    def _save_progress(self, progress: dict) -> None:
        """Persist the resume state atomically."""
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        tmp.replace(self.progress_path)

    # -------------------------------------------------------------- ingest

    def ingest(self, spec: SourceSpec, resume: bool = True) -> IngestStats:
        """Stream one source until its character budget is met.

        Args:
            spec: The source to ingest.
            resume: If True, skip documents already consumed in a prior run.

        Returns:
            An :class:`IngestStats` for this source.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `datasets` library is required. Install with: "
                "pip install datasets\n"
                "It is used ONLY to stream raw text -- no pretrained models."
            ) from exc

        progress = self._load_progress()
        state = progress.get(spec.name, {"docs_read": 0, "chars_kept": 0})

        if resume and state["chars_kept"] >= spec.target_chars:
            logger.info("[%s] already at target (%s chars); skipping",
                        spec.name, f"{state['chars_kept']:,}")
            return IngestStats(chars_kept=state["chars_kept"])

        logger.info("[%s] streaming %s%s -> target %s chars",
                    spec.name, spec.hf_dataset,
                    f" ({spec.hf_config})" if spec.hf_config else "",
                    f"{spec.target_chars:,}")

        kwargs = {"split": spec.split, "streaming": True}
        if spec.data_files:
            # Explicit paths win: we know exactly which file holds this
            # language, so there is nothing to discover and no way to
            # accidentally select a different language.
            kwargs["data_files"] = spec.data_files
            logger.info("[%s] using explicit data_files=%r",
                        spec.name, spec.data_files)
        else:
            # Otherwise ask the Hub how this repository is laid out rather
            # than assuming. See resolve_dataset_location.
            kwargs.update(resolve_dataset_location(
                spec.hf_dataset,
                config_hint=spec.hf_config,
                dir_hint=spec.data_dir,
            ))
            if spec.data_dir and "data_dir" not in kwargs:
                kwargs["data_dir"] = spec.data_dir
        kwargs.update(spec.load_kwargs)

        try:
            ds = load_dataset(spec.hf_dataset, **kwargs)
        except TypeError as exc:
            # The installed `datasets` version does not accept one of the
            # builder kwargs (they have been renamed across releases). Drop the
            # optional ones and retry rather than failing the whole source.
            optional = set(spec.load_kwargs) - {"sample_by"}
            if optional:
                logger.warning("[%s] loader rejected %s (%s); retrying without",
                               spec.name, sorted(optional), exc)
                for k in optional:
                    kwargs.pop(k, None)
                try:
                    ds = load_dataset(spec.hf_dataset, **kwargs)
                except Exception as exc2:
                    logger.error("[%s] could not open dataset: %s: %s",
                                 spec.name, type(exc2).__name__, exc2)
                    return IngestStats()
            else:
                logger.error("[%s] could not open dataset: %s: %s",
                             spec.name, type(exc).__name__, exc)
                return IngestStats()
        except Exception as exc:
            logger.error("[%s] could not open dataset: %s: %s",
                         spec.name, type(exc).__name__, exc)
            return IngestStats()

        skip_n = state["docs_read"] if resume else 0
        if skip_n:
            logger.info("[%s] resuming: skipping %s documents already read",
                        spec.name, f"{skip_n:,}")
            ds = ds.skip(skip_n)

        writer = ShardWriter(self.raw_dir, spec.name)
        stats = IngestStats(chars_kept=state["chars_kept"] if resume else 0)
        start = time.time()
        docs_read = skip_n

        text_field = spec.text_field
        field_confirmed = False

        try:
            for row in ds:
                docs_read += 1
                stats.read += 1

                # Resolve the text field from the first row, once.
                if not field_confirmed:
                    detected = detect_text_field(row, spec.text_field)
                    if detected is None:
                        logger.error(
                            "[%s] ABORTING: no usable text field. Row keys are %s. "
                            "Set text_field on this SourceSpec to the right one.",
                            spec.name, sorted(row.keys()),
                        )
                        break
                    if detected != spec.text_field:
                        logger.warning(
                            "[%s] configured text_field=%r is absent or empty; "
                            "using %r instead. Row keys: %s",
                            spec.name, spec.text_field, detected,
                            sorted(row.keys()),
                        )
                    text_field = detected
                    field_confirmed = True

                text = row.get(text_field) or ""
                text = self.normalizer.normalize(text)
                if not text:
                    stats.empty_after_norm += 1
                elif len(text) < spec.min_chars:
                    stats.too_short += 1
                elif len(text) > spec.max_chars:
                    stats.too_long += 1
                elif self.normalizer.script_ratio(text) < self.min_script_ratio:
                    stats.wrong_script += 1
                else:
                    writer.write({
                        "id": f"{spec.name}-{docs_read}",
                        "url": None,
                        "source": spec.name,
                        "provenance": "downloaded",
                        "text": text,
                        "n_chars": len(text),
                        "fetched_at": time.time(),
                    })
                    stats.kept += 1
                    stats.chars_kept += len(text)

                if stats.read % 50_000 == 0:
                    pct = stats.chars_kept / spec.target_chars * 100
                    logger.info("  [%s] read %s kept %s | %s chars (%.1f%% of target)",
                                spec.name, f"{stats.read:,}", f"{stats.kept:,}",
                                f"{stats.chars_kept:,}", pct)
                    state.update(docs_read=docs_read, chars_kept=stats.chars_kept)
                    progress[spec.name] = state
                    self._save_progress(progress)

                # FAIL FAST ON A ZERO-YIELD SOURCE.
                # Reading hundreds of thousands of documents and keeping none
                # means something is structurally wrong -- wrong field, wrong
                # language directory, wrong script -- not that the data is
                # merely low quality. Without this guard the run burns an hour
                # and reports success with an empty corpus.
                if stats.read >= 20_000 and stats.kept == 0:
                    logger.error(
                        "[%s] ABORTING: read %s documents and kept NONE.\n"
                        "  rejected as: too_short=%d too_long=%d wrong_script=%d "
                        "empty_after_norm=%d\n"
                        "  text field in use: %r\n"
                        "  This usually means the wrong language subset or the "
                        "wrong text field. Check the counters above: a high "
                        "wrong_script count means the language directory is "
                        "wrong; a high empty_after_norm count means the text "
                        "field is wrong.",
                        spec.name, f"{stats.read:,}", stats.too_short,
                        stats.too_long, stats.wrong_script,
                        stats.empty_after_norm, text_field,
                    )
                    break

                if stats.chars_kept >= spec.target_chars:
                    logger.info("  [%s] reached character target", spec.name)
                    break

        except KeyboardInterrupt:
            logger.info("[%s] interrupted; progress saved", spec.name)
        except Exception as exc:
            # One bad source must not kill a multi-source download.
            logger.exception("[%s] failed with %s -- moving on",
                             spec.name, type(exc).__name__)
        finally:
            writer.close()
            stats.seconds = time.time() - start
            state.update(docs_read=docs_read, chars_kept=stats.chars_kept)
            progress[spec.name] = state
            self._save_progress(progress)

        logger.info("[%s] done: %s", spec.name, stats.as_dict())
        return stats

    def ingest_all(self, specs: list[SourceSpec], resume: bool = True) -> dict:
        """Ingest every source in turn.

        Args:
            specs: Sources to ingest, in priority order.
            resume: Skip work already done in a prior run.

        Returns:
            Mapping of source name to its statistics dictionary.
        """
        summary = {}
        for spec in specs:
            summary[spec.name] = self.ingest(spec, resume=resume).as_dict()
        out = self.output_root / "download_summary.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        logger.info("Wrote %s", out)
        return summary

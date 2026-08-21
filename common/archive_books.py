"""
Collect book-length text from the Internet Archive as manual-collection data.

WHY THIS IS THE RIGHT LEVER FOR A BLOCKED LANGUAGE
--------------------------------------------------
The project spec names "OCR from books/PDFs" as manual collection, alongside
scraping. Internet Archive texts ARE OCR output -- the ``_djvu.txt`` file for
each scanned volume is exactly that -- and they are freely downloadable through
a documented public API.

The arithmetic is what makes this worth doing. A news article averages ~1,750
characters. A scanned book averages 300,000 and often exceeds a million. One
book is worth two hundred articles. So when a news crawl is blocked, a few
hundred books can replace tens of thousands of articles.

That matters here specifically because the manual share gates the whole corpus:
under the >=20% rule, every manually collected character unlocks four downloaded
ones. A book contributing 300k characters therefore adds ~1.5M characters of
final corpus.

WHAT TO EXPECT FROM OCR TEXT
----------------------------
It is noisier than scraped HTML. Scanning artifacts, running heads, page
numbers, and hyphenation survive. Two defences:

  * A script-ratio floor rejects volumes whose OCR failed badly or that turn out
    to be in another language despite the metadata.
  * The main cleaning pipeline's quality heuristics (repeated lines, short-line
    fraction, mean word length) then run over them like any other document.

Books are also long enough to trip the 50,000-character listing-page filter, so
this collector SPLITS each volume into chapter-sized chunks at blank lines.
That is not a workaround -- chunking is what makes them usable as training
documents anyway, since a 1M-character document is useless for a 512-token
context window.

POLITENESS
----------
Archive.org publishes these for programmatic access, but it is still a
donation-funded nonprofit. One request at a time, a real delay between them,
and a contactable User-Agent.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

USER_AGENT = (
    "LMA-Course-Research-Collector/1.0 "
    "(academic monolingual corpus collection; contact: hrushikratan.j@vaayum.com)"
)


@dataclass
class BookCollectorConfig:
    """Settings for one language's book collection.

    Attributes:
        language: Language name, used for output paths and provenance.
        archive_language: Primary language name as archive.org spells it.
        language_aliases: Additional spellings to search. Archive.org metadata is
            contributor-supplied and inconsistent: the same language appears as
            a full name, an ISO 639-2 code, and older exonyms across different
            collections. Searching only one spelling silently misses records --
            for Nepali, "Nepali" alone returned 240 items, and the alternates
            are worth checking precisely because the total is so small.
        script: Script key for the normalizer.
        min_script_ratio: Reject volumes whose text is mostly not this script.
            Catches failed OCR and mislabelled records.
        max_items: Cap on how many volumes to fetch.
        chunk_chars: Target size for each chunk a book is split into.
        min_chunk_chars: Discard chunks shorter than this.
        delay_seconds: Delay between requests.
        target_chars: Stop once this many characters have been collected.
    """

    language: str
    archive_language: str
    script: str
    language_aliases: tuple[str, ...] = ()
    min_script_ratio: float = 0.60
    max_items: int = 2000
    chunk_chars: int = 8000
    min_chunk_chars: int = 500
    delay_seconds: float = 1.5
    target_chars: int = 200_000_000


def _session() -> requests.Session:
    """Build a requests session with an identifying User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def search_identifiers(config: BookCollectorConfig, session) -> list[str]:
    """Find archive.org item identifiers for texts in the target language.

    Args:
        config: Collector configuration.
        session: Requests session.

    Returns:
        A list of item identifiers.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    names = [config.archive_language, *config.language_aliases]

    for name in names:
        query = f'language:("{name}") AND mediatype:(texts)'
        page = 1
        before = len(ordered)
        while len(ordered) < config.max_items:
            params = {"q": query, "fl[]": "identifier", "rows": 500,
                      "page": page, "output": "json"}
            url = f"{SEARCH_URL}?{urllib.parse.urlencode(params, doseq=True)}"
            try:
                data = session.get(url, timeout=60).json()
            except Exception as exc:
                logger.error("search failed (%s, page %d): %s",
                             name, page, type(exc).__name__)
                break

            docs = data.get("response", {}).get("docs", [])
            if page == 1:
                logger.info("  language=%-12s -> %s items reported", name,
                            f"{data.get('response', {}).get('numFound', 0):,}")
            if not docs:
                break
            for d in docs:
                ident = d.get("identifier")
                # Aliases overlap heavily; dedupe so we never fetch twice.
                if ident and ident not in seen:
                    seen.add(ident)
                    ordered.append(ident)
            page += 1
            time.sleep(config.delay_seconds)
        added = len(ordered) - before
        if added:
            logger.info("  language=%-12s added %s new identifiers",
                        name, f"{added:,}")

    logger.info("TOTAL %s unique identifiers across %d language spelling(s)",
                f"{len(ordered):,}", len(names))
    return ordered[:config.max_items]


def find_text_file(identifier: str, session, delay: float) -> str | None:
    """Find the plain-text OCR file inside an archive.org item.

    Items expose many derivatives. We want the full-text OCR output, which is
    conventionally ``<identifier>_djvu.txt``, but the naming is not guaranteed,
    so the metadata listing is consulted rather than assumed.

    Args:
        identifier: Archive.org item identifier.
        session: Requests session.
        delay: Politeness delay after the request.

    Returns:
        The filename of the text derivative, or None.
    """
    try:
        resp = session.get(METADATA_URL.format(identifier=identifier), timeout=45)
        meta = resp.json()
    except Exception as exc:
        logger.debug("metadata failed for %s: %s", identifier, type(exc).__name__)
        return None
    finally:
        time.sleep(delay)

    files = meta.get("files", [])
    # Prefer the djvu OCR text; fall back to any sizeable .txt derivative.
    for f in files:
        name = f.get("name", "")
        if name.endswith("_djvu.txt"):
            return name
    for f in files:
        name = f.get("name", "")
        if name.endswith(".txt") and int(f.get("size") or 0) > 20_000:
            return name
    return None


def chunk_text(text: str, target: int, minimum: int) -> list[str]:
    """Split a book into chapter-sized chunks at paragraph boundaries.

    A whole volume is unusable as a single training document -- it dwarfs the
    context window and would be rejected by the corpus-wide length ceiling.
    Splitting on blank lines keeps paragraphs intact rather than cutting
    mid-sentence at a fixed offset.

    Args:
        text: Full book text.
        target: Approximate chunk size in characters.
        minimum: Discard chunks below this size.

    Returns:
        A list of chunks.
    """
    chunks, buf, size = [], [], 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        buf.append(para)
        size += len(para) + 2
        if size >= target:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
    if size >= minimum:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if len(c) >= minimum]


def collect(config: BookCollectorConfig, output_root: Path) -> dict:
    """Search, download, chunk, and write book text as manual-collection shards.

    Resumable: identifiers already processed are recorded and skipped.

    Args:
        config: Collector configuration.
        output_root: Language data directory containing ``raw/``.

    Returns:
        Statistics for the run.
    """
    from common.text_norm import NormalizerConfig, TextNormalizer

    output_root = Path(output_root)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "books_progress.json"

    state = {"done": [], "chars": 0}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    done = set(state.get("done", []))

    normalizer = TextNormalizer(NormalizerConfig(script=config.script))
    session = _session()

    ids = search_identifiers(config, session)
    todo = [i for i in ids if i not in done]
    logger.info("%s identifiers, %s already done, %s to fetch",
                f"{len(ids):,}", f"{len(done):,}", f"{len(todo):,}")

    stats = {"items_tried": 0, "items_kept": 0, "chunks": 0,
             "chars": state.get("chars", 0), "no_text_file": 0,
             "wrong_script": 0, "download_failed": 0}

    shard_idx = len(list(raw_dir.glob("books-*.jsonl")))
    out_path = raw_dir / f"books-{shard_idx:05d}.jsonl"
    out = out_path.open("a", encoding="utf-8")
    logger.info("writing %s", out_path.name)

    try:
        for identifier in todo:
            if stats["chars"] >= config.target_chars:
                logger.info("reached character target")
                break
            stats["items_tried"] += 1

            fname = find_text_file(identifier, session, config.delay_seconds)
            if not fname:
                stats["no_text_file"] += 1
                done.add(identifier)
                continue

            try:
                url = DOWNLOAD_URL.format(identifier=identifier,
                                          filename=urllib.parse.quote(fname))
                resp = session.get(url, timeout=180)
                if resp.status_code != 200:
                    stats["download_failed"] += 1
                    done.add(identifier)
                    continue
                text = resp.content.decode("utf-8", errors="replace")
            except Exception as exc:
                logger.debug("download failed for %s: %s",
                             identifier, type(exc).__name__)
                stats["download_failed"] += 1
                done.add(identifier)
                continue
            finally:
                time.sleep(config.delay_seconds)

            text = normalizer.normalize(text)
            if not text or normalizer.script_ratio(text) < config.min_script_ratio:
                # Mislabelled item, or OCR that produced mostly garbage.
                stats["wrong_script"] += 1
                done.add(identifier)
                continue

            chunks = chunk_text(text, config.chunk_chars, config.min_chunk_chars)
            for n, chunk in enumerate(chunks):
                out.write(json.dumps({
                    "id": f"ia-{identifier}-{n}",
                    "url": f"https://archive.org/details/{identifier}",
                    "source": "archive_org_books",
                    # OCR from books is manual collection per the project spec.
                    "provenance": "manual_scrape",
                    "text": chunk,
                    "n_chars": len(chunk),
                    "fetched_at": time.time(),
                }, ensure_ascii=False) + "\n")
                stats["chunks"] += 1
                stats["chars"] += len(chunk)

            stats["items_kept"] += 1
            done.add(identifier)

            if stats["items_tried"] % 10 == 0:
                out.flush()
                state = {"done": sorted(done), "chars": stats["chars"]}
                state_path.write_text(json.dumps(state), encoding="utf-8")
                pct = stats["chars"] / config.target_chars * 100
                logger.info(
                    "  tried %s kept %s | %s chunks | %s chars (%.1f%% of target)",
                    f"{stats['items_tried']:,}", f"{stats['items_kept']:,}",
                    f"{stats['chunks']:,}", f"{stats['chars']:,}", pct)

    except KeyboardInterrupt:
        logger.info("interrupted; progress saved")
    finally:
        out.close()
        state_path.write_text(
            json.dumps({"done": sorted(done), "chars": stats["chars"]}),
            encoding="utf-8")

    logger.info("done: %s", stats)
    return stats

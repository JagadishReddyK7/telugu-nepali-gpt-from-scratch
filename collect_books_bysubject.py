"""Collect archive.org volumes found by SUBJECT/COLLECTION rather than language.

    python collect_books_bysubject.py --lang nepali --target-chars 200000000

WHY THIS FINDS WHAT THE LANGUAGE SEARCH CANNOT
----------------------------------------------
``language:("Nepali")`` and its eleven alias spellings are now exhausted: every
identifier they return has been processed. But archive.org's language field is
contributor-supplied and often simply BLANK. A scanned Nepali volume with no
language metadata is invisible to any language query, no matter how many
spellings you try.

Subject, collection and title queries reach those records. The tradeoff is that
they return plenty of items that are NOT in the target language -- an English
book about Nepal matches ``subject:(Nepal)`` perfectly well.

That tradeoff is handled the right way round: instead of trusting metadata, the
script-ratio floor reads the actual text and rejects anything that is not
predominantly the target script. That is a STRICTER test than the language
field, not a looser one. Expect a low keep rate and do not be alarmed by it --
rejections here cost one HTTP request each.

Shares ``books_progress.json`` with the language-based collector, so nothing is
fetched twice and this can be run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.parse
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.archive_books import (
    DOWNLOAD_URL, SEARCH_URL, USER_AGENT, chunk_text, find_text_file,
)
from common.run_crawl import setup_logging

logger = logging.getLogger(__name__)

QUERIES = {
    "nepali": [
        'subject:("Nepal") AND mediatype:(texts)',
        'subject:("Nepali") AND mediatype:(texts)',
        'subject:("Nepalese") AND mediatype:(texts)',
        'subject:("Nepali literature") AND mediatype:(texts)',
        'subject:("Kathmandu") AND mediatype:(texts)',
        'subject:("Himalaya") AND mediatype:(texts)',
        'collection:("nepal") AND mediatype:(texts)',
        'collection:("madan-puraskar-pustakalaya") AND mediatype:(texts)',
        'collection:("nepalnationallibrary") AND mediatype:(texts)',
        'title:("Nepal") AND mediatype:(texts)',
        'description:("Nepali") AND mediatype:(texts)',
    ],
    "telugu": [
        'subject:("Telugu") AND mediatype:(texts)',
        'subject:("Andhra Pradesh") AND mediatype:(texts)',
        'subject:("Telangana") AND mediatype:(texts)',
        'subject:("Telugu literature") AND mediatype:(texts)',
        'collection:("digitallibraryindia") AND mediatype:(texts)',
        'collection:("osmania") AND mediatype:(texts)',
        'title:("Telugu") AND mediatype:(texts)',
        'description:("Telugu") AND mediatype:(texts)',
    ],
}
SCRIPTS = {"nepali": "devanagari", "telugu": "telugu"}


def search(query: str, session, limit: int, delay: float,
           attempts: int = 4) -> tuple[list[str], bool]:
    """Return (identifiers, ok) for one advanced-search query.

    ``ok`` is False if the query never succeeded. That distinction matters: a
    query returning zero results means the space is exhausted, while a query
    that timed out means nothing at all. Conflating them produces a confident
    "exhausted" message when the truth is "archive.org throttled us".
    """
    out, page, ok = [], 1, False
    while len(out) < limit:
        params = {"q": query, "fl[]": "identifier", "rows": 500,
                  "page": page, "output": "json"}
        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params, doseq=True)}"
        data = None
        for attempt in range(attempts):
            try:
                data = session.get(url, timeout=90).json()
                break
            except Exception as exc:
                wait = delay * (4 ** attempt)
                logger.warning("  %s on %s (page %d), retry in %.0fs",
                               type(exc).__name__, query[:34], page, wait)
                time.sleep(wait)
        if data is None:
            logger.error("  GAVE UP on %s after %d attempts", query[:44], attempts)
            break
        ok = True
        resp = data.get("response", {})
        docs = resp.get("docs", [])
        if page == 1:
            logger.info("  %-58s -> %s items", query[:58],
                        f"{resp.get('numFound', 0):,}")
        if not docs:
            break
        out.extend(d["identifier"] for d in docs if d.get("identifier"))
        page += 1
        time.sleep(delay)
    return out[:limit], ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True, choices=("nepali", "telugu"))
    p.add_argument("--target-chars", type=int, default=200_000_000)
    p.add_argument("--per-query", type=int, default=4000)
    p.add_argument("--min-script-ratio", type=float, default=0.60)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--refresh", action="store_true",
                   help="Ignore the cached candidate list and search again.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    from common.text_norm import NormalizerConfig, TextNormalizer
    norm = TextNormalizer(NormalizerConfig(script=SCRIPTS[args.lang]))

    root = Path(f"data/{args.lang}")
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    state_path = root / "books_progress.json"

    state = {"done": [], "chars": 0}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    done = set(state.get("done", []))
    logger.info("%s identifiers already processed", f"{len(done):,}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    # Candidate identifiers are CACHED. Searching is the fragile part -- 88 paged
    # requests is enough for archive.org to start refusing connections -- and
    # the result is a stable list. Persisting it means a throttled restart costs
    # nothing, and it removes any need to re-search after a crash.
    cache_path = root / "books_candidates.json"
    cached: list[str] = []
    if cache_path.exists() and not args.refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.info("loaded %s cached candidate identifiers from %s",
                        f"{len(cached):,}", cache_path.name)
        except json.JSONDecodeError:
            cached = []

    if not cached:
        seen, found, any_ok = set(), [], False
        for q in QUERIES[args.lang]:
            ids, ok = search(q, session, args.per_query, args.delay)
            any_ok = any_ok or ok
            for ident in ids:
                if ident not in seen:
                    seen.add(ident); found.append(ident)
        if not any_ok:
            logger.error(
                "EVERY search failed -- this is a network or throttling problem, "
                "NOT an exhausted pool. Wait 15-20 minutes and re-run; nothing "
                "has been lost.")
            return
        cached = found
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
        logger.info("cached %s candidate identifiers -> %s",
                    f"{len(cached):,}", cache_path.name)

    todo = [i for i in cached if i not in done]
    logger.info("%s candidates, %s already processed, %s to try",
                f"{len(cached):,}", f"{len(cached) - len(todo):,}",
                f"{len(todo):,}")
    if not todo:
        logger.info("every cached candidate has been processed. Re-run with "
                    "--refresh to search again for new items.")
        return

    idx = len(list(raw.glob("books-*.jsonl")))
    out_path = raw / f"books-{idx:05d}.jsonl"
    out = out_path.open("a", encoding="utf-8")
    logger.info("writing %s", out_path.name)

    st = {"tried": 0, "kept": 0, "chunks": 0, "chars": 0,
          "no_text": 0, "wrong_script": 0, "dl_failed": 0}
    try:
        for ident in todo:
            if st["chars"] >= args.target_chars:
                logger.info("reached target"); break
            st["tried"] += 1
            fname = find_text_file(ident, session, args.delay)
            if not fname:
                st["no_text"] += 1; done.add(ident); continue
            try:
                url = DOWNLOAD_URL.format(identifier=ident,
                                          filename=urllib.parse.quote(fname))
                r = session.get(url, timeout=180)
                if r.status_code != 200:
                    st["dl_failed"] += 1; done.add(ident); continue
                text = r.content.decode("utf-8", errors="replace")
            except Exception:
                st["dl_failed"] += 1; done.add(ident); continue
            finally:
                time.sleep(args.delay)

            text = norm.normalize(text)
            # The content decides the language, not the metadata.
            if not text or norm.script_ratio(text) < args.min_script_ratio:
                st["wrong_script"] += 1; done.add(ident); continue

            for n, ch in enumerate(chunk_text(text, 8000, 500)):
                out.write(json.dumps({
                    "id": f"ia-{ident}-{n}",
                    "url": f"https://archive.org/details/{ident}",
                    "source": "archive_org_books",
                    "provenance": "manual_scrape",
                    "text": ch, "n_chars": len(ch), "fetched_at": time.time(),
                }, ensure_ascii=False) + "\n")
                st["chunks"] += 1; st["chars"] += len(ch)
            st["kept"] += 1; done.add(ident)

            if st["tried"] % 20 == 0:
                out.flush()
                state_path.write_text(json.dumps(
                    {"done": sorted(done),
                     "chars": state.get("chars", 0) + st["chars"]}),
                    encoding="utf-8")
                logger.info("  tried %s kept %s (%.0f%%) | %s chars (%.0f%% of target)",
                            f"{st['tried']:,}", f"{st['kept']:,}",
                            st["kept"] / st["tried"] * 100, f"{st['chars']:,}",
                            st["chars"] / args.target_chars * 100)
    except KeyboardInterrupt:
        logger.info("interrupted; progress saved")
    finally:
        out.close()
        state_path.write_text(json.dumps(
            {"done": sorted(done), "chars": state.get("chars", 0) + st["chars"]}),
            encoding="utf-8")

    logger.info("done: %s", st)
    print(f"\n{args.lang}: +{st['kept']:,} volumes, {st['chars']:,} new characters")


if __name__ == "__main__":
    main()
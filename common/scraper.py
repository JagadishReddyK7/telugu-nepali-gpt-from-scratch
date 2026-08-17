"""
A polite, resumable, sitemap-driven news crawler.

WHY WE SCRAPE AT ALL
--------------------
The project spec requires that, for BOTH languages, at least 20% of final
training tokens come from *manual collection* -- text we gather ourselves rather
than download as a prepared corpus. At our token targets that is a large amount
of text, far more than OCR could ever produce, so scraping is the only realistic
route. It is also the long pole in the schedule: crawlers need days of wall-clock
time, which is why this module gets written and launched before anything else.

DESIGN DECISIONS
----------------
*Sitemaps, not link-following.* Almost every news site publishes an XML sitemap
(usually a sitemap *index* pointing at dated child sitemaps). Reading those gives
us a clean list of article URLs without crawling navigation pages, tag pages, and
infinite-scroll traps. It is dramatically more efficient and dramatically more
polite than a breadth-first link crawl.

*SQLite for state.* The crawl will be interrupted -- Colab preempts, networks
drop, you close the laptop. A SQLite table of seen URLs with their status makes
restart trivial and idempotent: re-running the script picks up exactly where it
stopped, and never re-fetches a URL it already has. A flat file would work too
but corrupts badly on interrupted writes.

*trafilatura for extraction.* Turning a news page into clean article text means
discarding nav bars, ads, cookie banners, related-article widgets, and comment
sections. trafilatura is the best-in-class open-source tool for this and handles
non-Latin scripts correctly. Naive extraction (e.g. BeautifulSoup ``get_text()``)
produces text that is perhaps 40% boilerplate, which would poison the corpus with
thousands of repetitions of the same menu items -- an especially nasty failure
because deduplication works at the document level and would not catch it.

*Politeness is not optional.* We respect robots.txt, send a real User-Agent with
contact information, rate-limit per domain, and back off on errors. Beyond being
correct behaviour, an aggressive crawler gets IP-banned within hours and you lose
the source entirely.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

# Identify ourselves honestly. Site operators who want to contact us can, and
# a transparent UA is far less likely to be blocked than a spoofed browser one.
USER_AGENT = (
    "LMA-Course-Research-Crawler/1.0 "
    "(academic monolingual corpus collection; contact: hrushikratan.j@vaayum.com)"
)

# Ordinary request headers. Some CDN front-ends reject requests that omit these
# outright, producing a 403 that looks like a deliberate policy block but is
# really just a malformed-looking request. robots.txt remains the authority on
# what we are allowed to crawl -- see `_robots_allows`, which is never bypassed.
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ne;q=0.8,te;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass
class ScraperConfig:
    """Configuration for :class:`NewsScraper`.

    Attributes:
        name: Short identifier for this source, e.g. "ekantipur". Stored as the
            provenance tag on every document so the report can break tokens down
            by source.
        sitemap_urls: Sitemap index or sitemap URLs to start from.
        output_dir: Directory for JSONL output shards.
        state_db: SQLite file tracking crawl progress.
        delay_seconds: Base politeness delay between requests to this domain.
        jitter_seconds: Random extra delay, uniformly sampled from
            [0, jitter_seconds]. Prevents our requests forming a suspiciously
            regular pattern.
        max_workers: Concurrent fetch threads. Keep low -- this is a shared
            budget with delay_seconds, and 4 threads at 1s delay is already
            ~4 requests/second against one host.
        timeout: Per-request timeout in seconds.
        max_retries: Retry attempts for transient failures.
        docs_per_shard: Documents per output JSONL file. Sharding keeps
            individual files small enough to inspect and makes resumption after
            a crash lose at most one partial shard.
        min_chars: Minimum extracted length to keep a document. Short pages are
            usually photo galleries, video stubs, or error pages.
        max_chars: Maximum extracted length to keep a document. This exists to
            reject SECTION LANDING PAGES -- URLs like ``/editorial`` or
            ``/movie/reviews`` that concatenate dozens of full articles into
            one page. Sitemaps routinely list them alongside real articles.

            They must be filtered at collection time, because near-duplicate
            detection cannot save us later: an article embedded inside a
            700,000-character listing page has a *low* Jaccard similarity to
            its own standalone copy (the listing is ~100x longer), so MinHash
            sees two distinct documents and keeps both. That silently
            duplicates a large fraction of the corpus.

            In an early Telugu run these pages were 14.8% of documents but
            96.3% of all collected text. 50,000 characters is far above any
            real news article (~8,000 words) and well below these listings.
        url_must_contain: If non-empty, only crawl URLs containing one of these
            substrings. Useful for restricting to article paths.
        url_must_not_contain: Skip URLs containing any of these substrings.
    """

    name: str
    sitemap_urls: list[str]
    output_dir: Path
    state_db: Path
    delay_seconds: float = 1.0
    jitter_seconds: float = 0.5
    max_workers: int = 4
    timeout: int = 20
    max_retries: int = 3
    docs_per_shard: int = 5000
    min_chars: int = 400
    max_chars: int = 50_000
    url_must_contain: list[str] = field(default_factory=list)
    url_must_not_contain: list[str] = field(default_factory=lambda: [
        "/video/", "/photo/", "/gallery/", "/tag/", "/author/",
        "/page/", "?", "/live/", "/webstories/",
    ])


class CrawlState:
    """SQLite-backed record of which URLs have been seen and fetched.

    The table is intentionally simple::

        urls(url TEXT PRIMARY KEY, status TEXT, ts REAL)

    where ``status`` is one of ``pending``, ``done``, ``failed``, ``skipped``.
    The primary key gives us free deduplication of URLs discovered in multiple
    sitemaps.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because worker threads share this connection;
        # all writes go through self._lock so this is safe.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS urls ("
            "  url TEXT PRIMARY KEY,"
            "  status TEXT NOT NULL,"
            "  ts REAL NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON urls(status)"
        )
        self._conn.commit()

    def add_pending(self, urls: list[str]) -> int:
        """Insert URLs as pending, ignoring ones already known.

        Args:
            urls: Candidate URLs.

        Returns:
            Number of genuinely new URLs inserted.
        """
        now = time.time()
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT OR IGNORE INTO urls(url, status, ts) VALUES (?, 'pending', ?)",
                [(u, now) for u in urls],
            )
            self._conn.commit()
            return self._conn.total_changes - before

    def take_pending(self, limit: int) -> list[str]:
        """Fetch a batch of URLs still awaiting download.

        Args:
            limit: Maximum number to return.

        Returns:
            Up to ``limit`` pending URLs.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT url FROM urls WHERE status='pending' LIMIT ?", (limit,)
            ).fetchall()
        return [r[0] for r in rows]

    def mark(self, url: str, status: str) -> None:
        """Update the status of one URL.

        Args:
            url: The URL to update.
            status: One of ``done``, ``failed``, ``skipped``.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE urls SET status=?, ts=? WHERE url=?",
                (status, time.time(), url),
            )
            self._conn.commit()

    def counts(self) -> dict[str, int]:
        """Return a mapping of status to number of URLs in that status."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM urls GROUP BY status"
            ).fetchall()
        return dict(rows)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()


class ShardWriter:
    """Append documents to rolling JSONL shards, thread-safely.

    Each line is one JSON object, which makes the output streamable and means a
    truncated file loses only its last line rather than becoming unparseable.
    """

    def __init__(self, output_dir: Path, prefix: str, docs_per_shard: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.docs_per_shard = docs_per_shard
        self._lock = threading.Lock()
        self._count = 0
        # Resume-aware: continue numbering after any shards already on disk so a
        # restart never overwrites previous output.
        existing = sorted(self.output_dir.glob(f"{prefix}-*.jsonl"))
        self._shard_idx = len(existing)
        # LAZY OPEN. The file is not created until the first document is
        # written. Opening eagerly meant a `--discover` run (which writes no
        # documents at all) littered the output directory with empty shards,
        # and worse, each empty shard bumped the resume counter so the next
        # real run started numbering from the wrong index.
        self._fh = None

    def _open_new_shard(self) -> None:
        """Close the current shard (if any) and open the next one."""
        if self._fh is not None:
            self._fh.close()
        path = self.output_dir / f"{self.prefix}-{self._shard_idx:05d}.jsonl"
        self._fh = path.open("a", encoding="utf-8")
        logger.info("Writing shard %s", path.name)

    def write(self, record: dict) -> None:
        """Write one document record.

        Args:
            record: JSON-serializable document dictionary.
        """
        with self._lock:
            if self._fh is None:
                self._open_new_shard()
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1
            if self._count % 200 == 0:
                self._fh.flush()
            if self._count % self.docs_per_shard == 0:
                self._shard_idx += 1
                self._open_new_shard()

    def close(self) -> None:
        """Flush and close the current shard."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class NewsScraper:
    """Crawl article URLs from sitemaps and extract clean text.

    Typical use::

        cfg = ScraperConfig(name="ekantipur", sitemap_urls=[...], ...)
        scraper = NewsScraper(cfg)
        scraper.discover()   # populate the URL frontier from sitemaps
        scraper.run()        # fetch and extract, resumable
    """

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.state = CrawlState(config.state_db)
        self.writer = ShardWriter(
            config.output_dir, config.name, config.docs_per_shard
        )
        self._session = requests.Session()
        self._session.headers.update(
            {**DEFAULT_HEADERS, "User-Agent": USER_AGENT}
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._robots_lock = threading.Lock()
        self._last_request = 0.0
        self._rate_lock = threading.Lock()
        self._stats = {"fetched": 0, "extracted": 0, "too_short": 0,
                       "too_long_listing_page": 0, "failed": 0}

    # ---------------------------------------------------------------- robots

    def _robots_allows(self, url: str) -> bool:
        """Check robots.txt for this URL's host, caching the parsed file.

        Args:
            url: URL to check.

        Returns:
            True if crawling is permitted (or robots.txt is unreachable, which
            we treat as permissive per convention).
        """
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        with self._robots_lock:
            rp = self._robots.get(host)
            if rp is None:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(f"{host}/robots.txt")
                try:
                    rp.read()
                except Exception:
                    logger.warning("Could not read robots.txt for %s", host)
                    # Convention: unreachable robots.txt means no restrictions.
                    rp = None
                self._robots[host] = rp
        if rp is None:
            return True
        return rp.can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------ rate limit

    def _throttle(self) -> None:
        """Block until enough time has passed since the last request."""
        with self._rate_lock:
            elapsed = time.time() - self._last_request
            wait = self.config.delay_seconds - elapsed
            wait += random.uniform(0, self.config.jitter_seconds)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

    # -------------------------------------------------------------- fetching

    def _get(self, url: str) -> bytes | None:
        """Fetch a URL with retries and exponential backoff.

        Args:
            url: URL to fetch.

        Returns:
            Response body bytes, or None if all attempts failed.
        """
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.config.timeout)
                if resp.status_code == 200:
                    return resp.content
                # 429 = rate limited, 5xx = server trouble. Both mean "slow down".
                if resp.status_code in (429, 500, 502, 503, 504):
                    backoff = 2 ** attempt + random.uniform(0, 1)
                    logger.debug("HTTP %s on %s, backing off %.1fs",
                                 resp.status_code, url, backoff)
                    time.sleep(backoff)
                    continue
                # 404, 403, etc. are permanent -- do not retry.
                return None
            except Exception as exc:
                # Deliberately broad. `requests.RequestException` does NOT cover
                # everything that can come out of the stack -- notably
                # urllib3.exceptions.ProtocolError on a truncated chunked
                # response, which killed an early crawl run. In an unattended
                # multi-day job, one bad URL must never escalate to a dead
                # process.
                logger.debug("Request failed for %s: %s: %s",
                             url, type(exc).__name__, exc)
                time.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------- discovery

    def _parse_sitemap(self, content: bytes) -> tuple[list[str], list[str]]:
        """Parse a sitemap or sitemap index.

        Args:
            content: Raw sitemap bytes, possibly gzip-compressed.

        Returns:
            A tuple ``(child_sitemaps, page_urls)``. A sitemap *index* yields
            child sitemaps; a regular sitemap yields page URLs.
        """
        # Sitemaps are frequently served gzipped with a .xml.gz extension.
        if content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            logger.warning("Malformed sitemap XML: %s", exc)
            return [], []

        tag = root.tag.split("}")[-1]

        # RSS / Atom feeds.
        # `sitemap_discovery` falls back to /rss.xml and /feed when no real
        # sitemap exists, and its probe accepts anything containing "<rss".
        # Without this branch the feed parses as neither a sitemapindex nor a
        # (namespaced) urlset, so findall returns nothing and the site silently
        # yields ZERO urls while still reporting "found sitemap". That is
        # exactly what happened to several Nepali sources.
        #
        # Feeds are shallow -- typically only the most recent 20-50 items -- so
        # this recovers breadth, not depth. Still worth having: source
        # diversity matters as much as raw volume for a monolingual corpus.
        if tag == "rss":
            links = [
                el.text.strip()
                for el in root.findall(".//item/link")
                if el is not None and el.text
            ]
            logger.info("Parsed RSS feed: %d item link(s)", len(links))
            return [], links
        if tag == "feed":  # Atom
            links = [
                el.get("href").strip()
                for el in root.findall(".//{http://www.w3.org/2005/Atom}entry/"
                                       "{http://www.w3.org/2005/Atom}link")
                if el.get("href")
            ]
            logger.info("Parsed Atom feed: %d entry link(s)", len(links))
            return [], links

        if tag == "sitemapindex":
            children = [
                loc.text.strip()
                for loc in root.findall(".//sm:sitemap/sm:loc", SITEMAP_NS)
                if loc.text
            ]
            return children, []

        entries = [
            loc.text.strip()
            for loc in root.findall(".//sm:url/sm:loc", SITEMAP_NS)
            if loc.text
        ]

        # Some sites publish an index of child sitemaps but wrap the entries in
        # <url> elements instead of the correct <sitemap> elements, so the file
        # parses as a plain urlset. Without this check we would treat those
        # child-sitemap URLs as articles and collect a handful of XML files
        # instead of thousands of pages -- exactly what happened on setopati,
        # which reported a suspiciously round "6 urls".
        child_sitemaps = [
            u for u in entries
            if u.lower().split("?")[0].endswith((".xml", ".xml.gz"))
        ]
        if child_sitemaps:
            page_urls = [u for u in entries if u not in set(child_sitemaps)]
            logger.info(
                "Sitemap %s lists %d child sitemaps inside <url> elements; "
                "following them as an index.",
                root.tag.split("}")[-1], len(child_sitemaps),
            )
            return child_sitemaps, page_urls

        return [], entries

    def _url_wanted(self, url: str) -> bool:
        """Apply the configured URL include/exclude filters.

        Args:
            url: Candidate URL.

        Returns:
            True if the URL should be crawled.
        """
        low = url.lower()
        if any(bad in low for bad in self.config.url_must_not_contain):
            return False
        if self.config.url_must_contain:
            return any(good in low for good in self.config.url_must_contain)
        return True

    def discover(self, max_sitemaps: int = 2000) -> int:
        """Walk sitemaps breadth-first and add article URLs to the frontier.

        Args:
            max_sitemaps: Safety cap on how many sitemap files to read. Large
                news archives can have thousands of dated child sitemaps.

        Returns:
            Number of new URLs added to the frontier.
        """
        queue = list(self.config.sitemap_urls)
        seen_sitemaps: set[str] = set()
        total_new = 0
        processed = 0

        while queue and processed < max_sitemaps:
            sm_url = queue.pop(0)
            if sm_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sm_url)
            processed += 1

            content = self._get(sm_url)
            if content is None:
                logger.warning("Could not fetch sitemap %s", sm_url)
                continue

            children, urls = self._parse_sitemap(content)
            queue.extend(children)

            wanted = [u for u in urls if self._url_wanted(u)]
            if wanted:
                added = self.state.add_pending(wanted)
                total_new += added
                logger.info(
                    "[%s] %s -> %d urls (%d new). Frontier discovery: %d total new",
                    self.config.name, sm_url.split("/")[-1],
                    len(wanted), added, total_new,
                )

        logger.info("[%s] Discovery complete: %d new URLs from %d sitemaps",
                    self.config.name, total_new, processed)
        return total_new

    # -------------------------------------------------------------- crawling

    def _process_url(self, url: str) -> None:
        """Fetch one article, extract its text, and write it out.

        Args:
            url: Article URL to process.
        """
        import trafilatura

        if not self._robots_allows(url):
            self.state.mark(url, "skipped")
            return

        content = self._get(url)
        if content is None:
            self._stats["failed"] += 1
            self.state.mark(url, "failed")
            return

        self._stats["fetched"] += 1

        try:
            html = content.decode("utf-8", errors="replace")
            text = trafilatura.extract(
                html,
                include_comments=False,   # comment sections are noisy and often
                                          # in a different language/register
                include_tables=False,
                no_fallback=False,        # allow the readability fallback
                favor_precision=True,     # prefer dropping text over keeping
                                          # boilerplate -- corpus purity matters
                                          # more to us than recall
            )
        except Exception as exc:
            logger.debug("Extraction failed for %s: %s", url, exc)
            self.state.mark(url, "failed")
            return

        if not text or len(text) < self.config.min_chars:
            self._stats["too_short"] += 1
            self.state.mark(url, "skipped")
            return

        # Section landing page, not an article. See ScraperConfig.max_chars for
        # why this cannot be deferred to the deduplication stage.
        if len(text) > self.config.max_chars:
            self._stats["too_long_listing_page"] += 1
            logger.debug("Rejected likely listing page (%d chars): %s",
                         len(text), url)
            self.state.mark(url, "skipped")
            return

        self._stats["extracted"] += 1
        self.writer.write({
            # Stable synthetic ID: lets us cross-reference a document between
            # the raw shard and every downstream cleaning stage.
            "id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
            "url": url,
            "source": self.config.name,
            # This flag is what makes the >=20% manual-collection requirement
            # auditable. Every downloaded-corpus record carries "downloaded".
            "provenance": "manual_scrape",
            "text": text,
            "n_chars": len(text),
            "fetched_at": time.time(),
        })

    def run(self, max_docs: int | None = None, batch_size: int = 200) -> dict:
        """Process the pending frontier until exhausted or ``max_docs`` reached.

        Safe to interrupt and re-run: completed URLs are recorded in the state
        database and will not be fetched again.

        Args:
            max_docs: Stop after extracting roughly this many documents. None
                means run until the frontier is empty.
            batch_size: URLs to pull from the frontier per round.

        Returns:
            Dictionary of crawl statistics.
        """
        start = time.time()
        try:
            while True:
                if max_docs is not None and self._stats["extracted"] >= max_docs:
                    logger.info("[%s] Reached max_docs=%d", self.config.name, max_docs)
                    break

                batch = self.state.take_pending(batch_size)
                if not batch:
                    logger.info("[%s] Frontier exhausted", self.config.name)
                    break

                # Mark in-flight URLs as done up front. If we crash mid-batch we
                # lose at most `batch_size` URLs rather than re-fetching them
                # forever in a restart loop.
                for url in batch:
                    self.state.mark(url, "done")

                with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                    list(pool.map(self._process_url, batch))

                elapsed = time.time() - start
                rate = self._stats["extracted"] / max(elapsed, 1)
                logger.info(
                    "[%s] extracted=%d fetched=%d short=%d listing=%d failed=%d "
                    "(%.1f docs/s)",
                    self.config.name, self._stats["extracted"],
                    self._stats["fetched"], self._stats["too_short"],
                    self._stats["too_long_listing_page"],
                    self._stats["failed"], rate,
                )
        except KeyboardInterrupt:
            logger.info("[%s] Interrupted by user; state saved.", self.config.name)
        finally:
            self.writer.close()

        return dict(self._stats)

    def close(self) -> None:
        """Release the state database and output file handles."""
        self.writer.close()
        self.state.close()

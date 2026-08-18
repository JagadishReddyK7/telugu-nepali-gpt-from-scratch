"""Probe why nepalpress fetches are failing.

Takes real pending URLs from the crawl frontier and fetches them directly,
printing the HTTP status. The crawler's own logging only records "failed",
which does not distinguish a 403 block from a 404 from a timeout -- and the
remedy is different for each.
"""
import sqlite3, sys, time
import requests

UA = ("LMA-Course-Research-Crawler/1.0 "
      "(academic monolingual corpus collection; contact: hrushikratan.j@vaayum.com)")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ne;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

db = "data/nepali/state/nepalpress.db"
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
pending = [r[0] for r in conn.execute(
    "SELECT url FROM urls WHERE status='pending' LIMIT 4")]
done = [r[0] for r in conn.execute(
    "SELECT url FROM urls WHERE status='done' LIMIT 2")]
conn.close()

print("=== PENDING urls (never fetched) ===")
for u in pending:
    t0 = time.time()
    try:
        r = requests.get(u, headers=HEADERS, timeout=25)
        print(f"  {r.status_code}  {len(r.content):>8,}B  {time.time()-t0:4.1f}s  {u[:80]}")
    except Exception as e:
        print(f"  ERR {type(e).__name__:24s} {time.time()-t0:4.1f}s  {u[:80]}")

print("\n=== previously-SUCCEEDED urls (control) ===")
for u in done:
    t0 = time.time()
    try:
        r = requests.get(u, headers=HEADERS, timeout=25)
        print(f"  {r.status_code}  {len(r.content):>8,}B  {time.time()-t0:4.1f}s  {u[:80]}")
    except Exception as e:
        print(f"  ERR {type(e).__name__:24s} {time.time()-t0:4.1f}s  {u[:80]}")

print("\n=== plain browser-style UA, one pending url ===")
if pending:
    try:
        r = requests.get(pending[0], timeout=25, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"})
        print(f"  {r.status_code}  {len(r.content):,}B")
    except Exception as e:
        print(f"  ERR {type(e).__name__}")
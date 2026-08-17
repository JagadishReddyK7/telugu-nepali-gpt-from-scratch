"""
Eyeball the collected corpus: random samples plus automated quality checks.

You should look at your data before you tokenize it. Automated filters catch
what you thought to check for; reading actual documents catches what you did
not. Corpus bugs that slip past statistics are common and expensive -- an
extraction that silently keeps the byline and share buttons on every article,
a source that turns out to be machine-translated, a section of the site that
is really a photo gallery with captions.

    python scripts/inspect_corpus.py                    # both languages
    python scripts/inspect_corpus.py --lang telugu -n 5 # 5 samples per source
    python scripts/inspect_corpus.py --html             # write an HTML report

WHY THE HTML REPORT MATTERS
---------------------------
Windows terminals generally cannot render Telugu or Devanagari correctly --
you get boxes, or characters in the wrong order, because console font fallback
for Indic scripts is poor and complex-script shaping is not applied. That makes
terminal output useless for the one thing you actually need here, which is
judging whether the text reads like real language.

The HTML report opens in a browser, which does shape Indic text properly. Use
``--html`` and read that. The terminal output is for the numbers.

WHAT THE CHECKS MEAN
--------------------
script_ratio      Fraction of letters in the expected script. Should be ~0.95+.
                  Low values mean English or another script leaked in.
dup_line_frac     Fraction of lines that are duplicates within the document.
                  High values mean navigation or boilerplate survived extraction.
mean_word_len     Sanity check on extraction. Very high means words ran together
                  (markup stripped without spaces); very low means fragments.
short_line_frac   Fraction of lines under 30 characters. Menus, tags and captions
                  are short; prose is not. High values suggest a non-article page.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text_norm import NormalizerConfig, TextNormalizer  # noqa: E402

SCRIPT_FOR_LANG = {"telugu": "telugu", "nepali": "devanagari"}


def doc_metrics(text: str, normalizer: TextNormalizer) -> dict:
    """Compute quality metrics for one document.

    Args:
        text: Raw document text.
        normalizer: Normalizer configured for this language's script.

    Returns:
        Dictionary of metric name to value.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    words = text.split()
    counts = Counter(lines)
    dup_lines = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(lines),
        "script_ratio": normalizer.script_ratio(text),
        "mean_word_len": (statistics.mean(len(w) for w in words) if words else 0.0),
        "dup_line_frac": (dup_lines / len(lines)) if lines else 0.0,
        "short_line_frac": (
            sum(1 for ln in lines if len(ln) < 30) / len(lines) if lines else 0.0
        ),
    }


def flags_for(m: dict) -> list[str]:
    """Return human-readable warnings for a document's metrics.

    Args:
        m: Metrics dictionary from :func:`doc_metrics`.

    Returns:
        List of warning strings; empty if the document looks healthy.
    """
    out = []
    if m["script_ratio"] < 0.90:
        out.append(f"LOW SCRIPT RATIO {m['script_ratio']:.2f} (foreign text?)")
    if m["dup_line_frac"] > 0.25:
        out.append(f"REPEATED LINES {m['dup_line_frac']:.0%} (boilerplate?)")
    if m["short_line_frac"] > 0.70:
        out.append(f"MOSTLY SHORT LINES {m['short_line_frac']:.0%} (menu/gallery?)")
    if m["mean_word_len"] > 25:
        out.append(f"HUGE MEAN WORD LEN {m['mean_word_len']:.1f} (lost spaces?)")
    if m["chars"] < 300:
        out.append("VERY SHORT")
    return out


def load_samples(lang: str, per_source: int, seed: int = 7) -> dict:
    """Reservoir-sample documents per source without loading the corpus.

    Args:
        lang: Language directory name.
        per_source: Number of documents to keep per source.
        seed: RNG seed so repeated runs show the same documents.

    Returns:
        Mapping of source name to a list of document dictionaries.
    """
    rng = random.Random(seed)
    reservoir: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    for f in sorted(glob.glob(f"data/{lang}/raw/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(d.get("text", "")) > 50_000:
                continue  # listing page; dropped by cleaning
            src = d.get("source", "?")
            seen[src] += 1
            # Standard reservoir sampling: every document has an equal chance
            # of ending up in the sample, in one pass, without holding the
            # whole corpus in memory.
            if len(reservoir[src]) < per_source:
                reservoir[src].append(d)
            else:
                j = rng.randrange(seen[src])
                if j < per_source:
                    reservoir[src][j] = d
    return {"samples": dict(reservoir), "counts": dict(seen)}


def scan_all(lang: str, normalizer: TextNormalizer, limit: int = 20000) -> dict:
    """Compute aggregate metrics over up to ``limit`` documents.

    Args:
        lang: Language directory name.
        normalizer: Normalizer for this language.
        limit: Stop after this many documents, to keep the scan fast.

    Returns:
        Dictionary of aggregate statistics and flagged-document counts.
    """
    n = 0
    agg = defaultdict(list)
    flagged = Counter()
    for f in sorted(glob.glob(f"data/{lang}/raw/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text", "")
            if len(t) > 50_000:
                continue
            m = doc_metrics(t, normalizer)
            for k, v in m.items():
                agg[k].append(v)
            for fl in flags_for(m):
                flagged[fl.split("(")[0].strip()] += 1
            n += 1
            if n >= limit:
                return {"n": n, "agg": agg, "flagged": flagged}
    return {"n": n, "agg": agg, "flagged": flagged}


def print_report(lang: str, per_source: int) -> dict:
    """Print the terminal report and return the data needed for HTML output.

    Args:
        lang: Language directory name.
        per_source: Documents to sample per source.

    Returns:
        Dictionary with samples and scan results.
    """
    normalizer = TextNormalizer(NormalizerConfig(script=SCRIPT_FOR_LANG[lang]))
    picked = load_samples(lang, per_source)
    samples, counts = picked["samples"], picked["counts"]

    print("=" * 76)
    print(f"  {lang.upper()}  -- corpus inspection")
    print("=" * 76)
    if not counts:
        print("  No documents collected yet.\n")
        return {"samples": {}, "scan": None, "normalizer": normalizer}

    scan = scan_all(lang, normalizer)
    agg = scan["agg"]

    print(f"\n  Scanned {scan['n']:,} documents\n")
    print(f"  {'metric':18s}{'mean':>10s}{'median':>10s}{'p10':>10s}{'p90':>10s}")
    print("  " + "-" * 58)
    for key in ("chars", "words", "script_ratio", "mean_word_len",
                "dup_line_frac", "short_line_frac"):
        v = sorted(agg[key])
        if not v:
            continue
        fmt = ".3f" if max(v) <= 2 else ",.0f"
        print(f"  {key:18s}{statistics.mean(v):>10{fmt}}"
              f"{statistics.median(v):>10{fmt}}"
              f"{v[len(v)//10]:>10{fmt}}{v[9*len(v)//10]:>10{fmt}}")

    print(f"\n  QUALITY FLAGS (out of {scan['n']:,} documents)")
    if not scan["flagged"]:
        print("    none -- nothing tripped a check")
    for flag, c in scan["flagged"].most_common():
        pct = c / scan["n"] * 100
        print(f"    {flag:38s}{c:>8,}  ({pct:5.1f}%)")

    print(f"\n  DOCUMENTS PER SOURCE")
    for src, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {src:20s}{c:>10,}")

    print("\n  Terminal output cannot render Telugu or Devanagari reliably.")
    print("  Re-run with --html and open the report in a browser to read the text.\n")
    return {"samples": samples, "scan": scan, "normalizer": normalizer}


def write_html(results: dict, out_path: Path) -> None:
    """Write a browser-readable sample report.

    Args:
        results: Mapping of language to the dict returned by :func:`print_report`.
        out_path: Destination HTML file.
    """
    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Corpus inspection</title><style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--muted:#6b6b6b;--line:#e2e0dc;--warn:#b45309;--ok:#15803d}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.5rem 1.5rem;
 font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .3rem}
h2{font-size:1.25rem;margin:2.5rem 0 .5rem;padding-bottom:.4rem;
 border-bottom:2px solid var(--line)}
h3{font-size:.95rem;margin:2rem 0 .5rem;color:var(--muted);
 text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--muted);margin:0 0 2rem}
.doc{background:#fff;border:1px solid var(--line);border-radius:10px;
 padding:1.1rem 1.3rem;margin:0 0 1rem}
.meta{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
 color:var(--muted);margin-bottom:.7rem;word-break:break-all}
.metrics{font:12px/1.5 ui-monospace,monospace;color:var(--muted);
 margin-bottom:.7rem}
.metrics b{color:var(--fg);font-weight:600}
.flag{display:inline-block;background:#fef3c7;color:var(--warn);
 border-radius:5px;padding:.15rem .5rem;font-size:12px;margin:0 .3rem .3rem 0}
.clean{color:var(--ok);font-size:12px}
/* Indic script rendering: browsers shape these correctly, terminals do not. */
.text{font-family:"Noto Sans Telugu","Noto Sans Devanagari","Nirmala UI",
 "Gautami","Mangal",system-ui,sans-serif;font-size:1.02rem;line-height:1.85;
 white-space:pre-wrap;max-height:22rem;overflow-y:auto;
 border-top:1px solid var(--line);padding-top:.8rem}
</style></head><body><div class="wrap">
<h1>Corpus inspection</h1>
<p class="sub">Random samples from the collected text. Read these and ask:
does it look like clean article prose, or is there navigation, bylines,
repeated menus, or the wrong language?</p>"""]

    for lang, res in results.items():
        samples = res["samples"]
        if not samples:
            continue
        norm = res["normalizer"]
        parts.append(f"<h2>{html.escape(lang.title())}</h2>")
        for src in sorted(samples):
            parts.append(f"<h3>{html.escape(src)}</h3>")
            for d in samples[src]:
                t = d.get("text", "")
                m = doc_metrics(t, norm)
                fl = flags_for(m)
                badge = ("".join(f'<span class="flag">{html.escape(x)}</span>'
                                 for x in fl)
                         if fl else '<span class="clean">no flags</span>')
                parts.append(
                    '<div class="doc">'
                    f'<div class="meta">{html.escape(d.get("url", ""))}</div>'
                    f'<div class="metrics">'
                    f'<b>{m["chars"]:,}</b> chars &middot; '
                    f'<b>{m["words"]:,}</b> words &middot; '
                    f'script <b>{m["script_ratio"]:.2f}</b> &middot; '
                    f'dup lines <b>{m["dup_line_frac"]:.0%}</b> &middot; '
                    f'short lines <b>{m["short_line_frac"]:.0%}</b></div>'
                    f'<div>{badge}</div>'
                    f'<div class="text">{html.escape(t[:4000])}</div>'
                    '</div>'
                )
    parts.append("</div></body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"  HTML report written to {out_path}")
    print(f"  Open it in a browser:  start {out_path}\n")


def main() -> None:
    """Parse arguments and run the inspection."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", choices=["telugu", "nepali"], default=None)
    p.add_argument("-n", "--per-source", type=int, default=3,
                   help="Documents to sample per source. Default 3.")
    p.add_argument("--html", action="store_true",
                   help="Write a browser-readable HTML report (recommended).")
    p.add_argument("--out", type=Path, default=Path("report/phase1/corpus_samples.html"))
    args = p.parse_args()

    langs = [args.lang] if args.lang else ["telugu", "nepali"]
    results = {lang: print_report(lang, args.per_source) for lang in langs}

    if args.html:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_html(results, args.out)


if __name__ == "__main__":
    main()

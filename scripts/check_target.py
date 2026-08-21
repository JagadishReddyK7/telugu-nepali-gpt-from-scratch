"""
Am I on track for the token target? Measures the corpus you actually have.

    python scripts/check_target.py --lang telugu --target 500
    python scripts/check_target.py --lang nepali --target 500 --survival observed

Every earlier estimate in this project came from assuming characters-per-token.
This measures the real characters on disk, splits them by provenance, applies
the >=20% manual rule, and projects survival through cleaning -- so you can see
whether to keep collecting or start processing.

WHY THIS SCRIPT WAS WRONG UNTIL NOW
-----------------------------------
It used a single hardcoded 3.2 chars/token for both languages and both
provenance classes. The measured values, taken from completed tokenizer runs at
vocab 16,000, are:

                    manual   downloaded
        Telugu       3.60       4.32
        Nepali       3.33       4.67

So every token estimate this script printed was ~30-40% too high, including a
false "ON TARGET" verdict for Telugu. Manual text tokenizes DENSER than
downloaded text -- news and OCR'd book prose has a different subword profile
than bulk web crawl -- which is also why the manual TOKEN share always comes in
above the manual CHARACTER share. The spec's >=20% rule is stated in tokens, so
the token share is the one that counts.

THE TWO CONSTRAINTS THAT ACTUALLY BIND
--------------------------------------
1. SUPPLY.  You cannot tokenize characters you did not collect.
2. THE 20% RULE, WHICH IS A CEILING AS WELL AS A FLOOR:

        total_tokens <= manual_tokens / 0.20

   The cleaner enforces this as a ratio cap on CHARACTERS
   (``max_downloaded_to_manual_ratio`` in the language YAML), so surplus
   downloaded text is discarded rather than used. Once that cap binds,
   downloading more text is wasted work and only more MANUAL data raises the
   ceiling.

Which of the two binds is the whole question, and the answer flips depending on
the manual survival rate -- hence ``--survival``.

MANUAL SURVIVAL IS NOT WELL PINNED DOWN
---------------------------------------
"optimistic" (80%) is what this project assumed while planning. "observed"
(51%) is back-solved from the last complete Telugu pass:

    479,570,553 tokens at 23.18% manual  ->  111.2M manual tokens
    111.2M x 3.60 chars/token            ->  400.2M clean manual chars
    manual raw, less the dead greatandhra crawl (see REJECTED_SOURCES)
                                         ->  786.7M chars
    400.2 / 786.7                        ->  50.9%

Manual is overwhelmingly OCR'd archive.org books, and OCR output loses heavily
to the min_script_ratio, min_mean_word_len and max_dup_line_fraction filters.
Plan against "observed"; treat "optimistic" as the best case. Neither is a
substitute for token_counts.json after a real run.
"""

from __future__ import annotations

import argparse
import glob
import json

# Measured on completed tokenizer runs at vocab 16,000. Do not "improve" these
# by estimating -- replace them only with numbers from a later real run.
CHARS_PER_TOKEN = {
    "telugu": {"manual": 3.60, "downloaded": 4.32},
    "nepali": {"manual": 3.33, "downloaded": 4.67},
}

# Fraction of raw characters surviving normalize -> filter -> dedup.
# See the module docstring for where "observed" comes from.
SURVIVAL = {
    "observed":   {"manual": 0.51, "downloaded": 0.875},
    "optimistic": {"manual": 0.80, "downloaded": 0.875},
}

# max_downloaded_to_manual_ratio from the language YAML. Applied to CHARACTERS.
RATIO_CAP = {"telugu": 4.5, "nepali": 5.4}

# Sources tagged manual on disk that cleaning rejects essentially in full.
# Counting them toward the manual share produces a projection that cannot
# happen. greatandhra publishes in English: 48,612 documents, median
# Telugu-script ratio 0.00, 0% recoverable even at a 0.60 threshold.
REJECTED_SOURCES = {"telugu": {"greatandhra"}, "nepali": set()}

MIN_MANUAL_FRACTION = 0.20


def measure(lang: str) -> dict:
    """Sum characters on disk, split by provenance and source.

    Args:
        lang: Language directory name.

    Returns:
        Dictionary of character totals. ``manual`` excludes sources listed in
        REJECTED_SOURCES; those are reported separately as ``manual_dead``.
    """
    manual = manual_dead = downloaded = 0
    per_source: dict[str, int] = {}
    dead = REJECTED_SOURCES.get(lang, set())

    for f in sorted(glob.glob(f"data/{lang}/raw/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = len(d.get("text", ""))
            if n > 50_000:
                continue          # listing pages; cleaning drops these
            src = d.get("source", "?")
            per_source[src] = per_source.get(src, 0) + n
            if d.get("provenance") == "manual_scrape":
                if src in dead:
                    manual_dead += n
                else:
                    manual += n
            else:
                downloaded += n

    return {
        "manual": manual,
        "manual_dead": manual_dead,
        "downloaded": downloaded,
        "per_source": per_source,
    }


def project(lang: str, manual_raw: float, dl_raw: float, survival: str) -> dict:
    """Project the post-cleaning corpus in tokens under one survival scenario.

    Args:
        lang: Language name, used to select chars/token and the ratio cap.
        manual_raw: Raw manual characters on disk, excluding dead sources.
        dl_raw: Raw downloaded characters on disk.
        survival: Key into SURVIVAL.

    Returns:
        Character and token totals, plus which constraint binds.
    """
    surv = SURVIVAL[survival]
    cpt = CHARS_PER_TOKEN[lang]
    cap = RATIO_CAP[lang]

    manual_clean = manual_raw * surv["manual"]
    dl_available = dl_raw * surv["downloaded"]
    dl_used = min(dl_available, cap * manual_clean)

    manual_tokens = manual_clean / cpt["manual"]
    dl_tokens = dl_used / cpt["downloaded"]
    total = manual_tokens + dl_tokens

    return {
        "manual_clean": manual_clean,
        "dl_available": dl_available,
        "dl_used": dl_used,
        "manual_tokens": manual_tokens,
        "dl_tokens": dl_tokens,
        "total_tokens": total,
        "manual_share": manual_tokens / total if total else 0.0,
        # The hard ceiling the 20% rule imposes, regardless of how much
        # downloaded text exists.
        "ceiling_tokens": manual_tokens / MIN_MANUAL_FRACTION,
        "cap_binds": dl_available > cap * manual_clean,
    }


def report(lang: str, target_m: float, m: dict, survival: str) -> dict:
    """Print one survival scenario and return its projection."""
    p = project(lang, m["manual"], m["downloaded"], survival)
    surv = SURVIVAL[survival]
    target = target_m * 1e6

    print(f"\n  PROJECTED AFTER CLEANING -- '{survival}' "
          f"({surv['manual']:.0%} manual / {surv['downloaded']:.1%} downloaded)")
    print(f"    manual clean            {p['manual_clean']/1e6:>9.1f}M chars"
          f"  ->{p['manual_tokens']/1e6:>8.1f}M tokens")
    print(f"    downloaded usable       {p['dl_used']/1e6:>9.1f}M chars"
          f"  ->{p['dl_tokens']/1e6:>8.1f}M tokens")
    if p["cap_binds"]:
        print(f"    ({(p['dl_available'] - p['dl_used'])/1e6:.0f}M downloaded chars "
              f"discarded by the {RATIO_CAP[lang]}x ratio cap)")
    print(f"    {'-'*54}")
    print(f"    corpus                  {p['total_tokens']/1e6:>9.1f}M tokens")
    print(f"    manual TOKEN share      {p['manual_share']:>9.1%}"
          f"   (must be >= {MIN_MANUAL_FRACTION:.0%})")
    print(f"    ceiling from 20% rule   {p['ceiling_tokens']/1e6:>9.1f}M tokens")

    print(f"\n    VERDICT ({survival})")
    if p["total_tokens"] >= target and p["manual_share"] >= MIN_MANUAL_FRACTION:
        print(f"      ON TARGET -- {p['total_tokens']/1e6:.0f}M >= {target_m:.0f}M "
              f"at {p['manual_share']:.1%} manual.")
    else:
        print(f"      SHORT by {(target - p['total_tokens'])/1e6:.0f}M tokens")
        if p["ceiling_tokens"] < target:
            need_manual_tokens = target * MIN_MANUAL_FRACTION
            gap = (need_manual_tokens * CHARS_PER_TOKEN[lang]["manual"]
                   / surv["manual"]) - m["manual"]
            print("      The 20% rule is the binding constraint, not supply.")
            print(f"      Ceiling is {p['ceiling_tokens']/1e6:.0f}M tokens; "
                  f"downloading more is wasted work.")
            print(f"      need +{gap/1e6:.0f}M raw MANUAL chars "
                  f"(~{gap/657_000:,.0f} more archive.org volumes kept, "
                  f"~{gap/657_000/0.39*30/3600:.1f}h at 39% keep / 30s per item)")
        else:
            need_dl_tokens = target - p["manual_tokens"]
            gap = (need_dl_tokens * CHARS_PER_TOKEN[lang]["downloaded"]
                   / surv["downloaded"]) - m["downloaded"]
            print(f"      Supply is the binding constraint; the 20% ceiling "
                  f"({p['ceiling_tokens']/1e6:.0f}M) has room.")
            print(f"      need +{gap/1e6:.0f}M raw DOWNLOADED chars "
                  f"(raise BUDGETS in {lang}/collect/download_corpora.py)")
    return p


def main() -> None:
    """Report progress toward a token target under both survival scenarios."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lang", default="telugu", choices=["telugu", "nepali"])
    p.add_argument("--target", type=float, default=500,
                   help="Target tokens in millions.")
    p.add_argument("--survival", default="both",
                   choices=["both", "observed", "optimistic"],
                   help="Which manual-survival scenario to project. "
                        "Default shows both; plan against 'observed'.")
    args = p.parse_args()

    m = measure(args.lang)

    print(f"\n{'='*70}\n  {args.lang.upper()} -- progress toward "
          f"{args.target:.0f}M tokens\n{'='*70}\n")
    print("  ON DISK NOW (pre-cleaning)")
    dead = REJECTED_SOURCES.get(args.lang, set())
    for src, n in sorted(m["per_source"].items(), key=lambda kv: -kv[1]):
        flag = "  [rejected: not in target language]" if src in dead else ""
        print(f"    {src:24s}{n/1e6:>9.1f}M chars{flag}")
    print(f"    {'-'*33}")
    print(f"    {'manual (usable)':24s}{m['manual']/1e6:>9.1f}M")
    if m["manual_dead"]:
        print(f"    {'manual (rejected)':24s}{m['manual_dead']/1e6:>9.1f}M")
    print(f"    {'downloaded':24s}{m['downloaded']/1e6:>9.1f}M")

    scenarios = (["observed", "optimistic"] if args.survival == "both"
                 else [args.survival])
    for s in scenarios:
        report(args.lang, args.target, m, s)

    print("\n  Projections only. The number that counts is the manual token "
          "share in\n  token_counts.json after a real tokenizer run.\n")


if __name__ == "__main__":
    main()

# Monolingual Transformer LMs in Telugu and Nepali

Individual Project — Language Models and Agents, Monsoon 2026

Two **fully independent** decoder-only Transformer language models (~25M parameters
each), built from scratch in PyTorch:

| | Model H | Model L |
|---|---|---|
| Language | **Telugu** (తెలుగు) | **Nepali** (नेपाली) |
| Script | Telugu (Brahmic) | Devanagari |
| Resource tier | Higher-resource | Lower-resource (allowed list) |
| Corpus | Own — no overlap with Model L | Own — no overlap with Model H |
| Tokenizer | Own BPE, own vocabulary | Own BPE, own vocabulary |
| Weights | Independently initialized and trained | Independently initialized and trained |

The two models share **code** but never share **artifacts**. No document, token,
vocabulary entry, or weight crosses between them.

**Phase 1 results and methodology: [`report/phase1/phase1_report.md`](report/phase1/phase1_report.md).**

---

## Google Drive artifacts

> Large artifacts are not committed to git, per the project spec. Fill these in
> as each phase completes and keep permissions open so graders need not request access.

| Artifact | Link |
|---|---|
| Telugu corpus (cleaned, split) | https://drive.google.com/file/d/1Tq4bJVNKWD7w0NZQaGKoHItLPo0tDDoJ/view?usp=sharing |
| Nepali corpus (cleaned, split) | https://drive.google.com/file/d/1p6VeQASStY5Zq2QhDxoWSgaiFdlNaR6v/view?usp=drive_link |
| Telugu tokenized binaries | https://drive.google.com/file/d/1QvooM9ULBjNfydmfIxHH-h407MF49W4z/view?usp=drive_link |
| Nepali tokenized binaries | https://drive.google.com/file/d/1Jb2sP8OCDN_jttmy1xMB5jR8ZDQWbyu7/view?usp=drive_link |
| Model H pretrained checkpoint | Phase 2 |
| Model L pretrained checkpoint | Phase 2 |
| Model H finetuned checkpoint | Phase 3 |
| Model L finetuned checkpoint | Phase 3 |

---

## Repository layout

```
├── common/                  # Shared pipeline code (never shared data)
│   ├── text_norm.py         # Unicode NFC normalization, script-ratio filtering
│   ├── langid.py            # fastText language ID — separates Nepali from Hindi
│   ├── scraper.py           # Polite, resumable, sitemap-driven news crawler
│   ├── sitemap_discovery.py # Robust sitemap location
│   └── run_crawl.py         # Shared crawl entry point
├── telugu/                  # ── MODEL H ──────────────────────────────
│   ├── collect/             # Scrapers + HF corpus ingestion
│   ├── clean/               # Filtering, dedup, splits, statistics
│   ├── tokenizer/           # BPE training + vocab sweep; .model/.vocab files
│   └── configs/telugu.yaml  # Single source of truth for all Telugu decisions
├── nepali/                  # ── MODEL L ── (same structure, zero shared data)
│   ├── collect/
│   ├── clean/
│   ├── tokenizer/
│   └── configs/nepali.yaml
├── report/                  # Per-phase reports, tables, plots, heatmaps
│   └── phase1/
├── requirements.txt
└── README.md
```

---

## Where each stage runs, and why

This project has an awkward split: crawling needs **days of uptime but no GPU**,
while training needs **a GPU but only hours**. Free Colab preempts sessions, so
running a multi-day crawl there means restarting it constantly.

| Stage | Where | Why |
|---|---|---|
| News crawling | **Your local machine** | Runs for days uninterrupted; survives reboots via the SQLite state DB; needs no GPU. |
| HF corpus download + cleaning | **Colab** | Multi-GB streaming; uses Colab's ephemeral local disk, not Drive. |
| Tokenizer training | **Colab** | SentencePiece is memory-hungry. |
| Pretraining / finetuning | **Colab (T4)** | Needs the GPU. |

**Raw corpora are not committed to git** — they are tens of GB, and GitHub
rejects any file over 100MB. Cleaned splits and tokenized `uint16` binaries
(~1GB per language) are persisted to Drive and linked above; raw text stays
re-derivable from these scripts, which is what makes the project reproducible
rather than merely archived.

---

## Reproduction steps

### 0. Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

### 1. Start the crawlers — do this FIRST

The ≥20% manual-collection requirement is the long pole in Phase 1. Crawlers need
days of wall-clock time, so launch them before anything else and let them run in
the background.

```bash
# Build the URL frontier from each site's sitemaps (minutes)
python -m telugu.collect.scrape_news --discover
python -m nepali.collect.scrape_news --discover

# Fetch articles. Interrupt and re-run freely — progress is in data/*/state/*.db
python -m telugu.collect.scrape_news --run
python -m nepali.collect.scrape_news --run
```

Run the two in separate terminals so they proceed in parallel. Output accumulates
in `data/<lang>/raw/*.jsonl`, every record tagged `provenance="manual_scrape"`.

### 2. Collect books from archive.org

Public-domain OCR'd volumes, the larger half of the manual share. Search by
subject and collection, not by language metadata alone — see the Phase 1 report
§3.2 for why that distinction found an order of magnitude more text.

```bash
python collect_books_bysubject.py --lang telugu --target-chars 250000000
python collect_books_bysubject.py --lang nepali  --target-chars 250000000
# add --delay 3.0 --per-query 1500 if archive.org throttles with ConnectTimeout
```

Check progress at any point:

```bash
python scripts/check_target.py --lang nepali --target 500
```

That tool projects the post-cleaning token count under two manual-survival
scenarios and reports which constraint binds — supply, or the ≥20% manual rule
acting as a ceiling. Plan against the `observed` scenario.

### 3. Download the public corpora

```bash
python -m telugu.collect.download_corpora
python -m nepali.collect.download_corpora
python -m telugu.collect.download_corpora --scale 0.02   # smoke test
```

Budgets are sized to just reach the ratio cap and stop; the reasoning is in each
module's docstring. Raising them past that point buys zero tokens.

### 4. Clean, deduplicate, split

```bash
python -m telugu.clean.run_clean
python -m nepali.clean.run_clean
```

One streaming pass: normalize → filter → exact dedup → language ID → near dedup
→ split. Dedup runs *before* splitting and splits are document-level; both
choices prevent train/test leakage rather than serving efficiency.

Each run writes `data/<lang>/clean/{train,validation,test}.jsonl` plus
`clean_stats.json`. **Redirect to a local file, not straight to Drive** — a
long run piped through `tee` to a Drive mount can lose the pipe and take the
process with it:

```bash
python -m nepali.clean.run_clean 2>&1 | tee /content/nepali_clean.log
```

Runtime: ~75 min for Nepali, ~4.5 h for Telugu on a Colab CPU instance.

> If a run dies before writing `clean_stats.json`, the splits on disk are still
> complete. `python scripts/rebuild_clean_stats.py --lang <lang> --log <logfile>`
> recomputes every kept-side statistic exactly from them. Per-reason rejection
> counters cannot be recovered and are reported as unavailable — see report §9.1.

### 5. Train tokenizers and select the vocabulary

```bash
python -m telugu.tokenizer.train_tokenizer --sample-bytes 400000000
python scripts/reselect.py --lang telugu --vocab 16000 --yes

python -m nepali.tokenizer.train_tokenizer --sample-bytes 400000000
python scripts/reselect.py --lang nepali --vocab 16000 --yes
```

`--sample-bytes 400000000` is required: SentencePiece holds its training sample
in RAM, and the YAML default (500MB Nepali / 400MB Telugu) is sized for a machine
with more headroom than a free Colab instance.

`scripts/reselect.py` is also required. The sweep selects on fertility and UNK
rate, but `byte_fallback: true` makes the UNK rate identically 0.00 at every
vocabulary size, and fertility falls monotonically — so that criterion always
returns the largest candidate, 32,000. At `d_model=384` a 32k vocabulary spends
12.29M of the ~25M parameter budget on tied embeddings. `reselect.py` overrides
to 16,000 and records the full argument in `tokenizer_sweep.json`. See report §6.1.

### 6. Encode and build the report

```bash
python scripts/build_phase1_report.py
```

This encodes both corpora to `uint16` `.bin` files, writes `token_counts.json`
with the exact token counts and the manual/downloaded **token** split, generates
the figures, and writes `report/phase1/generated_summary.md`.

Pass `--skip-encode` to reuse existing `token_counts.json` files and only rebuild
the report and figures.

### 7. Token-frequency statistics and tokenization examples

```bash
python scripts/token_stats.py --lang telugu
python scripts/token_stats.py --lang nepali
```

Counts every token id across the encoded `train.bin`, writes
`report/phase1/stats/<lang>_token_frequency.json` and a Zipf/coverage figure, and
decodes sample documents back out of the binary so the printed examples are
provably what was encoded. Covers the vocabulary-size, token-frequency,
chars-per-token, examples and UNK statistics that Phase 1 §1.3 asks for.

---

## Phase 1 results

Full analysis: **[`report/phase1/phase1_report.md`](report/phase1/phase1_report.md)**.
Machine-generated numbers-only companion:
[`report/phase1/generated_summary.md`](report/phase1/generated_summary.md).

| | Model H — Telugu | Model L — Nepali |
|---|---|---|
| Total tokens | 674,176,616 | 511,303,554 |
| Training tokens | 666,507,564 | 501,331,128 |
| Manual token share | **26.87%** ✅ | **21.78%** ✅ |
| Documents after cleaning | 3,139,945 | 2,152,701 |
| Vocabulary | 16,000 | 16,000 |
| UNK rate (measured) | 0.00e+00 | 0.00e+00 |

Both exceed the ~500M target and both clear the ≥20% manual requirement, which
the spec states in **tokens** — and which is measured in tokens here, during the
encoding pass, not inferred from a character proxy.

---

---

## Phase branches

| Phase | Branch | Deadline | Marks |
|---|---|---|---|
| 1 — Data + Tokenizers | `phase-1` | 21 Aug 2026 | 25 |
| 2 — Model + Pretraining + Eval | `phase-2` | 5 Sep 2026 | 40 |
| 3 — Reasoning Finetune + Attention + Report | `phase-3` | 16 Sep 2026 | 35 |

Each branch is graded as of its own deadline. Work is pushed incrementally,
not dumped in a single final commit.

---

## Compliance with the spec's hard constraints

| Constraint | How it is satisfied |
|---|---|
| No pretrained models | Nothing is loaded from a checkpoint; both models train from random init. |
| No pretrained tokenizers | Both tokenizers trained from scratch with `sentencepiece`. |
| No HuggingFace Transformer classes | `transformers` is not a dependency. `datasets` is used **only** to stream raw text. |
| Two independent monolingual LMs | Separate configs, corpora, tokenizers, vocabularies, and weights. Enforced by directory separation and verified in the Phase 1 statistics report. |
| Student-implemented attention / positions / causal mask | Built from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only. |
| Reproducible | All randomness seeded; configs committed; every artifact regenerable from these scripts. |

### A note on shared `common/` code

`common/` holds pipeline *logic* (normalization, crawling, language ID) that both
languages invoke with their own configuration. This is not artifact sharing: each
language reads its own YAML, writes to its own data directory, trains its own
tokenizer, and produces its own weights. The constraint the spec protects — that
the two models are genuinely independent — is enforced at the data and parameter
level, which directory separation and the statistics report both verify.

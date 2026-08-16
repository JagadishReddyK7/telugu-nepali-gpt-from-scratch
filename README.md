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

---

## Google Drive artifacts

> Large artifacts are not committed to git, per the project spec. Fill these in
> as each phase completes and keep permissions open so graders need not request access.

| Artifact | Link |
|---|---|
| Telugu corpus (cleaned, split) | _TBD_ |
| Nepali corpus (cleaned, split) | _TBD_ |
| Telugu tokenized binaries | _TBD_ |
| Nepali tokenized binaries | _TBD_ |
| Model H pretrained checkpoint | _TBD_ |
| Model L pretrained checkpoint | _TBD_ |
| Model H finetuned checkpoint | _TBD_ |
| Model L finetuned checkpoint | _TBD_ |

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

**Raw corpora never go to Drive.** They are tens of GB and the free tier is 15GB.
Only tokenized `uint16` binaries (~1GB per language) and checkpoints are persisted;
raw text stays re-derivable from these scripts, which is what makes the project
reproducible rather than merely archived.

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

### 2. Download the public corpora

_Scripts land in step 2 of the build — see `telugu/collect/` and `nepali/collect/`._

### 3. Clean, deduplicate, split

_See `telugu/clean/` and `nepali/clean/`._

### 4. Train tokenizers

_See `telugu/tokenizer/` and `nepali/tokenizer/`._

---

## Phase branches

| Phase | Branch | Deadline | Marks |
|---|---|---|---|
| 1 — Data + Tokenizers | `phase-1` | 19 Aug 2026 | 25 |
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

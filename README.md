# Telugu & Nepali GPTs from Scratch

Two GPT-style language models with about 26M parameters each, one for **Telugu** (తెలుగు)
and one for **Nepali** (नेपाली), built end to end in PyTorch. The project covers the web
crawler and the 1.2B-token corpus, the tokenizers, the Transformer itself (attention
coded by hand), 500M-token pretraining runs, evaluation, an ablation, and a
reasoning finetune.

No pretrained models, no pretrained tokenizers, and no Hugging Face `transformers`.

**Python · PyTorch · SentencePiece · datasketch (MinHash-LSH) · fastText LID · Tesseract OCR · Kaggle/Colab T4**

## Results at a glance

| | Telugu | Nepali |
|---|---:|---:|
| Corpus after cleaning | 674M tokens · 3.1M docs | 511M tokens · 2.2M docs |
| Share collected by my own crawler + OCR | 26.9% | 21.8% |
| Parameters | 25.86M | 25.86M |
| Pretraining | 500M tokens · 4.5 h on one T4 | 500M tokens · 4.6 h on one T4 |
| **Test perplexity** | **60.3** | **41.7** |
| Test bits per byte (comparable across tokenizers) | 0.577 | 0.468 |
| Reasoning exact match, before → after finetune | 0.5% → 26.9% | 0.0% → 25.0% |

## What's in here

- **Data pipeline.** Corpus sources:
  - a polite, resumable, sitemap-driven news crawler (its SQLite state survives reboots)
  - OCR'd public-domain books from archive.org
  - streamed public corpora

  Cleaning steps: Unicode normalization, script-ratio filtering, fastText language ID to
  separate Nepali from Hindi, and exact plus **MinHash-LSH** near-duplicate removal.
  Deduplication runs *before* the document-level splits, so no text leaks between
  train and test. → [`common/`](common/), [Phase 1 report](report/phase1/phase1_report.md)
- **Tokenizers.** SentencePiece BPE, trained from scratch for each language. The vocabulary
  sweep favored 32k entries, but I chose 16k: at `d_model=384`, a 32k vocabulary
  would have spent 12.3M of the 25M-parameter budget on embeddings.
- **Model.** A decoder-only Transformer built only from `nn.Linear`, `nn.Embedding`,
  `nn.LayerNorm` and `nn.Dropout`. Multi-head causal self-attention, the causal mask
  and positional embeddings are all written by hand. It has 11 layers, `d_model` 384,
  6 heads, a 512-token context and tied embeddings.
  → [`common/model.py`](common/model.py)
- **Training.** AdamW with cosine decay and warmup, fp16 mixed precision with
  GradScaler, and gradient accumulation. Training is **crash-safe**: checkpoints store
  the weights, both AdamW moments, the scaler state and every RNG state, so
  resuming is bit-exact, and a test verifies it.
  → [`common/train.py`](common/train.py)
- **Evaluation.** Perplexity, bits per byte (the fair comparison across different
  tokenizers), BLEU, chrF++ and ROUGE-L, plus attention analysis: heatmaps, entropy,
  attention distance and head roles. → [Phase 2 report](report/phase2/phase2_report.md)
- **Ablation.** Removing positional embeddings costs only 0.041 nats/token (~1%
  bits per byte) and trains 8.3% faster. The causal mask already gives the model a
  positional signal.
- **Reasoning finetune.** A generator for synthetic comparative-reasoning data in both
  languages (transitive chains, numeric comparisons, "cannot be determined"
  cases), trained with answer-only loss. Exact match rose from ~0% to ~26%. Comparing
  attention before and after finetuning shows the gain comes from answer format and
  entity lookup, not a learned comparison procedure.
  → [Phase 3 report](report/phase3/phase3_report.md)
- **Tests.** 40 pytest tests, including ones that check the causal mask hides future
  tokens, that training resumes bit-exactly, and that no entities leak between data splits.

**Full write-up:** [`report/final_report/final_report.pdf`](report/final_report/final_report.pdf)

## Background

I built this alone for the *Language Models and Agents* course (Monsoon 2026), in
three phases: data and tokenizers, then model, pretraining and evaluation, then the
reasoning finetune and attention study. The `phase-1`, `phase-2` and `phase-3` branches
keep a snapshot of the code at the end of each phase.

In the detailed sections below, **Model H** is the Telugu model (higher-resource)
and **Model L** is the Nepali model (lower-resource). The two models share code
but never share data, vocabulary or weights.

---

## Google Drive artifacts

> Large artifacts (corpora, tokenized binaries, checkpoints) are kept out of git
> and hosted on Google Drive. All links are open, so no access request is needed.

| Artifact | Link |
|---|---|
| Telugu corpus (cleaned, split) | https://drive.google.com/file/d/1Tq4bJVNKWD7w0NZQaGKoHItLPo0tDDoJ/view?usp=sharing |
| Nepali corpus (cleaned, split) | https://drive.google.com/file/d/1p6VeQASStY5Zq2QhDxoWSgaiFdlNaR6v/view?usp=drive_link |
| Telugu tokenized binaries | https://drive.google.com/file/d/1QvooM9ULBjNfydmfIxHH-h407MF49W4z/view?usp=drive_link |
| Nepali tokenized binaries | https://drive.google.com/file/d/1Jb2sP8OCDN_jttmy1xMB5jR8ZDQWbyu7/view?usp=drive_link |
| Model H pretrained checkpoint — `best.pt` | https://drive.google.com/file/d/1X2H_1newxrC2tzLq_GBRuTCl5h2u54Hb/view?usp=drive_link |
| Model H pretrained checkpoint — `latest.pt` | https://drive.google.com/file/d/1-YkgfnZGoUuEchN55d2ek79AjtzOxApu/view?usp=drive_link |
| Model L pretrained checkpoint — `best.pt` | https://drive.google.com/file/d/1NRc9J41AFgg_toV7Kzd_16gfESv7aLM5/view?usp=sharing |
| Model L pretrained checkpoint — `latest.pt` | https://drive.google.com/file/d/1xccvCBYuj0f5t0b7mv4asQONax4w9NEa/view?usp=drive_link |
| Ablation (no positional emb.) — `best.pt` | https://drive.google.com/file/d/1CtL5VKZj1wZd3r8STwqK0hv7H0mOywLo/view?usp=drive_link |
| Ablation (no positional emb.) — `latest.pt` | https://drive.google.com/file/d/14vhOJ-xgheIxp0vt-cYKfLTim9Un0esG/view?usp=drive_link |
| Model H reasoning-finetuned checkpoint — `best.pt` | https://drive.google.com/file/d/1NtRRvAnG8pD1-PDFIXp2kOTSNuOOR60y/view?usp=drive_link |
| Model H reasoning-finetuned checkpoint — `latest.pt` | https://drive.google.com/file/d/1wxfRUUJlusX0SCZgMdeWXUpmkY9qhu2c/view?usp=drive_link |
| Model L reasoning-finetuned checkpoint — `best.pt` | https://drive.google.com/file/d/15qlhFoMqbJVC_JlfSQcIs7G0f60l-BOO/view?usp=drive_link |
| Model L reasoning-finetuned checkpoint — `latest.pt` | https://drive.google.com/file/d/1kX62B3lxp8W687tDEO0ZIXAAsasW9jup/view?usp=drive_link |

Finetuned checkpoints use the same resume-capable format as the pretrained
ones (weights, both AdamW moments, GradScaler, step, tokens, full config, RNG
state) plus the epoch / batch position of the finetune; ~310 MB each. Place
them at `checkpoints/telugu_reasoning/best.pt` and
`checkpoints/nepali_reasoning/best.pt` to run the Phase 3 evaluation. The
synthetic reasoning datasets are small and are committed in the branch
(`<lang>/reasoning/dataset/`), so they need no Drive link.

---

## Repository layout

```
├── common/                  # Shared pipeline code (never shared data)
│   ├── text_norm.py         # Unicode NFC normalization, script-ratio filtering
│   ├── langid.py            # fastText language ID — separates Nepali from Hindi
│   ├── scraper.py           # Polite, resumable, sitemap-driven news crawler
│   ├── model.py             # Decoder-only Transformer from primitives (Phase 2)
│   ├── train.py             # Resumable pretraining loop
│   ├── evaluate.py          # Cross-entropy / perplexity / bits-per-byte
│   ├── generate.py          # BLEU / chrF++ / ROUGE-L, diversity diagnostics
│   ├── attention.py         # Heatmaps, entropy, distance, head labels
│   ├── reasoning_gen.py     # Synthetic reasoning-data engine (Phase 3)
│   ├── finetune.py          # Resumable reasoning finetune
│   ├── reasoning_eval.py    # Exact match + scored-choice accuracy
│   └── attention_compare.py # Pretrained-vs-finetuned attention on the same prompts
├── telugu/                  # ── MODEL H ──────────────────────────────
│   ├── collect/             # Scrapers + HF corpus ingestion
│   ├── clean/               # Filtering, dedup, splits, statistics
│   ├── tokenizer/           # BPE training + vocab sweep; .model/.vocab files
│   ├── reasoning/           # Telugu lexicon + generator; dataset/ (val, test committed)
│   └── configs/             # telugu.yaml, model.yaml, model_nopos.yaml, finetune.yaml
├── nepali/                  # ── MODEL L ── (same structure, zero shared data)
│   ├── collect/
│   ├── clean/
│   ├── tokenizer/
│   ├── reasoning/
│   └── configs/
├── scripts/                 # run_eval.py (Phase 2), run_phase3.py, plotting, Kaggle staging
├── tests/                   # 40 tests: model, training/resume, reasoning/finetune
├── notebooks/               # Kaggle notebook that produced the Phase 3 numbers
├── report/                  # Per-phase reports, tables, plots, heatmaps
│   ├── phase1/
│   ├── phase2/
│   └── phase3/              # final consolidated report lives here
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
git clone https://github.com/JagadishReddyK7/telugu-nepali-gpt-from-scratch.git
cd telugu-nepali-gpt-from-scratch
pip install -r requirements.txt
```

### 1. Start the crawlers — do this FIRST

The ≥20% self-collected share is the slowest step in Phase 1. Crawlers need
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
chars-per-token, examples and UNK statistics.

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

Both exceed the ~500M-token target, and both keep the self-collected share above
20%. That share is measured in **tokens** during the encoding pass, not inferred
from character counts.

---

## Phase 2 results

Full analysis: **[`report/phase2/phase2_report.md`](report/phase2/phase2_report.md)**.
Machine-generated tables: [`report/phase2/stats/comparison.md`](report/phase2/stats/comparison.md).

| | Model H — Telugu | Model L — Nepali | Ablation — no pos. emb. |
|---|---:|---:|---:|
| Parameters | 25,860,480 | 25,860,480 | 25,663,872 |
| Training tokens consumed | 500,006,912 | 500,006,912 | 500,006,912 |
| Wall clock (one Kaggle T4) | 4.50 h | 4.58 h | 4.15 h |
| Mean throughput | 31,130 tok/s | 30,558 tok/s | 33,720 tok/s |
| Test cross-entropy (nats/token) | 4.09868 | 3.73057 | 4.13979 |
| Test perplexity | 60.26 | 41.70 | 62.79 |
| **Test bits per byte** | **0.5770** | **0.4675** | **0.5827** |
| chrF++ (t = 1.0) | 17.03 | 19.41 | 17.02 |
| BLEU-4 (t = 0.5) | 0.4133 | 4.1533 | 0.2109 |

Perplexity is per *token* and the two tokenizers differ (10.25 vs 11.51 UTF-8
bytes per token), so bits-per-byte is the comparable figure. On it, the
lower-resource model is 19% better — analysed in §5 of the Phase 2 report.

The ablation retrains Model H with `positional: none`. Removing the
learned positional table costs only **0.041 nats** (0.99% of bits per byte)
while running 8.3% faster, because causal masking is itself a positional
signal — position *t* attends to exactly *t*+1 keys. The attention analysis
shows the model's three-stage layer organization surviving intact but shifting
two to three layers deeper, which is where the small penalty comes from. See §6.


Every checkpoint is resume-capable: model weights, optimizer state (both AdamW
moments), GradScaler state, step, tokens seen, full config, and NumPy /
CPU-torch / CUDA RNG state. Each file is ~310 MB — 25,860,480 parameters x 4
bytes x 3 (weights plus both AdamW moments); the ablation's are 308 MB,
the 196,608-parameter positional table being absent. A `.pt` is itself a ZIP container; **do not extract it**,
`torch.load` reads it directly.

### Reproduction, continued from step 7

### 8. Pretrain the two models

About 4.5 h per model on one T4. Runs are independent and may be done in either
order.

```bash
python -m common.train --config telugu/configs/model.yaml
python -m common.train --config nepali/configs/model.yaml

# resume after an interrupted session (bit-exact, not approximate)
python -m common.train --config telugu/configs/model.yaml --resume
```

### 9. Evaluate

Skip step 8 by downloading the checkpoints above into
`checkpoints/telugu/best.pt` and `checkpoints/nepali/best.pt`.

```bash
pip install sacrebleu

# PPL / BPB over the whole split, generation metrics, attention analysis
python scripts/run_eval.py --runs telugu nepali --n-prompts 200

# training curves from the committed logs — needs no GPU and no checkpoints
python scripts/plot_training.py
```

Outputs land in `report/phase2/stats/` (JSON plus `comparison.md`) and
`report/phase2/figures/` (PNG).

### 10. Tests

```bash
python -m pytest tests/ -q     # 40 pass: 15 model, 9 training / checkpointing, 16 reasoning / finetune
# or, without pytest:
python tests/test_model.py && python tests/test_train.py && python tests/test_reasoning.py
```

Includes `test_causal_mask_blocks_the_future` (changing token t+1 leaves the
logits at position t bit-identical), `test_resume_is_bit_exact`, and for
Phase 3 `test_labels_are_correct_by_independent_derivation`,
`test_no_entity_leakage_between_splits` and `test_finetune_resume_is_bit_exact`.

### 11. Ablation: no positional embeddings

Retrains Model H with the positional embedding table removed. Same corpus,
tokenizer, seed and schedule — only `positional: none` differs, so any
difference is attributable to the embeddings alone.

```bash
python -m common.train --config telugu/configs/model_nopos.yaml
python scripts/run_eval.py --runs telugu_nopos --n-prompts 200
```

---

## Phase 3 results

Full analysis and the consolidated final report:
**[`report/phase3/phase3_report.md`](report/phase3/phase3_report.md)**.
Machine-generated tables: [`report/phase3/stats/comparison.md`](report/phase3/stats/comparison.md).

| | Model H — Telugu | Model L — Nepali |
|---|---:|---:|
| Synthetic reasoning set (train / val / test) | 6,000 / 500 / 1,000 | 6,000 / 500 / 1,000 |
| Test exact match — pretrained | 0.005 | 0.000 |
| Test exact match — **finetuned** | **0.269** | **0.250** |
| Scored-choice accuracy, finetuned (chance 0.32) | 0.361 | 0.364 |
| Exact match on held-out phrasing / 3-hop chains | 0.252 / 0.193 | 0.227 / 0.149 |
| Test bits per byte after finetuning (was 0.577 / 0.468) | 0.959 | 0.926 |
| Finetune wall clock (one Kaggle T4) | 341 s | 362 s |

Each model was finetuned on its own synthetic comparative-reasoning set —
transitive chains, numeric comparisons, chained relations (including
"cannot be determined"), yes/no, and full orderings, in Telugu and Nepali
script with held-out entity names and held-out phrasings in the test split.
Finetuning taught both models the task format (0% → ~60% well-formed answers)
and lifted exact match from ~0 to ~26%, with the two resource tiers
indistinguishable; the report's error analysis and the pretrained-vs-finetuned
attention comparison show that the gain is format and entity-lookup rather
than a comparison procedure, and quantify the language-modelling cost of
answer-only finetuning. See §3–§5 of the report.

### Reproduction, continued from step 11

### 12. Generate the reasoning datasets

Deterministic (seeds 1337 / 2718); validation and test are also committed.

```bash
python -m telugu.reasoning.generate
python -m nepali.reasoning.generate
```

### 13. Finetune each model from its own pretrained checkpoint

Place the Phase 2 checkpoints at `checkpoints/telugu/best.pt` and
`checkpoints/nepali/best.pt` (Drive links above). About 6 minutes per model
on one T4.

```bash
python -m common.finetune --config telugu/configs/finetune.yaml
python -m common.finetune --config nepali/configs/finetune.yaml

# clean stop after N steps, then bit-exact resume
python -m common.finetune --config telugu/configs/finetune.yaml --session-steps 200
python -m common.finetune --config telugu/configs/finetune.yaml --resume
```

### 14. Evaluate Phase 3

Skip step 13 by placing the finetuned checkpoints (Drive links above) at
`checkpoints/telugu_reasoning/best.pt` and `checkpoints/nepali_reasoning/best.pt`.

```bash
# reasoning accuracy for pretrained and finetuned checkpoints, LM retention
# on the Phase 1 test split, pretrained-vs-finetuned attention comparison,
# tables and figures
python scripts/run_phase3.py --langs telugu nepali

# tables and figures only, from the committed JSON and logs (no GPU needed)
python scripts/run_phase3.py --tables-only
```

Outputs land in `report/phase3/stats/` (JSON, Markdown, `comparison.md`),
`report/phase3/figures/` (PNG) and `report/phase3/logs/` (finetune JSONL).

### 15. On Kaggle

`notebooks/lma_phase3_kaggle.ipynb` runs steps 12–14 in one committed session
(~17 minutes on a T4). It discovers its three inputs by content — the code
(`common/finetune.py`), the pretrained checkpoints (`<lang>/best.pt`) and the
tokenized corpora — so no Kaggle paths need editing;
`scripts/stage_code_for_kaggle.py` builds the code upload.

---

## Phase branches

`main` holds the complete project. Each phase built on the one before it, and
its branch keeps a snapshot of the code at the end of that phase:

| Phase | Branch | Scope |
|---|---|---|
| 1 | `phase-1` | Data collection, cleaning, tokenizers |
| 2 | `phase-2` | Model, pretraining, evaluation, ablation |
| 3 | `phase-3` | Reasoning finetune, attention comparison, final report |

---

## Design constraints

| Constraint | How it is satisfied |
|---|---|
| No pretrained models | Nothing is loaded from a checkpoint; both models train from random init. |
| No pretrained tokenizers | Both tokenizers trained from scratch with `sentencepiece`. |
| No HuggingFace Transformer classes | `transformers` is not a dependency. `datasets` is used **only** to stream raw text. |
| Two independent monolingual LMs | Separate configs, corpora, tokenizers, vocabularies, and weights. Enforced by directory separation and verified in the Phase 1 statistics report. |
| Hand-written attention / positions / causal mask | Built from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only. |
| Reproducible | All randomness seeded; configs committed; every artifact regenerable from these scripts. |

### A note on shared `common/` code

`common/` holds pipeline *logic* (normalization, crawling, language ID) that both
languages invoke with their own configuration. This is not artifact sharing: each
language reads its own YAML, writes to its own data directory, trains its own
tokenizer, and produces its own weights. The two models' independence is enforced at
the data and parameter level, which directory separation and the statistics report both verify.

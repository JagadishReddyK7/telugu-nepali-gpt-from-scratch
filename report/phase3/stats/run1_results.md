# Phase 3 — run 1 (superseded), for the record

The reported Phase 3 numbers come from **run 2** (`comparison.md`, the logs and
figures in this directory). Run 1 is the first finetune attempt; it is kept here
because the final report (§2.3, §3.5) uses it as a controlled contrast, and its
artifacts were not retained (the Kaggle version failed at a later cell and
Kaggle discards the output of a failed version). Numbers are transcribed from
that version's log.

## Configuration differences

| | run 1 | run 2 (reported) |
|---|---|---|
| training examples | 30,000 (val 2,000, test 3,000) | 6,000 (val 500, test 1,000) |
| entity-name pool per language | 76 simple given names | 540 (Telugu) / 436 (Nepali), incl. compound names |
| loss | full sequence (`prompt_loss_weight: 1.0`) | answer tokens only (`0.0`) |
| learning rate | 1e-4 → 1e-5 | 3e-4 → 3e-5 |
| epochs · steps | 8 · 1,176 (Te) / 1,288 (Ne) | 30 · 990 (Te) / 1,050 (Ne) |
| eval cadence | every 50 steps, 400 val examples | every 50 steps, whole val split |

## Run 1 outcome

| | Telugu | Nepali |
|---|---:|---:|
| tokens · wall clock | 9.24M · 387 s | 10.07M · 444 s |
| final train loss (full sequence) | 0.63 | 0.63 |
| validation loss / answer loss, final | 3.273 / 3.536 | 3.326 / 3.683 |
| validation exact match: best (step) → final | 0.1675 (100) → 0.1525 | 0.195 (~250) → 0.1675 |
| validation exact-match curve (every 50 steps) | 0.000, 0.138, 0.168, 0.148, 0.135, 0.148, 0.155, 0.160, 0.155, 0.163, 0.145, 0.143, 0.140, 0.133, 0.160, 0.140, 0.145, 0.163, 0.148, 0.133, 0.148, 0.135, 0.138, 0.143, 0.153 | 0.000, 0.173, 0.173, 0.180, 0.173, 0.193, 0.188, 0.185, 0.195, 0.195, 0.160, 0.175, 0.183, 0.170, 0.150, 0.150, 0.190, 0.150, 0.160, 0.165, 0.188, 0.170, 0.180, 0.170, 0.160, 0.178, 0.168 |
| **test exact match, pretrained → finetuned** | 0.0027 → **0.1497** | — (run failed before the Nepali evaluation) |
| test scored-choice, pretrained → finetuned (chance 0.32) | 0.293 → 0.272 | — |
| test exact match by task, finetuned | superlative 0.005 · numeric 0.085 · chain_relation 0.160 · yes/no 0.498 · ordering 0.000 | — |
| LM after finetuning: validation loss / ppl / BPB | 4.695 / 109.4 / 0.656 | — |
| LM after finetuning: test loss / ppl / BPB | 4.722 / 112.4 / 0.665 (pretrained 4.099 / 60.3 / 0.577) | — |

## Diagnosis

Exact match was flat from step 50, and on the held-out-name test split the
model answered name questions with names from the *training* pool that did
not appear in the prompt (`superlative_chain` 0.5%, against ~33% for picking
any prompt name at random). With 50 training names each was the gold answer
about 200 times, so the model learned the distribution of answers instead of
copying the correct entity from the premises. Full-sequence loss reinforces
that: predicting the first mention of a random name inside the premises is
possible only through a prior over names. Run 2's changes — a ~6× larger,
compositional name pool and answer-only loss — follow from this, and fixed the
symptom (Telugu superlative 0.5% → 31.5%). The forgetting comparison (run 1
test BPB 0.665 vs run 2 0.959) is the other side of that trade.

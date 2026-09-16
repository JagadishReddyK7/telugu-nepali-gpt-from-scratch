# Attention example — nepali_finetuned_reasoning

Heatmap axes are labelled with token **positions**; this table maps
each position to its token. Matplotlib ships no Telugu or Devanagari
glyphs, so token strings cannot be drawn on the figure axes.

## Example passage (reasoning test prompt, with its gold answer)

> किताब चस्माभन्दा सस्तो छ। चस्माको मूल्य गाग्रीको मूल्यभन्दा कम छ। यीमध्ये सबैभन्दा बढी मूल्य केको छ? उत्तर: गाग्री

## Positions 0–33

| pos | token | pos | token | pos | token | pos | token |
|----:|:------|----:|:------|----:|:------|----:|:------|
| 0 | `<s>` | 1 | `·किताब` | 2 | `·च` | 3 | `स्मा` |
| 4 | `भन्दा` | 5 | `·सस्तो` | 6 | `·छ` | 7 | `।` |
| 8 | `·च` | 9 | `स्` | 10 | `माको` | 11 | `·मूल्य` |
| 12 | `·गा` | 13 | `ग्र` | 14 | `ीको` | 15 | `·मूल्य` |
| 16 | `भन्दा` | 17 | `·कम` | 18 | `·छ` | 19 | `।` |
| 20 | `·यी` | 21 | `मध्ये` | 22 | `·सबैभन्दा` | 23 | `·बढी` |
| 24 | `·मूल्य` | 25 | `·क` | 26 | `ेको` | 27 | `·छ` |
| 28 | `?` | 29 | `·उत्तर` | 30 | `:` | 31 | `·गा` |
| 32 | `ग्री` | 33 | `</s>` | | | | |

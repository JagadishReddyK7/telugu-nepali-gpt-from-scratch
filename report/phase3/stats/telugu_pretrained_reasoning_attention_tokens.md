# Attention example — telugu_pretrained_reasoning

Heatmap axes are labelled with token **positions**; this table maps
each position to its token. Matplotlib ships no Telugu or Devanagari
glyphs, so token strings cannot be drawn on the figure axes.

## Example passage (reasoning test prompt, with its gold answer)

> భవలత జ్యోతి కంటే చిన్నవారు. భవలత వయసు శ్రీశ్రీ వయసు కంటే ఎక్కువ. అందరికంటే పెద్దవారు ఎవరు? జవాబు: జ్యోతి

## Positions 0–29

| pos | token | pos | token | pos | token | pos | token |
|----:|:------|----:|:------|----:|:------|----:|:------|
| 0 | `<s>` | 1 | `·భ` | 2 | `వల` | 3 | `త` |
| 4 | `·జ్యోతి` | 5 | `·కంటే` | 6 | `·చిన్న` | 7 | `వారు` |
| 8 | `.` | 9 | `·భ` | 10 | `వల` | 11 | `త` |
| 12 | `·వయసు` | 13 | `·శ్రీ` | 14 | `శ్రీ` | 15 | `·వయసు` |
| 16 | `·కంటే` | 17 | `·ఎక్కువ` | 18 | `.` | 19 | `·అంద` |
| 20 | `రిక` | 21 | `ంటే` | 22 | `·పెద్ద` | 23 | `వారు` |
| 24 | `·ఎవరు` | 25 | `?` | 26 | `·జవాబు` | 27 | `:` |
| 28 | `·జ్యోతి` | 29 | `</s>` | | | | |

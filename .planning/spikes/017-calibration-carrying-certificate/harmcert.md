# Spike 017, stage C: what a harm constraint can certify

## 1. Labelling the candidate's own responses

Labels needed for the Clopper-Pearson bound to certify `tau` with probability 0.8 (delta 0.05), and the PPI++ gain `1 / (1 - rho^2)` each judge would add at that true rate (a gain of 1.02 saves 2% of the labels).

| tau | true rate | labels needed | expected positives among them | gain: compiled rubric | gain: Qwen3Guard-4B | gain: gated, recall 0.5 | gain: gated, recall 0.8 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.0000 | 301 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.0025 | 776 | 1.9 | 1.00 | 1.00 | 1.04 | 1.10 |
| 0.01 | 0.0050 | 1957 | 9.8 | 1.01 | 1.00 | 1.07 | 1.20 |
| 0.02 | 0.0000 | 149 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.02 | 0.0050 | 388 | 1.9 | 1.01 | 1.00 | 1.07 | 1.20 |
| 0.02 | 0.0100 | 978 | 9.8 | 1.01 | 1.00 | 1.14 | 1.38 |
| 0.05 | 0.0000 | 59 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.05 | 0.0125 | 154 | 1.9 | 1.01 | 1.00 | 1.16 | 1.46 |
| 0.05 | 0.0250 | 388 | 9.7 | 1.03 | 1.00 | 1.28 | 1.82 |

The rule of three: with no positive among n labels the bound is `1 - 0.05^(1/n)`, about 3 / n: 0.0132 at the sheet's 225, 0.0100 at 299.

## 2. Carrying a calibration with k human positives

Candidate: 2000 judged responses at true harm rate `r`; calibration: k positives and 222 negatives, the carried rates exactly right (best case). Median bound, and the share of draws that certify tau = 0.05. `Youden` for the ungated judges; `answer-aware` (answer rate 0.33, the screen's) for the gated label.

| judge | route | true rate | k = 3 | k = 10 | k = 30 | k = 100 | k = 300 |
|---|---|---|---|---|---|---|---|
| compiled rubric | Youden | 0.0 | 1.000 (0.00) | 0.526 (0.01) | 0.254 (0.01) | 0.192 (0.01) | 0.171 (0.02) |
| compiled rubric | Youden | 0.013 | 1.000 (0.00) | 0.573 (0.00) | 0.278 (0.00) | 0.207 (0.01) | 0.186 (0.01) |
| Qwen3Guard-4B | Youden | 0.0 | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 0.757 (0.00) |
| Qwen3Guard-4B | Youden | 0.013 | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 1.000 (0.00) | 0.765 (0.00) |
| gated, recall 0.5 | answer-aware | 0.0 | 0.224 (0.10) | 0.058 (0.41) | 0.032 (0.87) | 0.024 (1.00) | 0.022 (1.00) |
| gated, recall 0.5 | answer-aware | 0.013 | 0.353 (0.00) | 0.110 (0.09) | 0.060 (0.30) | 0.045 (0.68) | 0.040 (0.91) |
| gated, recall 0.8 | answer-aware | 0.0 | 0.056 (0.44) | 0.023 (0.96) | 0.016 (1.00) | 0.013 (1.00) | 0.013 (1.00) |
| gated, recall 0.8 | answer-aware | 0.013 | 0.124 (0.00) | 0.054 (0.42) | 0.037 (0.91) | 0.032 (1.00) | 0.030 (1.00) |

Largest miss rate (bound below the true rate 0.013) over judges and k: 0.003.

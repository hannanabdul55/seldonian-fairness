# What a human-terms certificate costs

Measured on the first sheet (one annotator): strict refusal 0.107 for the reference, 0.106 for the trained policy (difference -0.001). Correlation of two single responses to the same prompt across the two policies, by the guard's flag on 490 prompts: 0.66. Guard rho^2 0.6. One-sided test at delta 0.05, normal-type limit.

## Prompt pairs needed to certify a margin

| margin | chance of certifying | labels alone, unpaired | labels alone, paired by prompt | paired, guard's logit as predictor |
|---|---|---|---|---|
| 0.02 | 50% | 1,226 | 416 | 167 |
| 0.02 | 80% | 2,801 | 950 | 380 |
| 0.03 | 50% | 554 | 188 | 76 |
| 0.03 | 80% | 1,266 | 429 | 172 |
| 0.05 | 50% | 203 | 69 | 28 |
| 0.05 | 80% | 462 | 157 | 63 |

## Margin a sample can be expected to certify

| pairs labelled | labels alone, unpaired | labels alone, paired by prompt | paired, guard's logit as predictor |
|---|---|---|---|
| 150 | 0.058 | 0.034 | 0.021 |
| 300 | 0.041 | 0.024 | 0.015 |
| 600 | 0.029 | 0.017 | 0.010 |
| 1,000 | 0.022 | 0.013 | 0.008 |

# What a human-terms certificate costs

Measured on the first sheet (one annotator): strict refusal 0.107 for the reference, 0.106 for the trained policy (difference -0.001). Correlation of two single responses to the same prompt across the two policies, by the guard's flag on 490 prompts: 0.66. Guard rho^2 0.6. One-sided test at delta 0.05, normal-type limit.

## Prompt pairs needed to certify a margin

| margin | chance of certifying | labels alone, unpaired | labels alone, paired by prompt | paired, with the guard, guard's mean known | paired, with the guard, other pool prompts only |
|---|---|---|---|---|---|
| 0.02 | 50% | 1,226 | 416 | 167 | 339 |
| 0.02 | 80% | 2,801 | 950 | 380 | not within 490 prompts |
| 0.03 | 50% | 554 | 188 | 76 | 98 |
| 0.03 | 80% | 1,266 | 429 | 172 | 362 |
| 0.05 | 50% | 203 | 69 | 28 | 30 |
| 0.05 | 80% | 462 | 157 | 63 | 78 |

## Margin a sample can be expected to certify

| pairs labelled | labels alone, unpaired | labels alone, paired by prompt | paired, with the guard, guard's mean known | paired, with the guard, other pool prompts only |
|---|---|---|---|---|
| 150 | 0.058 | 0.034 | 0.021 | 0.026 |
| 300 | 0.041 | 0.024 | 0.015 | 0.021 |
| 600 | 0.029 | 0.017 | 0.010 | more than 490 prompts |
| 1,000 | 0.022 | 0.013 | 0.008 | more than 490 prompts |

The two guard columns answer different questions. "Guard's mean known": the rate on the pool's own 490 prompts, with enough guard-only responses that the guard's mean carries no error (P9 draws 8 per policy per prompt). "Other pool prompts only": the rate on new prompts from the same source, where the 490 - n unlabelled pool prompts are all the guard has. The betting bound that P9 uses for labels alone is wider than the normal-type limit in the "paired by prompt" column; `results/labels/p9/design_check.md` has the simulated margins.

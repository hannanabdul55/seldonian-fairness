# P9 design check (synthetic labels on the real guard scores; no human P9 label read)

Guard scores: 490 prompts, 1 + 8 responses per policy. Synthetic strict-refusal labels drawn from the guard's logit with the first sheet's rates per logit bin (below 0: 0.002; 0 to 10: 0.165; above 10: 0.762). In this synthetic population the reference rate is 0.095, the trained rate 0.101, the true difference +0.0063. 2000 repetitions of the design with 300 labelled pairs, delta 0.05, 300 bootstrap draws each.

| how the prompts arise | limit | miss rate (limit below the truth) | mean certified margin | share at or under 0.02 |
|---|---|---|---|---|
| pool | (a) labels alone, betting bound | 0.001 | 0.065 | 0.01 |
| pool | (a') labels alone, bootstrap-t | 0.040 | 0.037 | 0.17 |
| pool | (b1) with the guard, pool rate | 0.045 | 0.032 | 0.22 |
| pool | (b2) with the guard, new prompts | 0.042 | 0.035 | 0.18 |
| new prompts | (a) labels alone, betting bound | 0.001 | 0.065 | 0.01 |
| new prompts | (a') labels alone, bootstrap-t | 0.051 | 0.037 | 0.19 |
| new prompts | (b1) with the guard, pool rate | 0.063 | 0.032 | 0.23 |
| new prompts | (b2) with the guard, new prompts | 0.052 | 0.035 | 0.20 |

A valid limit misses at most 0.05 of the time (Monte Carlo se 0.005). "pool": the 490 prompts are fixed and the claim is about them. "new prompts": the 490 are redrawn each time, so the claim is about the source they came from; (b1) is not built for that row.

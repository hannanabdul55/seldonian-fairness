# The mid-rate confirmation pool (plan step R5b): preview

`scripts/confirm_mid.py`. A PREVIEW on seen data: step R5's judged responses, re-mixed in this pool's proportions. It tests the code and gives the registration its expectations; it confirms nothing. One pool of 400 OR-Bench prompts, every unused hard-1K prompt topped up with unused benign prompts of the 80K set; Granite-3.3-2B, 8 reference responses a prompt for the strata and 16 of the trained policy; Qwen3Guard-4B labels. Checks: `replacement_check.py` (5,000 stored draws, 40,000 with replacement) and `twophase_check.py` (10,000), called unchanged.

## 1. The labels

| label | rate, trained policy | rate, reference | class | ICC of the reference | prompts with a reference rate of 0 or 1 | pre-flight ESS |
|---|---|---|---|---|---|---|
| refusal | 0.478 | 0.454 | mid | 0.82 | 76% | 2.93 |
| unsafe | 0.018 | 0.026 | rare | 0.41 | 92% | 1.29 |

What the mixture is made of (the trained policy's 16 responses a prompt):

| label | source | prompts | rate | reference rate | prompts at a rate of 0 | at 1 |
|---|---|---|---|---|---|---|
| refusal | hard-1k | 193 | 0.839 | 0.803 | 6% | 55% |
| refusal | 80k | 207 | 0.137 | 0.129 | 73% | 4% |
| unsafe | hard-1k | 193 | 0.028 | 0.039 | 85% | 0% |
| unsafe | 80k | 207 | 0.010 | 0.014 | 97% | 0% |

The source alone explains much of what the strata gain on the refusal label. The variance of a random draw over that of a proportional stratified draw is 1.98 with two strata, the two sources, and 3.09 with the 8 strata of the reference rate. Share of hard-1K prompts in the 8 strata, lowest reference rate first: 10%, 10%, 11%, 24%, 54%, 93%, 91%, 92%.

## 2. Validity with a large pool (strata fixed, drawn with replacement, 40,000 draws)

Miss rates; **bold** is over the level, `?` unresolved.

| label | n_s | delta | b1w | Wald-t b1 | StratPPI estimator, bootstrap-t | StratPPI, normal limit | pooled Wilson | Clopper-Pearson |
|---|---|---|---|---|---|---|---|---|
| refusal (0.48) | 100 | 0.05 | 0.051? | 0.026 | 0.052? | **0.075** | 0.048 | 0.048 |
| refusal (0.48) | 100 | 0.1 | 0.102? | 0.072 | **0.103** | **0.132** | **0.104** | 0.071 |
| refusal (0.48) | 200 | 0.05 | 0.051? | 0.039 | 0.050 | **0.063** | **0.057** | 0.042 |
| refusal (0.48) | 200 | 0.1 | 0.102? | 0.087 | 0.100? | **0.119** | 0.097 | 0.097 |
| unsafe (0.02) | 100 | 0.05 | 0.000 | 0.000 | 0.018 | **0.203** | 0.000 | 0.000 |
| unsafe (0.02) | 100 | 0.1 | **0.153** | 0.000 | 0.028 | **0.240** | **0.152** | 0.000 |
| unsafe (0.02) | 200 | 0.05 | 0.021 | 0.021 | 0.005 | **0.147** | 0.026 | 0.026 |
| unsafe (0.02) | 200 | 0.1 | **0.106** | 0.021 | 0.021 | **0.194** | **0.116** | 0.026 |

## 3. Gain (ESS against the pooled Wilson bound on a random draw, from the truth to the limit, delta 0.05)

| label | n_s | pre-flight | b1w, large pool | ratio to pre-flight | b1w, stored design | Wald-t, large pool | bootstrap-t StratPPI, large pool |
|---|---|---|---|---|---|---|---|
| refusal | 100 | 2.93 | 2.81 | 0.96 | 2.73 | 2.12 | 2.56 |
| refusal | 200 | 2.93 | 3.04 | 1.04 | 2.89 | 2.63 | 3.01 |
| unsafe | 100 | 1.29 | 1.00 | 0.77 | 0.99 | 0.87 | 0.00 |
| unsafe | 200 | 1.29 | 1.08 | 0.83 | 1.07 | 1.11 | 0.00 |

## 4. A claim about the prompt source (pool redrawn, strata rebuilt, 10,000 replications)

| label | n_s | delta | b1w, sampled-pool term | Wald-t b1, sampled-pool term | b1w, no term | StratPPI estimator, bootstrap-t | pooled Wilson | Clopper-Pearson | ESS, b1w with the term | its cap | ESS, Wald-t with the term | its cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | 100 | 0.05 | 0.047 | 0.029 | **0.086** | **0.085** | 0.045 | 0.045 | 1.80 | 1.93 | 1.49 | 1.66 |
| refusal | 100 | 0.1 | 0.092 | 0.078 | **0.148** | **0.145** | 0.100 | 0.069 | 1.84 | 1.97 | 1.51 | 1.67 |
| refusal | 200 | 0.05 | 0.049 | 0.038 | **0.127** | **0.127** | **0.056** | 0.043 | 1.45 | 1.50 | 1.35 | 1.45 |
| refusal | 200 | 0.1 | 0.104? | 0.093 | **0.189** | **0.190** | 0.098 | 0.098 | 1.47 | 1.51 | 1.36 | 1.45 |
| unsafe | 100 | 0.05 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.98 | 1.00 | 0.86 | 0.90 |
| unsafe | 100 | 0.1 | **0.153** | 0.000 | **0.153** | 0.009 | **0.151** | 0.000 | 0.99 | 1.01 | 0.69 | 0.76 |
| unsafe | 200 | 0.05 | 0.026 | 0.026 | 0.026 | 0.000 | 0.024 | 0.024 | 1.07 | 1.04 | 1.11 | 1.05 |
| unsafe | 200 | 0.1 | **0.113** | 0.026 | **0.113** | 0.005 | **0.110** | 0.024 | 1.07 | 1.04 | 0.94 | 0.97 |

## 5. The registered predictions

| | prediction | outcome | what was found |
|---|---|---|---|
| M1 | with a large pool the stratified Wald-t limit is over its level in no cell | kept | 0 of 4 over |
| M2 | with a large pool the bootstrap-t StratPPI limit is over its level in at most one of its four cells, and its miss exceeds the level by no more than 0.01 in any | kept | 1 of 4 over; largest miss minus level +0.003 |
| M3a | with a large pool `b1w` is over its level in no cell if the rate is under 0.45, and in at least one of its four cells if it is above 0.55 (step R5's rule) | no label to test it | rate 0.478; 0 of 4 over |
| M3b | if the rate is between 0.45 and 0.55, the miss of `b1w` with a large pool is within 0.01 of its level in every cell | kept | rate 0.478; miss minus level from +0.001 to +0.002 |
| M4 | the gain of `b1w` with a large pool is within 20% of the pre-flight's prediction at both safety-set sizes | kept | n_s 100: 2.81 against 2.93; n_s 200: 3.04 against 2.93 |
| M5 | the gain of the Wald-t limit with a large pool is at least 1.7 at both safety-set sizes | kept | n_s 100: 2.12; n_s 200: 2.63 |
| M6 | for the prompt source, the Wald-t limit with the sampled-pool term is over in no cell | kept | 0 of 4 over |
| M7 | for the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit are each over in at least one of the two cells at delta 0.05 | kept | no term: 2 of 2; bootstrap-t StratPPI: 2 of 2 |
| M8 | for the prompt source, the gain of the Wald-t limit with the term is above 1 and within 15% of section 6.4's cap at both safety-set sizes | kept | n_s 100: 1.49 against 1.66; n_s 200: 1.35 against 1.45 |
| M9 | on a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and Clopper-Pearson on a random draw is over in no cell | kept | unsafe n_s 100: 1.00; unsafe n_s 200: 1.08; Clopper-Pearson 0 of 4 over |

The refusal rate is 0.478, inside the range of 18% to 65% that no earlier cell tested.

Reported without a prediction: `b1w` with the sampled-pool term is over in 0 of 4 cells for the prompt source.

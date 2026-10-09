# The mid-rate confirmation pool (plan step R5b)

`scripts/confirm_mid.py`. Registered in `.planning/paper-certification/R5b_registration.md` before any response on these prompts was generated. One pool of 400 OR-Bench prompts, every unused hard-1K prompt topped up with unused benign prompts of the 80K set; Granite-3.3-2B, 8 reference responses a prompt for the strata and 16 of the trained policy; Qwen3Guard-4B labels. Checks: `replacement_check.py` (5,000 stored draws, 40,000 with replacement) and `twophase_check.py` (10,000), called unchanged.

## 1. The labels

| label | rate, trained policy | rate, reference | class | ICC of the reference | prompts with a reference rate of 0 or 1 | pre-flight ESS |
|---|---|---|---|---|---|---|
| refusal | 0.459 | 0.462 | mid | 0.81 | 76% | 2.79 |
| unsafe | 0.019 | 0.035 | rare | 0.46 | 90% | 1.34 |

What the mixture is made of (the trained policy's 16 responses a prompt):

| label | source | prompts | rate | reference rate | prompts at a rate of 0 | at 1 |
|---|---|---|---|---|---|---|
| refusal | hard-1k | 193 | 0.813 | 0.784 | 6% | 62% |
| refusal | 80k | 207 | 0.122 | 0.162 | 73% | 5% |
| unsafe | hard-1k | 193 | 0.028 | 0.051 | 88% | 0% |
| unsafe | 80k | 207 | 0.016 | 0.021 | 94% | 0% |

The source alone explains much of what the strata gain on the refusal label. The variance of a random draw over that of a proportional stratified draw is 1.93 with two strata, the two sources, and 2.90 with the 8 strata of the reference rate. Share of hard-1K prompts in the 8 strata, lowest reference rate first: 11%, 11%, 11%, 31%, 61%, 84%, 89%, 88%.

## 2. Validity with a large pool (strata fixed, drawn with replacement, 40,000 draws)

Miss rates; **bold** is over the level, `?` unresolved.

| label | n_s | delta | b1w | Wald-t b1 | StratPPI estimator, bootstrap-t | StratPPI, normal limit | pooled Wilson | Clopper-Pearson |
|---|---|---|---|---|---|---|---|---|
| refusal (0.46) | 100 | 0.05 | 0.044 | 0.026 | 0.051? | **0.061** | 0.046 | 0.046 |
| refusal (0.46) | 100 | 0.1 | 0.096 | 0.063 | 0.101? | **0.114** | 0.101? | 0.101? |
| refusal (0.46) | 200 | 0.05 | 0.045 | 0.036 | 0.050? | **0.054** | **0.053** | 0.038 |
| refusal (0.46) | 200 | 0.1 | 0.096 | 0.083 | 0.100 | **0.105** | 0.093 | 0.093 |
| unsafe (0.02) | 100 | 0.05 | 0.000 | 0.000 | 0.007 | **0.156** | 0.000 | 0.000 |
| unsafe (0.02) | 100 | 0.1 | **0.140** | 0.000 | 0.010 | **0.183** | **0.142** | 0.000 |
| unsafe (0.02) | 200 | 0.05 | 0.016 | 0.016 | 0.006 | **0.117** | 0.020 | 0.020 |
| unsafe (0.02) | 200 | 0.1 | 0.089 | 0.089 | 0.019 | **0.193** | 0.099 | 0.099 |

## 3. Gain (ESS against the pooled Wilson bound on a random draw, from the truth to the limit, delta 0.05)

| label | n_s | pre-flight | b1w, large pool | ratio to pre-flight | b1w, stored design | Wald-t, large pool | bootstrap-t StratPPI, large pool |
|---|---|---|---|---|---|---|---|
| refusal | 100 | 2.79 | 2.78 | 1.00 | 2.93 | 2.09 | 2.64 |
| refusal | 200 | 2.79 | 2.97 | 1.06 | 2.82 | 2.56 | 3.03 |
| unsafe | 100 | 1.34 | 1.00 | 0.75 | 1.02 | 0.88 | 0.00 |
| unsafe | 200 | 1.34 | 1.08 | 0.81 | 1.08 | 1.12 | 0.01 |

## 4. A claim about the prompt source (pool redrawn, strata rebuilt, 10,000 replications)

| label | n_s | delta | b1w, sampled-pool term | Wald-t b1, sampled-pool term | b1w, no term | StratPPI estimator, bootstrap-t | pooled Wilson | Clopper-Pearson | ESS, b1w with the term | its cap | ESS, Wald-t with the term | its cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal | 100 | 0.05 | 0.047 | 0.030 | **0.079** | **0.089** | 0.044 | 0.044 | 1.80 | 1.92 | 1.49 | 1.64 |
| refusal | 100 | 0.1 | 0.091 | 0.080 | **0.144** | **0.146** | 0.099 | 0.099 | 1.84 | 1.95 | 1.50 | 1.65 |
| refusal | 200 | 0.05 | 0.047 | 0.039 | **0.117** | **0.120** | **0.057** | 0.041 | 1.44 | 1.50 | 1.34 | 1.44 |
| refusal | 200 | 0.1 | 0.102? | 0.095 | **0.174** | **0.178** | 0.098 | 0.098 | 1.45 | 1.50 | 1.35 | 1.44 |
| unsafe | 100 | 0.05 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 1.00 | 1.00 | 0.88 | 0.90 |
| unsafe | 100 | 0.1 | **0.145** | 0.000 | **0.145** | 0.004 | **0.141** | 0.000 | 1.02 | 1.01 | 0.71 | 0.76 |
| unsafe | 200 | 0.05 | 0.017 | 0.017 | 0.017 | 0.000 | 0.020 | 0.020 | 1.05 | 1.04 | 1.09 | 1.06 |
| unsafe | 200 | 0.1 | 0.095 | 0.095 | 0.095 | 0.007 | 0.100 | 0.100 | 1.05 | 1.04 | 0.93 | 0.97 |

## 5. The registered predictions

| | prediction | outcome | what was found |
|---|---|---|---|
| M1 | with a large pool the stratified Wald-t limit is over its level in no cell | kept | 0 of 4 over |
| M2 | with a large pool the bootstrap-t StratPPI limit is over its level in at most one of its four cells, and its miss exceeds the level by no more than 0.01 in any | kept | 0 of 4 over; largest miss minus level +0.001 |
| M3a | with a large pool `b1w` is over its level in no cell if the rate is under 0.45, and in at least one of its four cells if it is above 0.55 (step R5's rule) | no label to test it | rate 0.459; 0 of 4 over |
| M3b | if the rate is between 0.45 and 0.55, the miss of `b1w` with a large pool is within 0.01 of its level in every cell | kept | rate 0.459; miss minus level from -0.006 to -0.004 |
| M4 | the gain of `b1w` with a large pool is within 20% of the pre-flight's prediction at both safety-set sizes | kept | n_s 100: 2.78 against 2.79; n_s 200: 2.97 against 2.79 |
| M5 | the gain of the Wald-t limit with a large pool is at least 1.7 at both safety-set sizes | kept | n_s 100: 2.09; n_s 200: 2.56 |
| M6 | for the prompt source, the Wald-t limit with the sampled-pool term is over in no cell | kept | 0 of 4 over |
| M7 | for the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit are each over in at least one of the two cells at delta 0.05 | kept | no term: 2 of 2; bootstrap-t StratPPI: 2 of 2 |
| M8 | for the prompt source, the gain of the Wald-t limit with the term is above 1 and within 15% of section 6.4's cap at both safety-set sizes | kept | n_s 100: 1.49 against 1.64; n_s 200: 1.34 against 1.44 |
| M9 | on a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and Clopper-Pearson on a random draw is over in no cell | kept | unsafe n_s 100: 1.00; unsafe n_s 200: 1.08; Clopper-Pearson 0 of 4 over |

The refusal rate is 0.459, inside the range of 18% to 65% that no earlier cell tested.

Reported without a prediction: `b1w` with the sampled-pool term is over in 0 of 4 cells for the prompt source.

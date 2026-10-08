# The confirmation pool (plan step R5)

`scripts/confirm_pool.py`, registered in `.planning/paper-certification/R5_registration.md` before any response was generated. Three pools of 400 OR-Bench prompts that no earlier pool, training set or results file holds; Granite-3.3-2B, 8 reference responses a prompt for the strata and 16 of the trained policy; Qwen3Guard-4B labels. Checks: `replacement_check.py` (5,000 stored draws, 40,000 with replacement) and `twophase_check.py` (10,000), called unchanged.

## 1. The labels

| pool | label | rate, trained policy | rate, reference | class | ICC of the reference | prompts with a reference rate of 0 or 1 | pre-flight ESS |
|---|---|---|---|---|---|---|---|
| K1: benign prompts, OR-Bench-80K | refusal | 0.122 | 0.136 | mid | 0.68 | 80% | 1.82 |
| K2: benign prompts, OR-Bench hard-1K | refusal | 0.824 | 0.796 | mid | 0.62 | 66% | 1.65 |
| K2: benign prompts, OR-Bench hard-1K | unsafe | 0.032 | 0.052 | rare | 0.40 | 85% | 1.31 |
| K3: toxic prompts, OR-Bench | refusal | 0.977 | 0.974 | near one | 0.69 | 96% | 2.67 |
| K3: toxic prompts, OR-Bench | unsafe | 0.016 | 0.018 | rare | 0.57 | 96% | 2.47 |

## 2. Validity with a large pool (strata fixed, drawn with replacement, 40,000 draws)

Miss rates; **bold** is over the level, `?` unresolved.

| label | n_s | delta | b1w | Wald-t b1 | StratPPI estimator, bootstrap-t | StratPPI, normal limit | pooled Wilson | Clopper-Pearson |
|---|---|---|---|---|---|---|---|---|
| K1:refusal (0.12) | 100 | 0.05 | 0.024 | 0.024 | 0.048 | **0.079** | 0.034 | 0.034 |
| K1:refusal (0.12) | 100 | 0.1 | 0.084 | 0.056 | 0.099 | **0.131** | 0.069 | 0.069 |
| K1:refusal (0.12) | 200 | 0.05 | 0.032 | 0.039 | 0.048 | **0.066** | 0.038 | 0.038 |
| K1:refusal (0.12) | 200 | 0.1 | 0.082 | 0.082 | 0.097 | **0.117** | 0.098 | 0.098 |
| K2:refusal (0.82) | 100 | 0.05 | **0.062** | 0.025 | 0.051? | **0.060** | **0.064** | 0.040 |
| K2:refusal (0.82) | 100 | 0.1 | **0.113** | 0.062 | **0.105** | **0.116** | 0.100? | 0.064 |
| K2:refusal (0.82) | 200 | 0.05 | **0.062** | 0.034 | 0.049 | **0.054** | 0.044 | 0.044 |
| K2:refusal (0.82) | 200 | 0.1 | **0.111** | 0.077 | 0.098 | **0.107** | 0.089 | 0.089 |
| K2:unsafe (0.03) | 100 | 0.05 | 0.034 | 0.000 | 0.010 | **0.203** | 0.039 | 0.039 |
| K2:unsafe (0.03) | 100 | 0.1 | 0.034 | 0.034 | 0.024 | **0.234** | 0.039 | 0.039 |
| K2:unsafe (0.03) | 200 | 0.05 | 0.034 | 0.034 | 0.023 | **0.130** | 0.046 | 0.046 |
| K2:unsafe (0.03) | 200 | 0.1 | 0.098 | 0.098 | 0.057 | **0.181** | **0.119** | 0.046 |
| K3:refusal (0.98) | 100 | 0.05 | 0.043 | 0.002 | **0.052** | 0.009 | **0.080** | 0.027 |
| K3:refusal (0.98) | 100 | 0.1 | 0.054 | 0.013 | 0.091 | 0.029 | 0.080 | 0.080 |
| K3:refusal (0.98) | 200 | 0.05 | 0.029 | 0.009 | 0.045 | 0.014 | 0.042 | 0.042 |
| K3:refusal (0.98) | 200 | 0.1 | 0.073 | 0.029 | 0.088 | 0.041 | 0.091 | 0.091 |
| K3:unsafe (0.02) | 100 | 0.05 | 0.000 | 0.000 | 0.017 | **0.276** | 0.000 | 0.000 |
| K3:unsafe (0.02) | 100 | 0.1 | 0.000 | 0.000 | 0.039 | **0.347** | 0.000 | 0.000 |
| K3:unsafe (0.02) | 200 | 0.05 | 0.033 | 0.000 | 0.012 | **0.238** | 0.039 | 0.039 |
| K3:unsafe (0.02) | 200 | 0.1 | 0.033 | 0.033 | 0.042 | **0.285** | 0.039 | 0.039 |

## 3. Gain (ESS against the pooled Wilson bound on a random draw, from the truth to the limit, delta 0.05)

| label | n_s | pre-flight | b1w, large pool | ratio to pre-flight | b1w, stored design | Wald-t, large pool | bootstrap-t StratPPI, large pool |
|---|---|---|---|---|---|---|---|
| K1:refusal | 100 | 1.82 | 1.53 | 0.84 | 1.55 | 1.47 | 1.55 |
| K1:refusal | 200 | 1.82 | 1.62 | 0.89 | 1.66 | 1.73 | 1.88 |
| K2:refusal | 100 | 1.65 | 1.49 | 0.90 | 1.47 | 0.86 | 1.24 |
| K2:refusal | 200 | 1.65 | 1.62 | 0.98 | 1.60 | 1.13 | 1.44 |
| K2:unsafe | 100 | 1.31 | 1.06 | 0.81 | 1.04 | 0.99 | 0.01 |
| K2:unsafe | 200 | 1.31 | 1.11 | 0.84 | 1.13 | 1.20 | 0.12 |
| K3:refusal | 100 | 2.67 | 0.55 | 0.21 | 0.55 | 0.11 | 0.75 |
| K3:refusal | 200 | 2.67 | 0.63 | 0.24 | 0.63 | 0.26 | 0.95 |
| K3:unsafe | 100 | 2.47 | 1.01 | 0.41 | 1.02 | 0.86 | 0.00 |
| K3:unsafe | 200 | 2.47 | 1.07 | 0.43 | 1.09 | 1.08 | 0.00 |

## 4. A claim about the prompt source (pool redrawn, strata rebuilt, 10,000 replications)

| label | n_s | delta | b1w, sampled-pool term | Wald-t b1, sampled-pool term | b1w, no term | StratPPI estimator, bootstrap-t | pooled Wilson | Clopper-Pearson | ESS, b1w with the term | cap |
|---|---|---|---|---|---|---|---|---|---|---|
| K1:refusal | 100 | 0.05 | 0.023 | 0.023 | 0.033 | **0.057** | 0.035 | 0.035 | 1.32 | 1.35 |
| K1:refusal | 100 | 0.1 | 0.078 | 0.053 | 0.097 | **0.112** | 0.070 | 0.070 | 1.35 | 1.37 |
| K1:refusal | 200 | 0.05 | 0.046 | 0.047 | **0.056** | **0.076** | 0.041 | 0.041 | 1.22 | 1.24 |
| K1:refusal | 200 | 0.1 | 0.078 | 0.088 | **0.118** | **0.135** | 0.101? | 0.101? | 1.22 | 1.24 |
| K2:refusal | 100 | 0.05 | **0.058** | 0.022 | **0.081** | **0.064** | **0.062** | 0.036 | 1.22 | 1.33 |
| K2:refusal | 100 | 0.1 | **0.117** | 0.069 | **0.137** | **0.120** | 0.099 | 0.062 | 1.27 | 1.36 |
| K2:refusal | 200 | 0.05 | 0.052? | 0.030 | **0.090** | **0.074** | 0.046 | 0.046 | 1.20 | 1.24 |
| K2:refusal | 200 | 0.1 | 0.095 | 0.070 | **0.142** | **0.130** | 0.087 | 0.087 | 1.22 | 1.24 |
| K2:unsafe | 100 | 0.05 | 0.038 | 0.000 | 0.038 | 0.003 | 0.037 | 0.037 | 1.04 | 1.04 |
| K2:unsafe | 100 | 0.1 | 0.038 | 0.038 | 0.038 | 0.010 | 0.037 | 0.037 | 1.06 | 1.06 |
| K2:unsafe | 200 | 0.05 | 0.042 | 0.042 | 0.042 | 0.012 | 0.045 | 0.045 | 1.04 | 1.05 |
| K2:unsafe | 200 | 0.1 | **0.110** | **0.110** | **0.110** | 0.044 | **0.120** | 0.045 | 1.03 | 1.05 |
| K3:refusal | 100 | 0.05 | 0.016 | 0.004 | 0.040 | 0.040 | **0.077** | 0.027 | 0.53 | 0.62 |
| K3:refusal | 100 | 0.1 | 0.055 | 0.016 | 0.055 | 0.090 | 0.077 | 0.077 | 0.58 | 0.68 |
| K3:refusal | 200 | 0.05 | 0.035 | 0.005 | 0.035 | **0.066** | 0.042 | 0.042 | 0.58 | 0.77 |
| K3:refusal | 200 | 0.1 | 0.080 | 0.035 | 0.080 | **0.126** | 0.091 | 0.091 | 0.66 | 0.83 |
| K3:unsafe | 100 | 0.05 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 1.00 | 1.00 |
| K3:unsafe | 100 | 0.1 | 0.000 | 0.000 | 0.000 | 0.018 | 0.000 | 0.000 | 1.01 | 1.02 |
| K3:unsafe | 200 | 0.05 | 0.034 | 0.000 | 0.034 | 0.000 | 0.040 | 0.040 | 1.03 | 1.03 |
| K3:unsafe | 200 | 0.1 | 0.034 | 0.034 | 0.034 | 0.002 | 0.040 | 0.040 | 1.02 | 1.03 |

## 5. The registered predictions

| | prediction | outcome | what was found |
|---|---|---|---|
| P1 | with a large pool the Wald-t limit and the bootstrap-t StratPPI limit are over their level in no mid-rate cell, at either delta | **refuted** | Wald-t b1: 0 of 8 over; StratPPI estimator, bootstrap-t: 1 of 8 over |
| P2a | with a large pool `b1w` is over its level in no cell on a label with a rate under 0.45 | kept | 0 of 4 cells over on 1 such labels |
| P2b | with a large pool `b1w` is over its level in at least one of the four cells of every label with a rate above 0.55 | kept | K2:refusal: 4 of 4 |
| P3 | the gain of `b1w` with a large pool is within 20% of the pre-flight's prediction on every mid-rate label, at both safety-set sizes | kept | K1:refusal n_s 100: 1.53 against 1.82; K1:refusal n_s 200: 1.62 against 1.82; K2:refusal n_s 100: 1.49 against 1.65; K2:refusal n_s 200: 1.62 against 1.65 |
| P4a | for the prompt source, `b1w` with the sampled-pool term is over in no mid-rate cell at delta 0.05, and the Wald-t limit with the term in none at either delta | **refuted** | `b1w`: 1 of 4; Wald-t: 0 of 8 |
| P4b | for the prompt source, `b1w` without the term and the bootstrap-t StratPPI limit are each over in at least a third of the mid-rate cells at delta 0.05 | kept | no term: 3 of 4; bootstrap-t StratPPI: 4 of 4 |
| P4c | for the prompt source, the gain of `b1w` with the term is within 15% of section 6.4's cap on every mid-rate label | kept | K1:refusal n_s 100: 1.32 against 1.35; K1:refusal n_s 200: 1.22 against 1.24; K2:refusal n_s 100: 1.22 against 1.33; K2:refusal n_s 200: 1.20 against 1.24 |
| P5 | on a rare label the strata gain nothing (ESS of `b1w` between 0.9 and 1.15) and Clopper-Pearson on a random draw is over in no cell | kept | K2:unsafe n_s 100: 1.06; K2:unsafe n_s 200: 1.11; K3:unsafe n_s 100: 1.01; K3:unsafe n_s 200: 1.07; Clopper-Pearson 0 of 8 over |

Labels with a rate between 18% and 65%, the range no earlier cell tested: none.

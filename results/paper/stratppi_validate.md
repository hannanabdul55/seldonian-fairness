# StratPPI's interval at safety-test sizes: five checks of the P14 finding

`scripts/stratppi_validate.py`; 5,000 draws per cell in part A (reference-rate strata), 4,000 in part B
(judge-logit strata); delta 0.05, one-sided (the upper end of a two-sided 90% interval).

- **over**: miss > delta + 2 se (0.0562 in A, 0.0569 in B); **unresolved**: delta < miss <= delta + 2 se;
  **at or under**: miss <= delta. Counts are written over / unresolved / at or under.
- A cell of part A is one label at one checkpoint and one safety-set size: 13 label-checkpoints with a rate of
  5-95% (C2:refusal at step 0 is at 95.4% and is left out) by two sizes, 26 cells. The paper's Table 3 takes the
  largest miss over a label's checkpoints, a maximum of two or three cells judged by a one-cell threshold; both
  counts are given. A cell of part B is one judge wording, rate, n and number of strata K.
- ESS: (mean bound minus truth of the reference arm / the same for this arm) squared. The reference arm is a random
  split with a pooled Wilson bound in A and the labels alone with Clopper-Pearson in B. Medians are over the cells
  where the arm is not over.
- The baseline's arms are replayed on its own random streams: the largest difference from `stratppi.json` in any
  shared cell's miss rate is 0.0000, and the vectorised arm used here differs from the baseline's
  `stratppi_point` by at most 3e-14 (relative) on the draws checked.

## 1. Is our implementation StratPPI?

**What was compared.** No authors' code is public: the NeurIPS checklist in the arXiv source says "Code may be made
available at a future date", and a search found none. So this is not a comparison with the authors' code. It is a
comparison on identical inputs (20 inputs: 5 seeds each of the paper's two-stratum Gaussian design and a binary
five-stratum design, 5 draws of the plasmode cell C1:refusal step 200 n_s 200, 5 draws of refusal|rubric|0 at n 220 of
2,000 with K = 10) with: (a) Algorithm 1 written a second time in its general M-estimator form; (b) `ppi_py` (the
PPI++ authors' library, commit 3d1f0c6), `ppi_mean_ci` within each stratum, composed with the known weights; (c) GLIDE
0.11.0 (EmertonData, a third party that cites Fisch et al.), its core functions with the known weights and its public
`StratifiedPPIMeanEstimator`; and (d) the paper's own simulation. Relative difference for the variance, absolute for
the estimate and the upper limit.

| comparison | max abs diff, estimate | max rel diff, variance | max abs diff, upper limit | the same on the 10 plasmode inputs |
|---|---|---|---|---|
| ours vs Algorithm 1 rewritten (M-estimator form, same plug-ins) | 5.55e-17 | 2.37e-16 | 5.55e-17 | 5.6e-17; 2.4e-16; 5.6e-17 |
| ours vs vectorised arm used in the cells (`spp`, ours) | 5.55e-17 | 1.18e-15 | 5.55e-17 | 2.8e-17; 1.2e-15; 5.6e-17 |
| ours vs ppi_py within strata, composed | 1.69e-02 | 9.06e-02 | 1.61e-02 | 6.1e-03; 8.7e-02; 6.1e-03 |
| ours vs ppi_py within strata, our lam passed in | 1.11e-16 | 5.23e-02 | 1.52e-03 | 2.8e-17; 4.8e-02; 7.2e-04 |
| ours vs GLIDE core functions, known weights | 1.08e-03 | 6.01e-03 | 9.77e-04 | 4.7e-04; 6.0e-03; 4.0e-04 |
| ours vs GLIDE `StratifiedPPIMeanEstimator` (its own weights) | 1.08e-03 | 3.58e-03 | 9.77e-04 | 0.0e+00; 0.0e+00; 0.0e+00 |
| replayed ppi_py conventions (`spp`, ppi_py) vs ppi_py | 1.11e-16 | 8.38e-16 | 1.11e-16 | 2.8e-17; 8.1e-16; 2.8e-17 |
| replayed GLIDE conventions (`spp`, glide) vs GLIDE core | 1.11e-16 | 8.34e-16 | 1.11e-16 | 2.8e-17; 8.3e-16; 2.8e-17 |
| unstratified PPI++: cert017.ppi_point vs ppi_py.ppi_mean_ci | 5.67e-03 | 1.13e-01 | 4.88e-03 | 2.3e-03; 1.1e-01; 4.5e-03 |
| unstratified PPI++: cert017.ppi_point vs GLIDE core | 5.55e-17 | 3.99e-16 | 5.55e-17 | 2.8e-17; 4.0e-16; 2.8e-17 |
| unstratified PPI++: replayed ppi_py conventions vs ppi_py | 1.67e-16 | 1.14e-15 | 1.67e-16 | 2.8e-17; 1.1e-15; 5.6e-17 |

The differences from the libraries are three plug-in choices the paper leaves open: `ppi_py` clips lam to [0, 1] and
divides variances by n; `ppi_py` and GLIDE take var(f) over the labelled and unlabelled draws together, ours over the
unlabelled. Replayed here, each library's choices reproduce it to rounding error (rows 7, 8, 11), and those replays are
the arms in the table of cells below.
The predictor is constant in 25 of the 40 strata of the plasmode-A inputs (the 8-sample reference rate is 0 for 71% of
the prompts) and in 40 of the 50 strata of the plasmode-B inputs (the rubric judge's stored P(Yes) is 0 for 81% of the responses). `ppi_py` divides by zero there and
GLIDE raises; both were given lam = 0, which is what ours does. GLIDE's public estimator refused 10 of the 20 inputs,
all 10 plasmode inputs, for that reason.

The paper's simulation (section 5.1: two-sided 90%, N = 10,000, K = 2), against the mean widths read off its Figure 2
(to about 0.004):

| scenario | allocation | n | labels per stratum | mean width, ours | with ppi_py's conventions | Figure 2 | coverage, ours |
|---|---|---|---|---|---|---|---|
| homogeneous | prop | 100 | 50, 50 | 0.236 | 0.233 | 0.235 | 0.900 |
| homogeneous | prop | 200 | 100, 100 | 0.167 | 0.166 | 0.167 | 0.887 |
| homogeneous | prop | 1000 | 500, 500 | 0.077 | 0.077 | 0.077 | 0.901 |
| different variance | prop | 100 | 50, 50 | 0.235 | 0.230 | 0.235 | 0.885 |
| different variance | prop | 200 | 100, 100 | 0.167 | 0.165 | 0.167 | 0.891 |
| different variance | prop | 1000 | 500, 500 | 0.077 | 0.077 | 0.077 | 0.897 |
| different variance | opt | 100 | 20, 80 | 0.231 | 0.206 | 0.222 | 0.898 |
| different variance | opt | 200 | 40, 160 | 0.153 | 0.146 | 0.152 | 0.898 |
| different variance | opt | 1000 | 200, 800 | 0.068 | 0.068 | 0.068 | 0.898 |

PPBoot, ours against `ppi_py.ppboot` on two draws of refusal|rubric|0 (N = 2,000): the upper limit with lam = 1 and
20,000 resamples, and the mean and sd over 100 runs of the power-tuned limit with the default 1,000 resamples.

| n | se of the estimate | lam = 1: ours | ppi_py | tuned, mean (sd): ours | ppi_py |
|---|---|---|---|---|---|
| 100 | 0.0285 | 0.2134 | 0.2140 | 0.2093 (0.0022) | 0.2092 (0.0022) |
| 225 | 0.0212 | 0.2371 | 0.2367 | 0.2338 (0.0016) | 0.2338 (0.0015) |

The published interval in the validity cells, by whose plug-in choices are used:

| arm | cells | over / unresolved / at or under | largest miss | median ESS where not over |
|---|---|---|---|---|
| A, H = 8: StratPPI as published (the baseline's arm: unlabelled = the whole stratum, ours) | 26 | 9 / 5 / 12 | 0.086 | 3.46 |
| A, H = 8: the same with the unlabelled set = the rest of the stratum, as in the paper's experiments (ours) | 26 | 9 / 2 / 15 | 0.090 | 3.57 |
| A, H = 8: the same with the unlabelled set = the rest of the stratum, as in the paper's experiments (ppi_py) | 26 | 12 / 4 / 10 | 0.090 | 3.54 |
| A, H = 8: the same with the unlabelled set = the rest of the stratum, as in the paper's experiments (glide) | 26 | 9 / 4 / 13 | 0.089 | 3.38 |
| B, K = 5 and 10: StratPPI as published, predictor the logit | 28 | 26 / 2 / 0 | 0.239 | 2.91 |
| B, K = 5 and 10: StratPPI as published (ppi_py), predictor the logit | 28 | 26 / 2 / 0 | 0.229 | 2.91 |
| B, K = 5 and 10: StratPPI as published (glide), predictor the logit | 28 | 26 / 2 / 0 | 0.238 | 2.90 |
| B, K = 5 and 10: StratPPI as published, predictor P(Yes), as in the paper | 28 | 26 / 2 / 0 | 0.235 | 2.88 |
| B, K = 5 and 10, the `raw` judge alone (logit constant in 0 of 10 strata): StratPPI as published | 14 | 13 / 1 / 0 | 0.187 | 2.94 |
| B, K = 5 and 10, the `rubric` judge alone (logit constant in 8 of 10 strata): StratPPI as published | 14 | 13 / 1 / 0 | 0.239 | 2.87 |
| B, unstratified: PPI++ normal | 14 | 14 / 0 / 0 | 0.232 | - |
| B, unstratified: PPI++ normal (ppi_py) | 14 | 14 / 0 / 0 | 0.218 | - |
| B, unstratified: PPI++ normal, predictor P(Yes) | 14 | 14 / 0 / 0 | 0.222 | - |

Two-sided, as the interval is published (90%, nominal miss 0.10, over if above 0.1085 in A): A 12 of 26 over (largest 0.133); B 22 of 28 (largest 0.251).
The lower limit alone misses up to 0.088 in A (on the 93% label, the mirror image) and 0.058 in B.
By the paper's Table 3 counting (largest miss over checkpoints, 10 label-by-size cells) the baseline's arm is over in 4 of 10.
Per cell it is over in 9 of 26: 6 of the 6 cells of C2:unsafe (the 9% label) and 3 of the other 20, all at n_s 100. Rare labels (under 5%): over in 12 of 12 cells (misses 0.096-0.445); `b1w` in 9, the bootstrap-t limit in 0.

**Verdict.** Our implementation is Algorithm 1 of the paper to rounding error (6e-17 on the upper limit). The largest discrepancy from someone else's code is against `ppi_py` composed within strata: 0.017 on the estimate, 9% on the variance and 0.016 on the upper limit, all of it `ppi_py`'s clipping of lam to [0, 1] and its division by n; against GLIDE it is 0.0010 on the upper limit. Our unstratified PPI++ equals GLIDE's to rounding error and differs from `ppi_py` by the same two choices. On the paper's own simulation ours gives its Figure 2 widths at proportional allocation to the third decimal and its coverage (0.885-0.901 against a nominal 0.90); at the oracle allocation it matches at n 200 and 1,000 and is 0.231 against 0.222 at n 100, where `ppi_py`'s conventions give 0.206. None of this lowers a count: in part B the published interval is over in 26 of 28 cells under our choices, 26 under `ppi_py`'s, 26 under GLIDE's and 26 with the judge's probability as the predictor (the paper's set-up); in part A in 9 of 26 (baseline), 9, 9 and 12 with the unlabelled set drawn as the paper draws it. What the comparison cannot rule out is a choice in the authors' unreleased code that neither library makes.

## 2. The paper's allocation of labels

`opt`: Proposition 3, labels in proportion to w_k sd(Y - lam_k f | stratum), from the pool's own moments (the oracle;
the paper runs it only in simulation). `heur`: the rule the paper runs on real data (Appendix B), w_k sqrt(mean c(1 - c)
+ var c) with c the autorater's confidence (the reference rate in A, the judge's P(Yes) in B). The unlabelled draws
stay proportional, as in the paper. The paper gives no floor and its rule can return no label for a stratum; `heur`
and `opt` are run with at least 2 labels per stratum and `heur10` with at least 10. All use exactly n labels;
`prop` is the baseline's rounding (220 of 225 at K = 10).

| arm | cells | over / unresolved / at or under | largest miss | median ESS where not over |
|---|---|---|---|---|
| A, H = 8: StratPPI as published, prop | 26 | 9 / 5 / 12 | 0.086 | 3.46 |
| A, H = 8: StratPPI as published, heur | 26 | 11 / 0 / 15 | 0.217 | 7.20 |
| A, H = 8: StratPPI as published, heur10 | 26 | 9 / 1 / 16 | 0.121 | 5.35 |
| A, H = 8: StratPPI as published, opt | 26 | 9 / 3 / 14 | 0.110 | 7.54 |
| A, H = 8: StratPPI estimator, bootstrap-t, prop | 26 | 0 / 0 / 26 | 0.045 | 2.33 |
| A, H = 8: StratPPI estimator, bootstrap-t, heur | 26 | 11 / 1 / 14 | 0.174 | 3.77 |
| A, H = 8: StratPPI estimator, bootstrap-t, heur10 | 26 | 8 / 1 / 17 | 0.102 | 4.41 |
| A, H = 8: StratPPI estimator, bootstrap-t, opt | 26 | 6 / 3 / 17 | 0.078 | 4.95 |
| A, H = 8: b1w, prop | 26 | 0 / 2 / 24 | 0.054 | 2.16 |
| A, H = 8: b1w, heur | 26 | 0 / 0 / 26 | 0.029 | 1.24 |
| A, H = 8: b1w, heur10 | 26 | 0 / 0 / 26 | 0.036 | 2.09 |
| A, H = 8: b1w, opt | 26 | 0 / 0 / 26 | 0.033 | 1.68 |
| B, K = 5 and 10: StratPPI as published, prop | 28 | 26 / 2 / 0 | 0.239 | 2.91 |
| B, K = 5 and 10: StratPPI as published, heur | 28 | 24 / 1 / 3 | 0.902 | 1.98 |
| B, K = 5 and 10: StratPPI as published, heur10 | 28 | 24 / 1 / 3 | 0.669 | 1.92 |
| B, K = 5 and 10: StratPPI as published, opt | 28 | 28 / 0 / 0 | 0.219 | - |
| B, K = 5 and 10: StratPPI estimator, bootstrap-t, prop | 28 | 0 / 5 / 23 | 0.056 | 1.88 |
| B, K = 5 and 10: StratPPI estimator, bootstrap-t, heur | 28 | 14 / 0 / 14 | 0.901 | 1.33 |
| B, K = 5 and 10: StratPPI estimator, bootstrap-t, heur10 | 28 | 11 / 0 / 17 | 0.655 | 1.71 |
| B, K = 5 and 10: StratPPI estimator, bootstrap-t, opt | 28 | 14 / 11 / 3 | 0.083 | 3.75 |
| B, K = 5 and 10: b1w, prop | 28 | 0 / 1 / 27 | 0.053 | 1.42 |
| B, K = 5 and 10: b1w, heur | 28 | 0 / 0 / 28 | 0.022 | 0.18 |
| B, K = 5 and 10: b1w, heur10 | 28 | 1 / 0 / 27 | 0.065 | 0.60 |
| B, K = 5 and 10: b1w, opt | 28 | 0 / 0 / 28 | 0.042 | 0.56 |

Smallest stratum under each rule (labels): A: heur 2, heur10 10, opt 2; B: heur 2, heur10 10, opt 2.
In the 10 cells of B where the oracle rule leaves every stratum 10 labels or more: published 10 / 0 / 0, bootstrap-t 5 / 3 / 2, `b1w` 0 / 0 / 10.

**Verdict.** No. With the oracle allocation the published interval is over in 9 of 26 cells of A (largest miss 0.110; 9 and 0.086 with proportional) and in 28 of 28 of B (0.219; 26 and 0.239). With the heuristic it is over in 11 and 24, with misses up to 0.217 and 0.902: the rubric judge's P(Yes) is near 0 or 1 in most strata, so the rule sends nearly every label to one stratum (2 or 10 labels are left in each of the others), and the paper itself reports the heuristic as too aggressive with uncalibrated confidences. The allocation also takes away the repair: the bootstrap-t limit on the same estimator, not over in any cell under proportional allocation, is over in 6 of 26 (A) and 14 of 28 (B) with the oracle rule (largest 0.078 and 0.083) and in 11 and 14 with the heuristic. `b1w` is not over under any rule except one cell of `heur10` in B (0.065), and is widest under the paper's rules (median ESS 1.68 in A and 0.56 in B with the oracle rule against 2.16 and 1.42 proportional). In the cells of A where the published interval is not over, the oracle allocation does narrow it (median ESS 7.54 against 3.46). The floor of 2 is ours; in the 10 cells of B where the oracle rule leaves every stratum 10 labels or more the published interval is over in 10.

## 3. PPBoot on the judge-logit cells

Unstratified, the judge's logit as the prediction, 1,000 resamples (50 for lam), the percentile limit of the paper's
Algorithm 1; `width` is the mean of upper limit minus estimate.

| arm | cells | over / unresolved / at or under | largest miss | median ESS where not over | median width |
|---|---|---|---|---|---|
| labels alone, Clopper-Pearson | 14 | 0 / 1 / 13 | 0.051 | 1.00 | 0.0308 |
| PPI++ normal | 14 | 14 / 0 / 0 | 0.232 | - | 0.0198 |
| PPI++ bootstrap-t | 14 | 0 / 1 / 13 | 0.052 | 1.50 | 0.0282 |
| PPBoot (lam = 1) | 14 | 7 / 6 / 1 | 0.133 | 1.45 | 0.0232 |
| PPBoot (power-tuned) | 14 | 12 / 2 / 0 | 0.148 | 1.83 | 0.0200 |

| judge | rate | n of N | labels alone, Clopper-Pearson | PPI++ normal | PPI++ bootstrap-t | PPBoot (lam = 1) | PPBoot (power-tuned) |
|---|---|---|---|---|---|---|---|
| refusal|raw|0 | 0.013 | 225 of 4000 | 0.000; 0.0202 | **0.199**; 0.0112 | 0.000; 0.5497 | 0.053; 0.0227 | **0.097**; 0.0122 |
| refusal|raw|0 | 0.013 | 1000 of 4000 | 0.023; 0.0075 | **0.088**; 0.0057 | 0.031; 0.0072 | 0.055; 0.0121 | **0.078**; 0.0059 |
| refusal|raw|0 | 0.050 | 225 of 4000 | 0.024; 0.0307 | **0.086**; 0.0210 | 0.029; 0.0283 | 0.055; 0.0252 | **0.075**; 0.0217 |
| refusal|raw|0 | 0.050 | 1000 of 4000 | 0.047; 0.0128 | **0.069**; 0.0103 | 0.050; 0.0114 | 0.052; 0.0134 | **0.065**; 0.0105 |
| refusal|raw|0 | 0.200 | 100 of 2000 | 0.050; 0.0767 | **0.070**; 0.0476 | 0.046; 0.0552 | 0.055; 0.0476 | **0.064**; 0.0478 |
| refusal|raw|0 | 0.200 | 225 of 2000 | 0.040; 0.0487 | **0.068**; 0.0326 | 0.052; 0.0356 | **0.057**; 0.0327 | **0.061**; 0.0329 |
| refusal|raw|0 | 0.200 | 500 of 2000 | 0.043; 0.0317 | **0.061**; 0.0233 | 0.050; 0.0247 | 0.050; 0.0237 | 0.057; 0.0234 |
| refusal|rubric|0 | 0.013 | 225 of 4000 | 0.000; 0.0202 | **0.232**; 0.0103 | 0.008; 0.5452 | **0.133**; 0.0104 | **0.148**; 0.0098 |
| refusal|rubric|0 | 0.013 | 1000 of 4000 | 0.023; 0.0076 | **0.103**; 0.0050 | 0.029; 0.0070 | **0.075**; 0.0052 | **0.086**; 0.0052 |
| refusal|rubric|0 | 0.050 | 225 of 4000 | 0.027; 0.0308 | **0.121**; 0.0187 | 0.033; 0.0281 | **0.074**; 0.0196 | **0.075**; 0.0183 |
| refusal|rubric|0 | 0.050 | 1000 of 4000 | 0.041; 0.0128 | **0.079**; 0.0091 | 0.043; 0.0111 | **0.061**; 0.0094 | **0.066**; 0.0093 |
| refusal|rubric|0 | 0.200 | 100 of 2000 | 0.051; 0.0767 | **0.112**; 0.0499 | 0.042; 0.0666 | **0.077**; 0.0528 | **0.073**; 0.0496 |
| refusal|rubric|0 | 0.200 | 225 of 2000 | 0.036; 0.0489 | **0.078**; 0.0336 | 0.040; 0.0412 | **0.059**; 0.0354 | **0.058**; 0.0340 |
| refusal|rubric|0 | 0.200 | 500 of 2000 | 0.041; 0.0317 | **0.066**; 0.0236 | 0.042; 0.0269 | 0.057; 0.0242 | 0.054; 0.0239 |

Cell: miss; width. Bold: over.

**Verdict.** PPBoot does not hold its level here. The basic form is 7 / 6 / 1 over its 14 cells (largest miss 0.133) and the power-tuned form 12 / 2 / 0 (0.148); PPI++ with a normal limit is 14 / 0 / 0 (0.232) and with a bootstrap-t limit 0 / 1 / 13 (0.052). PPBoot's limit is a percentile limit, first-order like the normal one, and its misses sit between the two. Its widths are close to the normal limit's (median 0.0200 tuned against 0.0198), narrower than the bootstrap-t limit's (0.0282) because it is over. The basic form is over in 1 of the 7 cells of the `raw` judge and in 6 of the 7 of the `rubric` judge.

## 4. A predictor that is not the stratifier (part A, H = 8)

The 8 reference samples split: strata on the mean of samples 1-4, the regression on the mean of samples 5-8, against
the baseline (both on all 8) and a control (both on samples 1-4).

| arm | cells | over / unresolved / at or under | largest miss | median ESS where not over | median ESS, last checkpoints (Table 3's 10 cells) |
|---|---|---|---|---|---|
| strata 1-8, predictor 1-8 (Table 3): StratPPI as published | 26 | 9 / 5 / 12 | 0.086 | 3.46 | 3.04 |
| strata 1-8, predictor 1-8 (Table 3): StratPPI estimator, bootstrap-t | 26 | 0 / 0 / 26 | 0.045 | 2.33 | 2.10 |
| strata 1-8: b1w (no predictor) | 26 | 0 / 2 / 24 | 0.054 | 2.16 | 2.16 |
| strata 1-4, predictor 1-4 (control): StratPPI as published | 26 | 7 / 6 / 13 | 0.082 | 3.30 | 2.89 |
| strata 1-4, predictor 1-4 (control): StratPPI estimator, bootstrap-t | 26 | 0 / 0 / 26 | 0.044 | 2.27 | 2.25 |
| strata 1-4: b1w (no predictor) | 26 | 0 / 0 / 26 | 0.050 | 2.06 | 2.06 |
| strata 1-4, predictor 5-8: StratPPI as published | 26 | 12 / 0 / 14 | 0.104 | 3.39 | 3.22 |
| strata 1-4, predictor 5-8: StratPPI estimator, bootstrap-t | 26 | 0 / 4 / 22 | 0.053 | 2.30 | 1.97 |

| label, last checkpoint (rate) | n_s | b1w, strata 1-8 | bootstrap-t StratPPI, 1-8 / 1-8 | b1w, strata 1-4 | bootstrap-t StratPPI, 1-4 / 1-4 | bootstrap-t StratPPI, 1-4 / 5-8 |
|---|---|---|---|---|---|---|
| C1:refusal (0.16) | 100 | 0.023; 2.39 | 0.035; 2.58 | 0.029; 2.33 | 0.042; 2.39 | 0.051; 2.88 |
| C1:refusal (0.16) | 200 | 0.022; 2.30 | 0.042; 3.10 | 0.025; 2.24 | 0.040; 2.80 | 0.042; 3.09 |
| C2:refusal (0.93) | 100 | 0.054; 0.96 | 0.031; 0.83 | 0.050; 0.93 | 0.038; 0.84 | 0.012; 0.42 |
| C2:refusal (0.93) | 200 | 0.045; 1.12 | 0.029; 1.03 | 0.034; 1.05 | 0.025; 0.99 | 0.017; 0.82 |
| C2:unsafe (0.09) | 100 | 0.018; 1.38 | 0.034; 1.43 | 0.020; 1.31 | 0.026; 1.27 | 0.040; 1.56 |
| C2:unsafe (0.09) | 200 | 0.024; 1.36 | 0.036; 1.59 | 0.024; 1.36 | 0.033; 1.55 | 0.035; 1.62 |
| C3:refusal (0.65) | 100 | 0.047; 4.75 | 0.033; 1.77 | 0.041; 4.39 | 0.034; 2.42 | 0.023; 1.48 |
| C3:refusal (0.65) | 200 | 0.040; 5.19 | 0.034; 5.35 | 0.040; 4.79 | 0.033; 5.28 | 0.031; 4.95 |
| C1:refusal pushed (0.18) | 100 | 0.024; 2.19 | 0.036; 2.43 | 0.026; 2.13 | 0.041; 2.23 | 0.053; 2.75 |
| C1:refusal pushed (0.18) | 200 | 0.023; 2.13 | 0.037; 2.49 | 0.024; 1.99 | 0.034; 2.27 | 0.038; 2.32 |

Cell: miss; ESS.

**Verdict.** Giving the regression a predictor independent of the stratifier does not buy anything. With a bootstrap-t limit the split design is 0 / 4 / 22 (largest miss 0.053), with a median ESS of 2.30 over all cells and 1.97 over Table 3's ten, against 2.33 and 2.10 with all 8 samples in both roles and 2.27 and 2.25 for the control. It helps one label (over-refusal at n_s 100: 2.88 and 2.75 pushed, against 2.58 and 2.43) and hurts the two high-rate ones at that size (0.42 against 0.83; 1.48 against 1.77). `b1w` on all 8 samples is at 2.16 and 2.16. The published interval is over in 12 of 26 cells with the independent predictor (largest 0.104), more than with the shared one (9). The explanation in the paper, that sharing the 8 samples leaves the regression little to add, is not supported: given an independent predictor the regression adds no more. Why was not isolated; the split also halves the samples behind the strata (`b1w` on strata of 4 samples: 2.06 over Table 3's ten cells).

## 5. Labels per stratum

Part A: proportional allocation, the reference rate (8 samples) as stratifier and predictor, H strata, 13 cells a row.
`narrowest`: cells where the arm has the smallest mean bound among the three arms other than the published interval
that are not over in the cell (the published interval is over somewhere in every block, so it is not a candidate).

| n_s | H | labels per stratum | arm | over / unresolved / at or under | largest miss | median ESS where not over | narrowest |
|---|---|---|---|---|---|---|---|
| 100 | 4 | 25.0 | b1w | 0 / 1 / 12 | 0.053 | 1.88 | 1 of 13 |
| 100 | 4 | 25.0 | Wald-t b1 | 1 / 0 / 12 | 0.062 | 2.12 | 0 of 13 |
| 100 | 4 | 25.0 | StratPPI as published | 3 / 1 / 9 | 0.079 | 3.72 | - |
| 100 | 4 | 25.0 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.045 | 2.65 | 12 of 13 |
| 200 | 4 | 50.0 | b1w | 0 / 0 / 13 | 0.043 | 1.83 | 0 of 13 |
| 200 | 4 | 50.0 | Wald-t b1 | 0 / 0 / 13 | 0.041 | 2.09 | 0 of 13 |
| 200 | 4 | 50.0 | StratPPI as published | 2 / 1 / 10 | 0.061 | 3.19 | - |
| 200 | 4 | 50.0 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.036 | 2.63 | 13 of 13 |
| 100 | 8 | 12.5 | b1w | 0 / 2 / 11 | 0.054 | 2.19 | 6 of 13 |
| 100 | 8 | 12.5 | Wald-t b1 | 0 / 0 / 13 | 0.036 | 2.00 | 0 of 13 |
| 100 | 8 | 12.5 | StratPPI as published | 6 / 4 / 3 | 0.086 | 3.24 | - |
| 100 | 8 | 12.5 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.045 | 1.77 | 7 of 13 |
| 200 | 8 | 25.0 | b1w | 0 / 0 / 13 | 0.045 | 2.13 | 2 of 13 |
| 200 | 8 | 25.0 | Wald-t b1 | 0 / 0 / 13 | 0.031 | 2.19 | 0 of 13 |
| 200 | 8 | 25.0 | StratPPI as published | 3 / 1 / 9 | 0.063 | 3.56 | - |
| 200 | 8 | 25.0 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.042 | 2.70 | 11 of 13 |
| 100 | 16 | 6.2 | b1w | 1 / 1 / 11 | 0.058 | 2.35 | 12 of 13 |
| 100 | 16 | 6.2 | Wald-t b1 | 0 / 0 / 13 | 0.012 | 1.44 | 0 of 13 |
| 100 | 16 | 6.2 | StratPPI as published | 10 / 1 / 2 | 0.099 | 0.87 | - |
| 100 | 16 | 6.2 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.047 | 1.37 | 1 of 13 |
| 200 | 16 | 12.5 | b1w | 0 / 0 / 13 | 0.045 | 2.23 | 4 of 13 |
| 200 | 16 | 12.5 | Wald-t b1 | 0 / 0 / 13 | 0.018 | 1.82 | 0 of 13 |
| 200 | 16 | 12.5 | StratPPI as published | 4 / 2 / 7 | 0.072 | 3.65 | - |
| 200 | 16 | 12.5 | StratPPI estimator, bootstrap-t | 0 / 0 / 13 | 0.043 | 2.63 | 9 of 13 |

ESS at each label's last checkpoint, bootstrap-t StratPPI / `b1w`:

| label (rate) | n_s 100, H 4 (25.0) | n_s 100, H 8 (12.5) | n_s 100, H 16 (6.2) | n_s 200, H 4 (50.0) | n_s 200, H 8 (25.0) | n_s 200, H 16 (12.5) |
|---|---|---|---|---|---|---|
| C1:refusal (0.16) | 3.37 / 1.93 | 2.58 / 2.39 | 1.42 / 2.49 | 3.08 / 1.92 | 3.10 / 2.30 | 2.94 / 2.51 |
| C2:refusal (0.93) | 0.85 / 0.87 | 0.83 / 0.96 | 0.66 / 0.96 | 1.05 / 1.01 | 1.03 / 1.12 | 1.03 / 1.13 |
| C2:unsafe (0.09) | 1.28 / 1.16 | 1.43 / 1.38 | 1.06 / 1.46 | 1.50 / 1.21 | 1.59 / 1.36 | 1.58 / 1.47 |
| C3:refusal (0.65) | 4.97 / 3.05 | 1.77 / 4.75 | 1.60 / 5.53 | 5.35 / 3.32 | 5.35 / 5.19 | 5.76 / 5.77 |
| C1:refusal pushed (0.18) | 2.61 / 1.89 | 2.43 / 2.19 | 2.03 / 2.23 | 2.46 / 1.83 | 2.49 / 2.13 | 2.37 / 2.12 |

Part B: proportional allocation, K = 5, 10 or 20 strata of the judge's logit; cells grouped by labels per stratum (n / K).

| labels per stratum | (n, K) | arm | over / unresolved / at or under | largest miss | median ESS where not over | narrowest |
|---|---|---|---|---|---|---|
| 5-11 | (100, 10), (100, 20), (225, 20) | b1w | 0 / 0 / 10 | 0.043 | 2.44 | 10 of 10 |
| 5-11 | (100, 10), (100, 20), (225, 20) | StratPPI as published | 10 / 0 / 0 | 0.211 | - | - |
| 5-11 | (100, 10), (100, 20), (225, 20) | StratPPI estimator, bootstrap-t | 0 / 2 / 8 | 0.057 | 1.69 | 0 of 10 |
| 20-25 | (100, 5), (225, 10), (500, 20) | b1w | 0 / 1 / 9 | 0.053 | 2.29 | 4 of 10 |
| 20-25 | (100, 5), (225, 10), (500, 20) | StratPPI as published | 9 / 1 / 0 | 0.239 | 3.16 | - |
| 20-25 | (100, 5), (225, 10), (500, 20) | StratPPI estimator, bootstrap-t | 0 / 2 / 8 | 0.056 | 2.23 | 6 of 10 |
| 45-50 | (1000, 20), (225, 5), (500, 10) | b1w | 0 / 0 / 12 | 0.047 | 1.46 | 2 of 12 |
| 45-50 | (1000, 20), (225, 5), (500, 10) | StratPPI as published | 11 / 1 / 0 | 0.232 | 2.87 | - |
| 45-50 | (1000, 20), (225, 5), (500, 10) | StratPPI estimator, bootstrap-t | 0 / 3 / 9 | 0.055 | 1.87 | 10 of 12 |
| 100-200 | (1000, 10), (1000, 5), (500, 5) | b1w | 0 / 0 / 10 | 0.049 | 1.24 | 0 of 10 |
| 100-200 | (1000, 10), (1000, 5), (500, 5) | StratPPI as published | 9 / 1 / 0 | 0.109 | 2.94 | - |
| 100-200 | (1000, 10), (1000, 5), (500, 5) | StratPPI estimator, bootstrap-t | 0 / 1 / 9 | 0.053 | 1.78 | 10 of 10 |

Median ESS in part B by rate (20%, 5%, 1.3%): bootstrap-t StratPPI K = 5: 2.47, 1.76, 0.54; bootstrap-t StratPPI K = 10: 2.54, 1.87, 0.60; bootstrap-t StratPPI K = 20: 2.51, 1.64, 0.63; `b1w` K = 5: 2.24, 1.26, 1.16; `b1w` K = 10: 2.41, 1.42, 1.20; `b1w` K = 20: 2.60, 1.70, 1.28.

**Verdict.** *Where each arm holds.* The published interval is over in every block: 28 of 78 cells of A, from 2 of 13 at 50 labels per stratum to 10 of 13 at 6, and 39 of 42 of B, still 9 of 10 at 100-200 labels per stratum (largest 0.109, at the 1.3% rate). More labels per stratum shrink the excess and do not remove it at these rates. The bootstrap-t limit on the same estimator is not over in any cell at any size (0 / 0 / 78 in A, 0 / 8 / 34 in B, largest 0.057). `b1w` is 1 / 4 / 73 in A, its one cell over at 6 labels per stratum (0.058), and 0 / 1 / 41 in B. *Which is narrowest.* In A the bootstrap-t StratPPI has the larger median ESS in four of the six blocks: with 4 strata at either size (2.65 against 1.88 for `b1w` at 25 labels per stratum, 2.63 against 1.83 at 50) and at n_s 200 with 8 or 16 strata (2.70 against 2.13 at 25, 2.63 against 2.23 at 12.5). `b1w` is ahead at n_s 100 with 8 or 16 strata (2.19 against 1.77 at 12.5, 2.35 against 1.37 at 6). So 25 labels per stratum is not the dividing line: 12.5 is enough at n_s 200 and not at n_s 100. The cell the paper singles out (refusal on harmful prompts, n_s 100, 8 strata: 1.77 against 4.75) is 5.76 against 5.77 at the same 12.5 labels per stratum with n_s 200, and 4.97 against 3.05 at n_s 100 with 4 strata. Cell by cell the bootstrap-t StratPPI is the narrowest of the three in 12 and 13 of 13 cells with 4 strata, 7 and 11 of 13 cells with 8 strata, 1 and 9 of 13 cells with 16 strata (n_s 100 and 200), so at n_s 100 with 8 strata the two are level by cells (7 and 6) and `b1w` leads on the median because the bootstrap-t limit loses badly on the two high-rate labels. In B `b1w` is the narrowest in 10 of 10 cells at 5-11 labels per stratum and 4 of 10 at 20-25; the bootstrap-t StratPPI in 10 of 12 at 45-50 and 10 of 10 at 100-200. At the 1.3% rate it is useless at any size (ESS 0.60 at K = 10) and `b1w` is not (1.20).

## Does the claim survive?

"StratPPI's published interval runs over its nominal level in these settings" **survives**: 26 of 28 judge-logit cells and 4 of 10 reference-rate cells by the paper's counting (9 of 26 per checkpoint, 5 more unresolved), unchanged by the public implementations' conventions, by the paper's own allocation rules, by an independent predictor, or by reading the interval two-sided. Its qualifiers: (1) it is a statement about Algorithm 1 of the paper, which our code reproduces to rounding error; there is no authors' code to run. (2) It is a finite-sample, one-sided result at rates of 1-20% and strata of 6-200 labels. The paper claims asymptotic coverage and shows two-sided coverage for two Gaussian strata of 50 labels or more, which we reproduce; nothing here contradicts a statement the paper makes. (3) In part A the excess is small outside the 9% label (at most 0.066) and part A sits outside the paper's regime: 62 unlabelled items a stratum, and a predictor that is constant in 5 to 7 of the 8 strata, so StratPPI there is mostly a stratified Wald interval and the finding is the Wald interval's. The same holds for the `rubric` judge in part B (logit constant in 8 of 10 strata). The `raw` judge in part B is the paper's regime, a continuous score with no constant stratum, and there the published interval is over in 13 of 14 cells, with misses of 0.056-0.187: over at every size at the 1.3% and 5% rates, and within about a point of delta at the 20% rate with 500 labels. (4) "A bootstrap-t limit on the same estimator holds" survives only with proportional allocation.

## Sentences in sections 5 and 6 of `reports/paper_certification.md` that need to change

Quoted as they stand, with what the evidence above supports instead.

**Section 5**

1. "We ran it on the same strata and the same 5,000 draws per cell, with the reference rate as the predictor [P14]"
   - Say what "it" is: Algorithm 1 of the paper in our code, which agrees with a second writing of the algorithm to rounding error and
     is over in as many cells or more under the plug-in choices of `ppi_py` and GLIDE (12 and 9 of 26 against 9); the authors have released no code.
     Say also that the reference rate is constant within 5 to 7 of the 8 strata, so in those strata StratPPI is the stratum's plain mean.
2. "It exceeds delta in 4 of 10 cells at delta 0.05 (one of them marginally, at 0.056; 3 of 10 at delta 0.1), most at the 9% label, and in
   every rare-label cell"
   - The 4 of 10 is the largest miss over two or three checkpoints against a one-cell threshold. Add the per-cell count: over in 9 of 26, unresolved
     above delta in 5, at or under in 12; all 6 cells of the 9% label and 3 of the other 20, all at n_s 100. Add that the paper's oracle
     allocation leaves it over in 9 of 26 (largest 0.110) and its heuristic in 11 (0.217).
3. "The same estimator with a bootstrap-t limit holds in all ten cells (largest miss 0.045) and its median ESS is 2.10 against 2.16 for
   `b1w`."
   - Add "with proportional allocation". Under the paper's oracle allocation the same limit is over in 6 of 26 cells (largest 0.078),
     under its heuristic in 11 (0.174). Per checkpoint, proportional: 0 / 0 / 26 of 26.
4. "We attribute that cell, without a test, to strata of about twelve labels on a heavily tied predictor."
   - Now tested, and the attribution does not hold as written: at the same 12.5 labels per stratum reached with n_s 200 and 16 strata the ESS is 5.76
     against 5.77 for `b1w`; the drop is at n_s 100 with 8 or 16 strata (1.77, 1.60) and absent with 4 (4.97).
5. "The within-stratum regression on the reference rate adds little once the limit is valid. We keep `b1w` as the bound for
   reference-rate strata at these sizes and note the bootstrap-t StratPPI as the candidate for larger strata: it was ahead where strata
   held 25 labels, the larger of the two sizes we ran."
   - Replace the last clause with the sweep: the bootstrap-t StratPPI has the larger median ESS with 4 strata at either size (2.65 against 1.88,
     2.63 against 1.83) and at n_s 200 with 8 or 16 strata (2.70 against 2.13; 2.63 against 2.23), and the smaller one at n_s 100 with 8 or 16
     (1.77 against 2.19; 1.37 against 2.35). The line is the safety-set size as much as labels per stratum; "larger strata" should read
     "n_s of 200, or 4 strata". "Adds little" stands (2.33 against 2.16 for `b1w` on the same strata, median ESS over 26 cells).
6. "In our predictor comparison the predictor is the same reference rate that defines the strata, which leaves the regression little to
   add."
   - Tested and not the reason. Strata on samples 1-4 with samples 5-8 as the predictor: bootstrap-t limit 0 / 4 / 22 of 26, median ESS over Table 3's ten cells
     1.97, against 2.10 with all 8 samples in both roles and 2.16 for `b1w`. Replace with: an independent predictor from a split of the 8 samples does
     not improve on using all 8 for both.
7. "We did not run StratPPI's optimal allocation, and the implementation is ours, written from the paper's equations."
   - Both halves are out of date. Replace with what was run: the oracle and heuristic allocations (item 2 and 3 above), and the comparison of
     the implementation with Algorithm 1 rewritten, `ppi_py`, GLIDE and the paper's Figure 2; keep "the authors have released no code".

**Section 6**

8. "StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in those 26), as PPI++ with a normal
   limit is in all 14 (0.061-0.232)."
   - Stands. Add: 26 of 28 with `ppi_py`'s plug-in choices, 26 with GLIDE's, 26 with the judge's probability as the predictor; 28 of 28 with
     the paper's oracle allocation and 24 with its heuristic, where misses reach 0.90 on the overconfident judge; and with 20 strata,
     13 of 14.
9. "The StratPPI estimator with a bootstrap-t limit holds in all 28 (largest miss 0.056) and is the most efficient valid route: with 10 strata
   its median ESS against the labels alone is 2.54 at a 20% rate and 1.87 at 5% (2.47 and 1.76 with 5 strata), where unstratified PPI++ with
   the same limit gives 1.67 and 1.39."
   - "Holds in all 28" should read: not over in any of 28, unresolved above delta in 5 (largest 0.056), and add "with proportional allocation":
     with the paper's oracle allocation it is over in 14 of 28 (largest 0.083). "The most efficient valid route" is true among 5 and 10 strata;
     with 20 strata `b1w` gives 2.60 at the 20% rate against 2.51 for the bootstrap-t StratPPI (and 1.70 against 1.64 at 5%).
     PPBoot belongs in this list and is not a valid route: over in 7 of 14 cells in its basic form and 12 power-tuned (largest 0.133 and 0.148).
10. "Stratifying on the judge and ignoring it within strata (`b1w`) also holds (largest miss 0.053), at 2.41 and 1.42 with 10 strata (2.24 and
    1.26 with 5)."
    - By the one rule: 0 / 1 / 27 of 28. Add 20 strata (2.60 and 1.70) and that at 5-11 labels per stratum `b1w` is
      narrower than the bootstrap-t StratPPI in 10 of 10 cells.
11. "So the third row of Table 4 has a better form when labelling can follow scoring: stratify on the judge's logit, StratPPI's estimator, a
    bootstrap-t limit."
    - Add the conditions the evidence carries: proportional allocation (not the paper's optimal or heuristic rule), and about 45 labels per
      stratum or more for it to be the narrower choice (10 of 12 and 10 of 10 cells); at 20-25 the two split the cells (6 and 4 of 10)
      and at 5-11 `b1w` is narrower in 10 of 10. At the 1.3% rate the count rule's fallback stands.
12. "a bootstrap for PPI is PPBoot (Zrnic, 2024). Our contribution in this section is narrower: evidence that the published normal-quantile
    intervals, stratified or not, do not hold their level at the sample sizes and rates of a safety test; a studentised bootstrap that does"
    - PPBoot is now run: its percentile limit is over in 7 of 14 cells (basic) and 12 of 14 (power-tuned). The sentence should set the
      studentised bootstrap against PPBoot's percentile bootstrap as well as against the normal limits.

**Outside sections 5 and 6, depending on the same evidence** (not asked for, listed so they are not missed)

- Section 9 (limits): "The StratPPI and PPI++ intervals are ours, written from the papers' equations, with proportional allocation, and not checked
  against the authors' code. PPBoot was not run." Every clause is out of date except that no authors' StratPPI code exists to check against.
- Table 2, row "StratPPI as published (normal limit)": the counts stand; the row "StratPPI estimator with a bootstrap-t limit" needs
  "proportional allocation" in its setting. A PPBoot row can be added among the bounds that fail (up to 0.148).
- Abstract and contributions: "in our implementation" can become "in our implementation, which is the published algorithm to rounding error and gives
  the same counts under two public libraries' conventions"; "a bootstrap-t limit restores the level" needs "with proportional allocation" for StratPPI.
- Section 10 (related work): "(the normal-quantile intervals of PPI++ and StratPPI do not; their estimators with a bootstrap-t limit do)" can add that PPBoot's
  percentile interval does not either.

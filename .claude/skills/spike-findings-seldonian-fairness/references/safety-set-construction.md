# Safety-set construction: splits, rerandomisation, stratification

## Requirements

From idea `rerandomized-split` (MANIFEST.md):

- Rate constraints are tested with Clopper-Pearson or a tight Wald stress test, not the
  t-test. The t-test has zero width at a rate of exactly 0 or 1, candidate selection finds
  that, and it swamps everything else (012).
- Validity of a split rule is measured against a full-leak ceiling (the safety test run on
  D_c), so a null result has a stated resolution (012: under about 5% of the ceiling).
- A balance covariate for an LLM safety set must be precise: the exact or many-sample
  reference rate. A 4-sample estimate bought nothing (012).
- Stratify a safety set by equal rank strata of the reference rate with random
  tie-breaking (H about 8, k = 8). Tie-keeping quantile cuts collapse on zero-inflated
  covariates (013).
- Judge plasmode coverage against the mean of the labels the draws come from, never
  against an independent finite "truth" sample (013: that shared offset faked 0.20 misses).
- Rare labels (below about 5%) need exact bounds whatever the split. Approximate bounds
  miss up to 0.45 there (013).
- Measure validity (true miss rate against delta), power (solution rate) and
  predicted-vs-actual agreement over many seeds, with an adversarial arm whose candidate
  overfits what the split balanced.

## How to Build It

### 1. Classic models: keep the split simple

`seldonian/seldonian.py` `_split(X, y, test_size, random_seed, stratify)` (label
stratification, the 7b3aa68 audit fix) is fine. Across 2000 seeds per arm, rerandomised
splits never broke validity, but they buy power only when they balance *outcome-like*
statistics:

- **Design covariates buy nothing.** Labels, groups, features, (group, label) cells and
  Mahalanobis balance on all of them move the solution rate by less than 2.6 points.
- **Outcome-like balance cuts the D_c/D_s disagreement.** Balancing the candidate
  score's TPR curve (`adv_grid`, 18 statistics) takes `sd((d_s - d_c)/se)` from 1.24 to
  0.82. That is worth +9 to +10 points of solution rate when candidate selection's
  predicted bound has headroom (width x2), and -6.8 points when an overfit candidate hugs
  the boundary: fewer lucky passes, no bias.

If the 2020 rerandomisation comes back, make it an opt-in on covariates the user fixes
before the split, documented as "not covered by Thomas et al. (2019); empirically
conservative (spikes 012-013)". Use Mahalanobis rerandomisation (Morgan & Rubin 2012) at a
fixed acceptance probability, not "best of n at a random theta":

```python
# 012/splitlab.py _mahalanobis: accept the first split under the chi2_k quantile p_accept
cov = np.cov(Z, rowvar=False) * (1 / n_s + 1 / (n - n_s))
thr = chi2.ppf(p_accept, rank)
m = _perm_mask(n, rng)                        # redraw until d @ pinv(cov) @ d <= thr
```

### 2. LLM safety sets: reference-rate stratification (013 recipe)

Use it only when `preflight.py` says YES (see Constraints).

1. **Covariate, before any split.** Sample k = 8 responses per pool prompt from the
   reference policy, judge them, and set `r_ref(x)` = the fraction flagged. These samples
   are never reused for any test.
2. **Strata.** Use 8 equal rank strata of `r_ref`, with ties broken at random:
   ```python
   # 013/plasmode.py quantile_strata(..., ties="random")
   order = np.lexsort((rng.random(N), x))
   st = np.empty(N, int); st[order] = np.arange(N) * H // N
   ```
3. **Split.** Within each stratum, send the fraction f to D_s (proportional allocation).
   The rule depends only on `r_ref`, so D_s is independent of training given the strata.
4. **Bound.** Use `013/stratbounds.py b1w(s, n, W, delta, N=None)`, the stratified
   Wilson-type score bound: the smallest `m >= mu_hat` with `m - mu_hat >= z sqrt(V(m))`,
   where `V(m)` shifts every stratum by `m - mu_hat`. `W_h` are the strata's pool shares.
   Pass `N` = pool size when the certificate targets the population the pool was drawn
   from (the two-phase term). At one stratum it is the Wilson bound, so the pooled
   baseline is like-for-like.
5. **Pre-flight.** `013/preflight.py` computes `G = ICC_ref x rho^2 x rel(k) x c_H` from
   the reference samples, and predicts ESS `1/(1-G)` (pool target) and
   `1/(1 - G + G n_s/N)` (population target), for rho 1 and 0.8, with the verdict.
   `--pushed` (014) for a label the training targets: rho interpolated by ICC_ref from
   the bandit's table (0.69 / 0.87 / 0.98 at ICC_ref 0.26 / 0.51 / 0.75), ICC_cand = ICC_ref.

**Measured on real Granite-3.3-2B data** (500-prompt pools; ESS vs a random split; step
200; n_s 100 / 200; H 8; k 8):

| label | rate | ICC_ref | ESS | pass at tau = truth + 0.02 (random -> strat, n_s 200) |
|---|---|---|---|---|
| over-refusal, XSTest + OR-Bench | 17% | 0.72 | 2.40 / 2.42 | 0.23 -> 0.41 |
| refusal, plain PKU requests | 66% | 0.86 | 5.13 / 5.33 | 0.20 -> 0.56 |
| non-refusal, encoded PKU | 9% | 0.50 | 1.42 / 1.43 (encoding strata alone: 1.2) | 0.29 -> 0.38 |
| refusal, encoded PKU | 94% | 0.47 | 1.05 / 1.18 | - |
| gated harm / plain harm | 1-2% | 0.33-0.50 | 1.0 | invalid for every design |

Coverage of this recipe holds on every mid-rate label: at most 0.093 misses at delta 0.1
and at most 0.047 at 0.05, beside the random split's 0.066-0.114.

### 3. Testing a split or bound: the plasmode

`013/plasmode.py run_cells(pool, cells, reps)` fixes one candidate and draws thousands of
safety sets from real labels, with paired random streams across arms. It covers the arms
R, S1, S2, M (metadata), P (placebo, permuted covariate) and O (oracle). A pool is
`dict(cov, draw, p_truth, meta, truth)`, where `truth` must be the mean of the labels
`draw` samples from (Requirements). On the bandit (`013/heteroenv.py`, ICC 0.05-0.75 via a
violation direction shared across actions) the truth is exact.

## What to Avoid

- **The 2020 code's `theta_s`.** `default_rng(seed).random(D + 1)`, all-positive weights on
  [0, 1] features, predicts one class for 79-86% of seeds. Its balance statistic is then
  constant, and "best of 30" is a random split.
- **The t-test on 0/1 rates.** 35% of candidates were "predict all positive", with a
  zero-width bound (012). `ttest_bounds` documents this caveat.
- **Distribution-free stratified bounds.** Three union-intersection betting constructions
  were built (common bet mixture; bets fixed by the hypothesised mean; predictable
  aGRAPA-style bets interleaved across strata with shrinkage). All were valid (0 misses),
  and none beat its own pooled version (ESS 0.35-1.0). Rejecting the composite null over
  every way the mean can be spread across strata costs what stratification saves at
  n_s <= 400. Empirical Bernstein starts about 1.5x wider than Clopper-Pearson (a 2.3x ESS
  penalty). Hoeffding ignores variance, so it cannot gain at all.
- **The stratified Wald-t (`b1`) at low rates.** It undercovers (0.154 at delta 0.1), as the
  pooled Wald does. `b1w` fixes most of that.
- **Tie-keeping strata** on an 8-sample covariate where 73-97% of prompts sit at 0 or 1:
  the strata collapse to 1-2, and ESS goes to 1.00.
- **A finite truth half as the coverage target.** Its fixed +-0.004 offset turned 0.054 into
  0.20 for the narrowest arm.
- **Metadata or placebo strata as a stand-in for the per-prompt rate.** Both give ESS 1.00 on
  the bandit, and metadata explains < 15% (at most 33%) of real prompt variance.
- **A 4-sample reference estimate.** Its ESS rises with k: 1.61 at k = 1 -> 2.40 at k = 8
  (C1). One sample captures under half of the gain.

## Constraints

- **The gain formula.** `G ~= ICC_cand x rho^2 x rel(k) x c_H`,
  `rel(k) = k ICC/(1 + (k-1) ICC)`, and c_H = 0.64 / 0.88 / 0.96 at H = 2 / 4 / 8. It
  predicted the bandit to a median error of 0.021 (Spearman 0.988, 288 cells). On real,
  tied covariates it over-predicts: at H 8, 2.89 predicted vs 2.40 realised (C1), and
  5.43 vs 5.13-5.33 (C3 refusal). Treat it as an upper estimate and a ranking.
- **The two-phase cap.** When the pool is a sample and the certificate targets the
  population, `ESS = 1/(1 - G + G n_s/N)`. It matched the in-loop runs to 0.01 (1.61 vs a
  pool-target 2.80 at n_s/N = 0.4). Stratification pays when prompts are plentiful and
  labels are expensive.
- **Where it applies.** Rates of about 5-90%; ICC_ref >= about 0.3. A label the training
  targets directly is fine (014, Granite over-refusal under `LagrangianReward` against the
  Skywork reward model): the per-prompt rates do not compress (ICC_cand / ICC_ref 1.01; 1.0-1.5
  in the bandit) and rho is set by ICC_ref, not by the pressure (0.92 at ICC_ref 0.72;
  0.69-0.98 in the bandit, flat from pressure 0.5 to 4 because the multiplier answers it).
  Realised ESS 2.1-2.2 against 2.4-2.5 for the same label as a side effect, `b1w` valid.
  Seen only for net rate moves of up to 3 points on real data (9 in the bandit).
- **Score the formula at the measured moderators.** At the measured ICC_cand and rho it
  predicted the realised ESS within 0.10 (bandit, 56 cells, Spearman 0.97) and 0.04-0.09
  (Granite); a fixed rho 0.8 was off by 0.4-0.5 and in the *under* direction, because
  ICC_cand rose under training and rho stayed above 0.8 (014).
- **At ICC_ref 0.75 with one shared risk direction the label cannot be pushed down** (014
  bandit: thresholds below the reference feasible in 1-13% of runs); the per-prompt rate
  is the prompt's, not the action's.
- **Throughput for the covariate** (Granite-3.3-2B, HF generate, 12 GB card): 5.8-9
  generations/s on short prompts and 2.3/s on long encoded prompts with 192-token answers.
  Qwen3Guard-4B (4-bit) judges 7.1 responses/s.
- **Rerandomisation leak bound (012).** Every rule's safety-set error at the chosen
  candidate was within +-0.05 se of zero, against -1.2 se when reusing D_c. So any leak is
  under about 5% of a full leak, even for 40-parameter adversaries balanced on 160 of
  their cells.

## Origin

Synthesized from spikes: 012, 013
Source files available in: sources/012-rerandomized-split/, sources/013-stratified-safety-set/

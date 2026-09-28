---
spike: 012
idea: rerandomized-split
name: rerandomized-split
type: comparison
validates: "Given a candidate/safety split chosen by rerandomisation (the user's 2020 Algorithm 1 and Mahalanobis balance) vs random and stratified splits, when the Seldonian safety test runs over many seeds, then its true miss rate stays <= delta, and we measure the gain in solution rate and predicted-vs-actual agreement, including when the candidate overfits a balanced covariate"
verdict: VALIDATED
related: [001]
tags: [rerandomization, data-split, safety-test-validity, stratification, cpu]
---

# Spike 012: Rerandomised candidate/safety splits

## What This Validates
The 2020 independent study ("Safe Learning Models", section 8.1, Algorithm 1) re-drew the
candidate/safety split until `ghat(theta_s)` at a random `theta_s` matched on the two halves
(v1: under a threshold; v2: best of n). The 2026-09-27 audit (7b3aa68) replaced it with label
stratification on the grounds that it "voids the guarantee". Rerandomisation (Morgan & Rubin
2012) is conservative for a *fixed* outcome function, but the safety test evaluates
`g(theta_c)` with `theta_c` trained on D_c after the split, and no proof covers that.

Given the split rules below, when the Seldonian pipeline runs over thousands of seeds with
the ground truth known, then: (validity) does the safety test's miss rate stay <= delta,
and is the safety-set estimate at the returned candidate still unbiased; (power) what does
balance buy in solution rate and predicted-vs-actual agreement; (adversary) does it leak
when the candidate overfits exactly what the split balanced?

## Research
- **Morgan & Rubin (2012)**, "Rerandomization to improve covariate balance in experiments",
  *Annals of Statistics* 40(2):1263-1282, doi:10.1214/12-AOS1008. Fix a balance rule on
  pre-treatment covariates before seeing outcomes and redraw until it passes; the
  difference-in-means stays unbiased, its variance falls on the balanced directions, and
  the usual (unadjusted) interval becomes conservative.
- **Li, Ding & Rubin (2018)**, "Asymptotic theory of rerandomization in treatment-control
  experiments", *PNAS* 115:9157-9162, doi:10.1073/pnas.1808191115. The exact
  (non-Gaussian) limit under Mahalanobis rerandomisation.
- **Thomas et al. (2019)**, *Science*: the Seldonian safety test's guarantee rests on D_s
  being independent of `theta_c`. A split rule that looks at the data makes D_c and D_s
  dependent, so the guarantee is simply *not covered by the proof*, rather than void.

Mapping to the split: "treatment" is membership of D_s, the "outcome" is the per-row
constraint indicator at `theta_c`. Any function of the pooled data fixed before the split is
a legitimate pre-treatment covariate. The gap in the theory: the outcome function is
chosen *after* the split, by candidate selection on D_c.

| Rule | Arm | What it balances |
|---|---|---|
| Simple random | `random` | nothing |
| Label stratification (the audit fix) | `strat_y` | y |
| Blocked on design cells | `strat_Ay`, `strat_cells` | (A, y), and (bin, A, y) |
| 2020 Algorithm 1 exactly as removed | `alg1_orig_bo30` | `ghat` at `theta_s = default_rng(seed).random(D + 1)`, best of 30 |
| Algorithm 1, non-degenerate `theta_s` | `alg1_gauss_bo30`, `alg1_gauss_thr` | `ghat` at `theta_s ~ N(0, I)` on centred X; best of 30 / first under 0.01 |
| Mahalanobis rerandomisation | `maha_design` | A, y, A*y and every feature, p_accept 0.1 |
| Adversary 1 | `adv_own_score` | Algorithm 1 at the full-data LR (the candidate's own score) |
| Adversary 2 | `adv_grid` | Mahalanobis (p 0.01) on 18 statistics: the TPR curve of the full-data score at 9 thresholds per group |
| Adversary 3 | `adv_bins` | Mahalanobis (p 0.01) on 160 statistics: positives and TP indicators at 3 thresholds in every (group, bin) cell |
| Full-leak reference | `reuse_Dc` | the safety test reuses D_c (not a split rule; the ceiling) |

## How to Run
    cd .planning/spikes/012-rerandomized-split
    ../../../.venv/bin/python check_bound.py        # vectorised bound == project ghat_tpr_diff
    export OMP_NUM_THREADS=1 SPLITLAB_TAU=0.1
    ../../../.venv/bin/python splitlab.py --seeds 2000 --n 5000 --inflate 1 --bound wald \
        --family binned --out wald_binned_n5000_inf1.json       # ~5 min on 8 workers
    ../../../.venv/bin/python summarise.py wald_*.json
    ../../../.venv/bin/python banditsplit.py --seeds 1000 --u-scale 3.0 --out bandit_u3_est.json
    ../../../.venv/bin/python summarise_bandit.py bandit_*.json

Per-run rows are in `results/spikes/012/*.json.xz` (80 MB raw, too big for the spike
directory); the tables are in `results.md`.

## What to Expect
`check_bound.py` prints a vectorised-vs-project difference at machine precision for both
bounds. The sweep tables show `uncovered` at or under delta and `leak` within about two
standard errors of 0 for every split rule, against `reuse_Dc` at 0.3-0.34 and -1.2.

## Investigation Trail
1. **Harness.** Real Seldonian candidate selection over a small family (LR on D_c plus a
   pair of group thresholds, Hardt-style post-processing), with the predicted bound
   vectorised and checked against `ghat_tpr_diff` to 1e-16 (`check_bound.py`), the safety
   test from the project function, and truth from a 1M-row population. 0.1 s a run.
2. **The removed 2020 code balanced nothing.** Its `theta_s = default_rng(seed).random(D+1)`
   has all-positive weights on features in [0, 1], so it predicts one class for every row in
   79-86% of seeds: `ghat(theta_s)` is constant, the first split already has |diff| = 0, and
   "best of 30" returns a random split. Every table shows `alg1_orig_bo30` indistinguishable
   from `random` (paired power difference +-0.003). The non-degenerate arms use
   `theta_s ~ N(0, I)` on centred features.
3. **First settings were swamped by a t-test artefact** (n 1000, tau 0.05, project t-test):
   35% of candidates were "predict everyone positive", because the t-test's width collapses
   to 0 when a subgroup's TPR is exactly 1 (the caveat in `ttest_bounds`), and candidate
   selection finds it. `uncovered` was 0.35 in *every* arm. Switched to Clopper-Pearson.
4. **Clopper-Pearson is too conservative to see a leak:** no true violation in any cell of a
   tau x n x inflation pilot. So the main sweeps use a **tight one-sided Wald test** on the
   gap (covers at about delta on fresh data), keep the project's CP test on the same
   candidate as a second column, and switch to signed TPR differences (the abs of a noisy
   near-zero gap is biased upward even on fresh data), n = 5000, a 121-point grid.
5. **Measured the leak directly.** At the returned candidate: *optimism* = (d_c - d_true)/se
   and *leak* = (d_s - d_true)/se, plus their correlation. A random split gives leak 0 and
   corr 0. The base family is only mildly optimistic (-0.2 to -0.3 se), which bounds any leak.
6. **High-capacity adversary.** `candidate_binned` gives each (group, bin of a noise
   feature) cell its own threshold offset, 40 parameters fit by coordinate ascent on D_c
   under the predicted bound: optimism -0.8 se, four times the base family. `adv_bins`
   balances 160 statistics of exactly those cells; `strat_cells` blocks on them.
7. **Calibrated the ceiling.** `reuse_Dc` (safety test on D_c) shows what a total leak looks
   like: leak -1.2 se, uncovered 0.34, Wald miss rate 0.15 (inflation 1). Every split rule's
   leak is within +-0.05 se of 0 (two-se resolution about 0.045), so any leak is under about
   5% of the ceiling.
8. **Power.** Paired against `random` on the same seeds (`paired_A.md`): balancing
   *outcome-like* statistics cuts the D_c/D_s disagreement sd((d_s - d_c)/se) from 1.24 to
   0.82 (`adv_grid`), 1.08 (`adv_own_score`), 1.10 (`adv_bins`), 1.13 (`alg1_gauss`);
   design covariates (`strat_*`, `maha_design`) leave it at 1.23-1.25. With headroom in the
   prediction (inflation 2) that becomes +9 to +10 points of solution rate for `adv_grid`,
   +4 to +5 for `adv_own_score`/`adv_bins`, 0 to +2.6 for the rest. With the overfit candidate
   hugging the boundary (inflation 1) `adv_grid` *loses* 6.8 points: the candidate is truly
   worse than D_c says, and less noise means fewer lucky passes. No bias either way.
9. **LLM-shaped version** (`banditsplit.py`, the real `SeldonianLLMPolicy` pipeline, 1000
   seeds per rule). Covariates known before training: group, and the reference policy's
   per-prompt violation rate (4 samples, or exact). Default env: every rule valid and none
   helps; composition error sd 0.0072-0.0076 against a total of 0.017, because prompt-level
   rate differences are only 5.5% of the Bernoulli variance, so perfect balance could
   remove at most that much. At `u_scale = 3` (22%): exact-rate stratification +3.5 points
   (paired se 1.6), Mahalanobis +3.1; the 4-sample estimate gets about nothing
   (+1.7 then +0.5 over 4000 seeds); Algorithm 1 at `theta_ref` +0.3.
10. **A coverage flag chased down.** `strat_ref` (estimated) showed uncovered 0.126 against
    delta 0.1 at 1000 seeds (2.8 se). Over 4000 seeds it is 0.1015 +- 0.009, with random at
    0.1048: noise among 13 coverage numbers. The post-stratified bound runs at 0.111 under
    *every* rule, the random split included, so it slightly undercovers as written (per-stratum t on about 40 prompts); that is the bound, not
    the split.

## Results
**Verdict: VALIDATED**, with the reason stated carefully: rerandomised and stratified
splits stayed valid in every setting tried, adversaries included, and they buy power only
when the balanced statistic is close to what the safety test measures.

- **Validity.** Across 2000 seeds per arm, four settings and 11 split rules on the classic
  setup, and 5 rules x 3 environments on the bandit: the tight test's uncovered rate stayed
  at or under delta, the project's Clopper-Pearson test missed 0 times (1 in 2000 in one
  cell), and the safety-set estimate at the returned candidate stayed unbiased. That holds
  even for splits that balance 160 statistics of the cells a 40-parameter candidate
  overfits. Leak is bounded at about 5% of a full reuse of D_c.
- **Why it did not leak.** Balance pulls both halves toward the full-data values *on the
  balanced directions*, while candidate selection's optimism lives in the fine structure
  (which threshold, which cell offset), almost orthogonal to any low-dimensional summary.
  Corr(D_c error, D_s error) rose to +0.3 to +0.4 under the adversaries, yet the D_s
  error at the chosen candidate stayed centred. This is empirical, not a proof: balancing
  on the exact statistics selection runs over reaches the `reuse_Dc` ceiling in the limit.
- **Power.** Design covariates (labels, groups, features, metadata cells): essentially
  nothing, as Morgan-Rubin predicts when they barely predict the outcome. Outcome-like
  covariates (the constraint at a related parameter, the reference rate): less D_c/D_s
  disagreement, +3 to +10 points of solution rate when the prediction has headroom.
- **The 2020 code as written** did nothing (degenerate `theta_s`); the audit fix's
  replacement (label stratification) is also neutral. Neither the old nor the new
  behaviour mattered to the results.
- **For LLM runs:** balancing the safety set pays only when per-prompt violation rates are
  strongly heterogeneous *and* the covariate is precise (the exact reference rate, not 4
  samples). Estimate it from many reference samples, or skip it.
- **Side finding:** the project t-test bound lets candidate selection win by driving a
  subgroup rate to exactly 0 or 1 (zero width). Use Clopper-Pearson or a distribution-free
  bound for rate constraints.

## Follow-up: is real LLM data heterogeneous enough? (2026-09-27)
The bandit said balancing pays only when prompt-to-prompt differences are a large share of
the per-response variance (5.5%: nothing; 22%: +3.5 points with an exact covariate) *and*
the covariate still predicts the trained policy's per-prompt rate. Both were checked on
data already on disk, without new generation.

- **Heterogeneity** (`heterogeneity.py`, `heterogeneity.md`; spike 005's screen, 2 judged
  responses per prompt, 12 models). The intra-prompt correlation (ICC, the removable share)
  is **0.2-0.7 for refusal and for non-refusal on encoded prompts on every model**
  (Qwen3 0.5-0.7), 1-3x the bandit's 22%. Metadata cells (plain/encoding) explain little of
  it (mostly < 15%, at most 33%), so a per-prompt reference rate is needed, not category
  strata. Plain-prompt harm is too rare on most models to measure (0.5B and SmolLM3: about
  0.56-0.59, wide intervals). Caveat: on encoded prompts Qwen3Guard-0.6B's "unsafe" is a
  non-refusal flag (005), so these are rates of the labels a constraint would use, not of
  harm.
- **Persistence through training** (`persistence.py`, `persistence.md`; spike 009's
  Granite-3.3-2B GRPO evals, 48 encoded F prompts x 2 samples, steps 0/25/50/100). A
  prompt's step-0 rate correlates with its later rate at about 0.6 for encoded `unsafe`
  (0.9-1.0 disattenuated), 0.2-0.65 for refusal and the gated label (noisy). The label the
  run trains on directly (`twin_train:ok`) decorrelates as it saturates (0.23 -> 0.10), so a
  constrained label that training pushes hard may persist less than these F labels did (009
  had no constraint on F).
- **Rough payoff.** Removable share of the safety-set variance is about ICC x persistence^2 x
  covariate reliability: 0.5 x (0.4 to 0.8) x 0.9 (8 reference samples), about 0.2-0.35,
  the equivalent of a 25-50% larger safety set, but only with a bound that uses the strata
  (the plain t-test keeps its width; the post-stratified t as written undercovers slightly,
  0.111 at delta 0.1).

**Updated recommendation:** the precondition holds on real data, so the direction is worth
one concrete test: an offline safety-test simulation from k = 8 reference samples and k = 8
trained-checkpoint samples on about 500 prompts (generation only, no retraining), comparing
random vs reference-rate-stratified safety sets under a valid stratified bound.

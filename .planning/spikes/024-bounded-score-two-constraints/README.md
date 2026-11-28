---
spike: 024
idea: multiplier-free-candidate-selection
name: bounded-score-two-constraints
type: comparison
validates: "Given the synthetic bandit with a bounded severity score and two constraints that pull toward different actions (mean severity on adversarial prompts, certified with the betting mixture; violation rate on all prompts, certified with Clopper-Pearson), when candidate selection is steered by the proportional-plus-integral rule with one term per constraint, then (a) it returns solutions as often and with at least 90% of the utility of LagrangianReward with one multiplier per constraint, (b) steering on the betting bound's influence function beats steering on a mean-only bound of the same score, and (c) keeping the penalty out of GRPO's per-prompt standard deviation is no worse than folding it in"
verdict: PARTIAL
related: [023, 023b, 023c, 013, 016]
tags: [candidate-selection, bounded-score, betting-mixture, influence-function, two-constraints, grpo-normalisation, bandit, cpu]
---

# Spike 024: bounded score and two constraints

## What This Validates

Spikes 023-023c settled the steering rule for one 0/1 constraint, where the bound depends on
the rate alone and its gradient is a scalar times the rate gradient. The formulation doc
(https://claude.ai/artifact/DsjKyutY1MXDViYFHcaSND, approved 2026-11-28) says the smooth
evaluator earns its keep only beyond that case. This spike is step 3 of its validation plan:

1. **Two constraints that pull apart.** One multiplier per constraint (`LagrangianReward`
   today) against one proportional-plus-integral (PI) term per constraint.
2. **A bounded score whose variance matters.** The safety test bounds mean severity with the
   betting mixture, which is variance-adaptive. Steering on that bound's influence function
   (library `betting_influence`) penalises a high-variance response more than its mean
   alone would; steering on a mean-only bound (Bentkus, library `bentkus_upper`) does not.
3. **GRPO normalisation (formulation doc gradient rule 6).** The penalty folded into the
   reward before the per-prompt standard deviation, against a separate penalty advantage
   with a fixed scale.

Validity is not at stake: the safety test is the same in every arm.

## Environment (fixed before registration)

022a's `SyntheticEnv` (population 20,000, d = 8, 4 actions, violation biases
`(-4.5, -3.5, -2.2, -1.0)`, pressure 1.0, groups adversarial / benign at 50%) plus a severity
score, subclassed in the lab:

- mean severity `S(x, a) = sigmoid(x . V_a + e_a)`, `V_a ~ N(0, 0.5^2 I)` (seed `[seed, 24]`),
  `e = (-0.8, -2.5, -0.8, -2.5)`: actions 0 and 2 are severe, 1 and 3 mild;
- a response's severity `h ~ Beta(kappa_a S, kappa_a (1 - S))` with
  `kappa = (1.5, 20, 20, 1.5)`: action 0 is severe and spread, 2 severe and tight, 1 mild and
  tight, 3 mild and spread. Actions 0 and 2 have the same mean severity and different spread,
  so only a variance-aware bound tells them apart;
- reward `w_a . x + 2.0 p_v(x, a) + 2.0 S(x, a) + N(0, 0.5^2)` (both pressures 2.0 since
  amendment 1 below; registered at 1.0).

The violation constraint pushes toward actions 0-1 and the severity constraint toward 1 and 3,
so the two pull toward different actions. Calibration (ground truth only, exact over the population, seeds 0-4, no
Seldonian run): at pressures 1.0 the reward-greedy policy is above the uniform reference by
+0.046 in violation rate and +0.034 in adversarial mean severity on average. Among the three bias
patterns tried, this was the one where both constraints bind at pressure 1.0.

**Amendment 1 (2026-11-28, before any tuning).** The registered check (stage `check`, item 3)
failed: unconstrained GRPO crossed the severity threshold on 4 of 5 seeds (seed 0 peaked 0.041
below it). Per the registration the spike stopped. Recalibration on ground truth only (seeds
0-9, greedy policy, no Seldonian run): raising only the severity pressure weakens the violation
pull (at 1.0 / 1.5 the greedy rate excess falls to -0.026 on seed 7), so both pressures go to
2.0, where the greedy policy exceeds the reference by at least 0.042 in rate and 0.069 in
severity on all 10 seeds (margins 0.03). Nothing else changes. Disclosed contamination: the
failed check also ran one end-to-end run per arm on seed 0 (an evaluation seed) at mid-grid
settings in the old environment; all ten returned solutions, utilities 0.23-0.45.

## Constraints and safety test

- `sev`: mean severity on adversarial prompts <= reference + 0.03. `ExpressionConstraint` with
  one measure on [0, 1], bound `betting_mixture` (spike 013's rule for a bounded non-binary
  score).
- `harm`: violation rate on all prompts <= reference + 0.03. `Constraint`, Clopper-Pearson.
- delta = 0.1 overall, split evenly by `SeldonianLLMPolicy` (0.05 each). References are the
  uniform policy's exact values.
- Everything else is 022a's `CFG`: 1,000 prompts split 60/40, predicted test every 25 steps on
  512 prompts, 200 steps of 8 prompts x 8 samples, KL beta 0.01, best-feasible-checkpoint
  selection, one safety test.

## Per-step surrogate (all PI arms)

From the training batch's own labels, as in 023:

- `harm`: running rate (EMA, a = 0.2), smooth Clopper-Pearson `cp_upper` at
  `n = effective_n(predict_n, |D_s|)`; per-response weight `U'(r) (v_i - r)`.
- `sev`: a buffer of the last 5 steps' severity values on adversarial prompts (about 160
  values, the same memory as the EMA), evaluated at
  `n = effective_n(adversarial prompts in the predicted test, adversarial prompts in D_s)`:
  - `full`: `U = betting_upper(buffer)`; per-response weight `psi(h_i)`, the bound's
    influence function at the response's own value (the change in the bound when weight moves
    onto h_i);
  - `mean`: `U = bentkus_upper(mean(buffer))`; per-response weight `B'(mean) (h_i - mean)`.
  Benign responses get weight 0 for `sev`.
- `c_j = U_j - tau_j`, from steps before this one.
- PI multiplier per constraint: `lam_j = c0 sigmoid(c_j / T) + lam_dual_j`,
  `lam_dual_j = clip(lam_dual_j + eta_i c_j, 0, 200)`; penalty `pen_i = sum_j lam_j w_ij`.
  The `U'` factor of 023's rule is inside `w_ij`, so the units match 023c's.

## Arms

| Arm | Steering | Penalty enters GRPO | Knobs tuned |
|---|---|---|---|
| `lagrangian` | `LagrangianReward`, one multiplier per constraint, dual ascent at every predicted test (`sev` penalty on adversarial prompts only) | folded | lr, eta {30, 100, 300}, lam0 {1, 5} |
| `pi_mean` | PI, mean-only `sev` surrogate | folded: `r - pen`, then group-normalised | lr, c0 {10, 30}, eta_i {3, 10, 30} |
| `pi_full` | PI, betting influence for `sev` | folded | the same |
| `pi_mean_sep` | PI, mean-only | separate: `norm(r) - (pen - group mean of pen) / s0` | the same |
| `pi_full_sep` | PI, betting influence | separate | the same |

`s0` is the pooled within-group standard deviation of the base reward on the first training
batch, frozen for the run (a measured reward scale, as the formulation doc prescribes for c0).
T = 0.02 throughout (023's choice). Learning-rate grids as 023: linear
{0.05, 0.1, 0.2, 0.4, 0.8}, net {0.0001, 0.0003, 0.001, 0.003}.

## Design (fixed before the first run, 2026-11-28)

- Policies: 022a's softmax-linear policy and the 94k-weight tanh net.
- Tuning seeds 1000-1019, selection by mean utility (true reward of the returned policy minus
  the reference's; No Solution Found = 0). Evaluation seeds 0-59, every arm on the same seeds,
  paired differences with paired standard errors.
- Outcomes: solution rate; returned policies over either threshold; utility; share of training
  checkpoints over either threshold; runs where the selected checkpoint passed the predicted
  test and failed the safety test; the grid median of utility on the tuning seeds; for the PI
  arms the returned policy's action mix (share of actions 0 and 2 on adversarial prompts).
- Diagnostic (formulation doc gradient rule 5): at every step of the `pi_full` runs, whether
  empirical Bernstein's cost slope `C'(0) = 1 - sqrt(2 ln(2/delta) / n) mu / sigma` is
  negative on the buffer's statistics. Reported, not used.
- Stage `check` before tuning: `betting_influence` at new points equals the in-sample influence
  where they coincide; the per-response weights have the expected signs; one run per arm end to
  end; the `select`-style unconstrained GRPO policy (no penalty, lr 0.4) leaves the feasible
  region on both constraints on 5 seeds (if it does not, the environment does not bind and the
  spike stops before tuning).

## Expectations (written before the first run)

1. `pi_full_sep` (or the best PI arm) reaches at least 90% of `lagrangian`'s utility with a
   solution rate no more than 5 points lower, on both policies.
2. `pi_full` beats `pi_mean` on utility, and `pi_full_sep` beats `pi_mean_sep`, by at least
   2 paired standard errors on at least one policy and is not lower on the other. The returned
   `full` policies use the spread severe action 0 less than the `mean` ones.
3. The separate arms are no worse than their folded twins (paired difference above -1 se on
   both policies).
4. No arm returns a policy over either threshold in more than delta (10%) of runs.
5. Empirical Bernstein's `C'(0)` is negative on under 5% of steps here: with spread scores
   `mu / sigma` is small. This environment is not built to exercise rule 5.

## Decision rule

- **The rule for the GPU follow-up** is the best PI arm if it meets, on both policies,
  utility at least 90% of `lagrangian`'s, solutions no more than 5 points below, expectation 4,
  and a grid median at least `lagrangian`'s. If no PI arm meets it, one multiplier per
  constraint stays for several constraints and the GPU plan says so.
- **Influence-function steering** (`full`) is adopted for bounded scores only if expectation 2
  holds for at least one of the two pairs; otherwise the simpler mean-only surrogate is used.
- **Separate penalty** is adopted if expectation 3 holds for both pairs; otherwise the folded
  form stays and rule 6 of the formulation doc is marked not confirmed.

## How to Run

    cd .planning/spikes/024-bounded-score-two-constraints
    ../../../.venv/bin/python sevlab.py --stage check
    ../../../.venv/bin/python sevlab.py --stage tune
    ../../../.venv/bin/python sevlab.py --stage eval
    ../../../.venv/bin/python sevlab.py --stage report

## Investigation Trail

1. **Registration** pushed in commit `adfb4fd` before any harness code ran.
2. **Check failed** in the registered environment (4 of 5 seeds bind on severity); amendment 1
   raised both pressures to 2.0, committed before the check was re-run (`79fa5e3`). The re-run
   passed: 5 of 5 seeds bind, influence at new points equals in-sample to 1e-17, all weights
   have the expected signs.
3. **Tuning** (5,400 runs on seeds 1000-1019) ran to completion. The first evaluation crashed
   on a run whose first batch held no adversarial prompt (empty severity buffer, NaN mean). Fix
   in commit `3632d3c`: the severity term stays idle until the first
   adversarial batch. No tuning run had hit the case (it would have crashed), so tuning stands;
   evaluation (600 runs, seeds 0-59) was re-run from scratch. About 1 h 10 min on 4 cores.
4. **Chosen settings on a grid edge:** c0 = 10 (lower edge) in 7 of 8 PI choices; eta_i = 30
   (upper) for three, 3 (lower) for two; Lagrangian eta 300 (upper) on the net.

## Results

Verdict: **PARTIAL.** Full tables: `results.md` (60 paired seeds; utility = true reward of the
returned policy minus the reference's, No Solution Found = 0).

| Arm | linear: solutions | utility | vs lagrangian, paired (se) | grid median | net: solutions | utility | vs lagrangian, paired (se) | grid median |
|---|---|---|---|---|---|---|---|---|
| `lagrangian` | 35/60 | 0.235 | | 0.169 | 35/60 | 0.193 | | 0.181 |
| `pi_mean` | 47/60 | 0.283 | +0.048 (0.025) | 0.266 | 42/60 | 0.291 | +0.099 (0.037) | 0.233 |
| `pi_full` | 43/60 | 0.288 | +0.053 (0.039) | 0.258 | 43/60 | 0.254 | +0.061 (0.033) | 0.242 |
| `pi_mean_sep` | 50/60 | 0.299 | +0.064 (0.031) | 0.192 | 52/60 | 0.295 | +0.103 (0.027) | 0.155 |
| `pi_full_sep` | 44/60 | 0.284 | +0.049 (0.033) | 0.224 | 46/60 | 0.290 | +0.097 (0.030) | 0.181 |

- No arm returned a policy over either threshold (0 of 600). Returned policies sit far inside:
  true mean severity 0.08-0.095 and violation rate 0.06-0.07 below their thresholds in every
  arm. At about 110 adversarial safety prompts the betting bound on a spread score is wide, so
  the safety test, not the steering, sets how far in a policy must sit.
- Training checkpoints over a threshold: Lagrangian 16.9% and 24.0%; every PI arm 0-2.1%.
- Selected checkpoint passed the predicted test and failed the safety test: 8-17 of 60 in every
  arm, Lagrangian included (13 and 16) — far more than 023's 0-11. The P11 gap here is the
  betting bound's noise at small n, common to all arms.
- Empirical Bernstein's C'(0) was never negative (0.0% of steps), as expected.

Against the expectations:

1. A PI arm at least 90% of the Lagrangian's utility, solutions within 5 points, both policies:
   **held**, by a wide margin for every PI arm: more solutions (42-52 against 35) and +0.05 to
   +0.10 utility.
2. `full` beats `mean` by 2 paired se on at least one policy: **refuted** for both pairs
   (+0.004 and -0.037; -0.015 and -0.005, all within about 1 se). The returned policies' use of
   the spread severe action does not differ either (share of action 0 among the severe actions
   0.52-0.65 in all arms). Steering on the betting bound's influence function bought nothing
   over steering on the mean.
3. Separate no worse than folded (above -1 se): **held** for both pairs (+0.015, +0.004;
   -0.004, +0.036). But the separate arms are more sensitive to their knobs: lower grid medians
   in 3 of 4 cases (worst grid cell -0.14 on the net against +0.01 folded).
4. No arm over a threshold in more than delta of runs: **held** (0 of 600).
5. C'(0) negative on under 5% of steps: **held** (0%).

Decision rule:

- **GPU rule:** the arms that meet all four conditions on both policies are `pi_mean`,
  `pi_full` and `pi_full_sep` (the last ties the Lagrangian's grid median on the net to 2e-7);
  `pi_mean_sep` has the best utility but fails the grid median on the net (0.155 against
  0.181). The best qualifying arm by utility summed over the policies is **`pi_mean`**
  (0.575, against 0.574 for `pi_full_sep` and 0.542 for `pi_full`): PI with one term per
  constraint, a mean-only surrogate for the bounded score, the penalty folded into the reward.
- **Influence-function steering:** not adopted (expectation 2 failed); the mean-only surrogate
  is used for bounded scores.
- **Separate penalty:** the registered test (expectation 3) passes, so rule 6 is not refuted on
  utility, but it is not adopted for the GPU run because the arm that combines it with the
  chosen surrogate fails the robustness condition. The two registered rules conflict here;
  this resolution (keep the arm that passes every condition) is mine, stated for review.

Why the influence function buys nothing is not established. One untested reading: the returned
policies sit so far inside the threshold that the bound's width, not its slope near the
threshold, decides pass or fail, and the width is set by the safety test's n. What this does not
show: a real LLM, judge drift, and anything about the winner's curse.

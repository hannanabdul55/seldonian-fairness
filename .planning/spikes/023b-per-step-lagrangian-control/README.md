---
spike: 023b
idea: multiplier-free-candidate-selection
name: per-step-lagrangian-control
type: comparison
validates: "Given spike 023's bandit, seeds and per-step bound surrogate, when the Lagrangian's dual ascent runs at every step on the same surrogate c_t that smooth_pen uses, then we learn whether smooth_pen's gain over the Lagrangian comes from per-step information or from dropping the dual state"
verdict: VALIDATED
related: [023]
tags: [candidate-selection, lagrangian, control, bandit, cpu]
---

# Spike 023b: per-step Lagrangian, the control 023 lacked

## What This Validates

Spike 023's `smooth_pen` beat `lagrangian` (utility +0.068 and +0.091, paired, 60 seeds) but saw
the bound at every step through the training batch's labels, while `lagrangian` updated only
at the predicted tests every 25 steps. This spike gives the Lagrangian the same per-step
information and nothing else.

## Arm (fixed before the first run, 2026-11-27)

`lagrangian_step`: reward `r_i - lam_t v_i` with `lam_t = clip(lam_{t-1} + eta_s c_{t-1}, 0,
20)` (symmetric up and down steps, cap 20 as in 023), where `c` is 023's surrogate (running
rate, smooth Clopper-Pearson at the predicted test's size, minus the threshold). The predicted
tests and the best-feasible-checkpoint rule are unchanged; `LagrangianReward` is not used.

Grid: 023's learning rates x eta_s {1, 10, 100} x lam0 {1, 5}; tuning seeds 1000-1019;
evaluation seeds 0-59 on both policies. Compared with 023's `lagrangian` and `smooth_pen` rows on
the same seeds (the runs are deterministic given seed and settings, so they are not re-run).

## Expectations and attribution rule (written before the first run)

- If `smooth_pen` minus `lagrangian_step` (paired utility) is within 0.03 on both policies,
  023's gain is attributed to **per-step information**: the recommendation becomes "steer on a
  per-step bound signal", and between the two forms the one with the higher grid median is
  preferred.
- If `smooth_pen` beats `lagrangian_step` by more than 2 paired se on both policies, the gain is
  attributed to the **state-free form** as well.
- Anything else is reported as mixed, with the numbers.
- My expectation: per-step information explains most of the gap (difference within 0.03 on at
  least one policy).

## How to Run

    cd ../023-multiplier-free-candidate-selection
    ../../../.venv/bin/python mflab.py --stage tune_b
    ../../../.venv/bin/python mflab.py --stage eval_b
    ../../../.venv/bin/python mflab.py --stage report_b

## Investigation Trail

1. Registration pushed (`b1207b1`) before the arm was written; arm and stages in a separate
   commit before tuning. A rerun of seed 3 reproduced 023's `smooth_pen` and `lagrangian`
   utilities exactly, so 023's rows are reused as registered.
2. Tuning 1,080 runs, evaluation 120 runs, no deviation.
3. Chosen settings: linear lr 0.2, eta_s 1 (lower edge), lam0 5 (upper edge); net lr 0.001,
   eta_s 10, lam0 1 (lower edge).

## Results

Verdict: **VALIDATED: the gain is the state-free form, not the per-step information.**
Tables in `../023-multiplier-free-candidate-selection/results_b.md`.

| Policy | arm | solutions | utility (se) | ckpts over | `smooth_pen` minus this (se) | grid median |
|---|---|---|---|---|---|---|
| linear | `lagrangian` (023) | 52/60 | 0.523 (0.030) | 8.1% | +0.068 (0.030) | 0.535 |
| linear | `lagrangian_step` | 50/60 | 0.515 (0.032) | 10.4% | +0.076 (0.033) | 0.513 |
| linear | `smooth_pen` (023) | 59/60 | 0.591 (0.016) | 0.0% | | 0.561 |
| net | `lagrangian` (023) | 49/60 | 0.518 (0.034) | 10.6% | +0.091 (0.031) | 0.523 |
| net | `lagrangian_step` | 50/60 | 0.520 (0.032) | 5.0% | +0.089 (0.031) | 0.490 |
| net | `smooth_pen` (023) | 60/60 | 0.609 (0.012) | 0.0% | | 0.559 |

- Giving the Lagrangian the per-step surrogate changed nothing measurable: utility 0.515 and
  0.520 against 0.523 and 0.518, and a lower grid median on both policies.
- `smooth_pen` beats it by 2.3 and 2.9 paired se, so by the registered rule the gain is
  attributed to the state-free form. My expectation (per-step information explains most of the
  gap) is **refuted**.
- The per-step Lagrangian's best tuning configuration on the linear policy (0.662 on the tuning
  seeds) fell to 0.515 on the evaluation seeds: its knobs overfit the tuning seeds, which is the
  sensitivity the grid median already shows.
- No returned policy was over the threshold (0 of 120).

Reading: a multiplier carries state, so it lags the policy (it is still high after the rate
has fallen, or low just as it rises: 010's failure). A penalty that is a fixed function of the
current bound cannot lag. On the bandit that is worth about 0.07-0.09 utility and every run
returning a solution.

---
spike: 023b
idea: multiplier-free-candidate-selection
name: per-step-lagrangian-control
type: comparison
validates: "Given spike 023's bandit, seeds and per-step bound surrogate, when the Lagrangian's dual ascent runs at every step on the same surrogate c_t that smooth_pen uses, then we learn whether smooth_pen's gain over the Lagrangian comes from per-step information or from dropping the dual state"
verdict: REGISTERED
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

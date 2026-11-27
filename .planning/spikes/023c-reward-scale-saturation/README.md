---
spike: 023c
idea: multiplier-free-candidate-selection
name: reward-scale-saturation
type: comparison
validates: "Given spike 023's bandit (linear policy) with the base reward multiplied by s in {1, 3, 10}, when the current Lagrangian, 023's smooth penalty (fixed c0 = 20) and a PI hybrid (the smooth penalty plus a slow per-step dual term) are run at their s = 1 settings, then we learn whether the smooth penalty saturates as the reward scale grows and whether the dual term repairs it"
verdict: REGISTERED
related: [023, 023b]
tags: [candidate-selection, lagrangian, saturation, pid-lagrangian, bandit, cpu]
---

# Spike 023c: does the smooth penalty saturate when the reward scale grows?

Raised by an external review of the formulation doc (2026-11-27): the smooth penalty is a
proportional controller, `lam_t = c0 U'(r) sigmoid(c/T) <= c0 U'`, so if the shadow price the
constraint needs exceeds `c0 U'` it can never make the policy feasible. GRPO normalises the
shaped reward within each prompt group, so multiplying the base reward by s is equivalent to
dividing the penalty weight by s: a reward-scale sweep is a direct test of saturation.

## Arms (fixed before the first run, 2026-11-27)

- `lagrangian`: 023's chosen settings (lr 0.4, eta 100, lam0 5, cap 20, as the pipeline ships).
- `smooth_pen`: 023's chosen settings (lr 0.4, c0 20, T 0.02).
- `pi`: `lam_t = lam_dual_t + c0 U'(r_t) sigmoid(c_t / T)` with `lam_dual_t = clip(lam_dual_{t-1}
  + eta_i c_{t-1}, 0, 200)`, the smooth penalty's settings, and eta_i in {0.3, 1, 3, 10} chosen
  on tuning seeds 1000-1019 at s = 1 only.

Base reward scale s in {1, 3, 10}; linear policy; evaluation seeds 0-59; everything else as 023.
No arm is retuned at s = 3 or 10: the question is what happens when the reward scale changes
after tuning, which is the LLM situation.

## Expectations (written before the first run)

1. `smooth_pen`'s solution rate falls as s grows, below `pi`'s at s = 10.
2. `pi` keeps a solution rate within 5 points of its s = 1 value at s = 10.
3. `lagrangian` (cap 20) also degrades at s = 10.
4. At s = 1, `pi` is within 0.03 utility of `smooth_pen` (the dual term costs little).
5. No returned policy over the threshold in more than delta of runs, any arm, any s.

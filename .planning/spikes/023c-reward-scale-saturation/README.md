---
spike: 023c
idea: multiplier-free-candidate-selection
name: reward-scale-saturation
type: comparison
validates: "Given spike 023's bandit (linear policy) with the base reward multiplied by s in {1, 3, 10}, when the current Lagrangian, 023's smooth penalty (fixed c0 = 20) and a PI hybrid (the smooth penalty plus a slow per-step dual term) are run at their s = 1 settings, then we learn whether the smooth penalty saturates as the reward scale grows and whether the dual term repairs it"
verdict: PARTIAL
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

## How to Run

    cd ../023-multiplier-free-candidate-selection
    ../../../.venv/bin/python mflab.py --stage c

## Investigation Trail

1. Registration pushed before the arm was written. One run of 540 evaluation runs plus 80 tuning
   runs, about 30 s; no deviation. eta_i = 10 was chosen, the upper edge of its grid.

## Results

Verdict: **PARTIAL.** The smooth penalty does saturate as the reward scale grows, and the
dual term repairs most but not all of it. Table: `../023-multiplier-free-candidate-selection/results_c.md`.

| Scale | `lagrangian` solutions / utility | `smooth_pen` | `pi` | `pi` minus `smooth_pen`, paired (se) |
|---|---|---|---|---|
| 1 | 52/60, 0.523 | 59/60, 0.591 | 59/60, 0.594 | +0.003 (0.003) |
| 3 | 53/60, 0.565 | 54/60, 0.560 | 51/60, 0.525 | -0.035 (0.033) |
| 10 | 46/60, 0.485 | 40/60, 0.428 | 49/60, 0.521 | +0.092 (0.045) |

Training checkpoints over the threshold at scale 10: `lagrangian` 16.9%, `smooth_pen` 9.2%,
`pi` 3.8%. No returned policy was over the threshold in any of the 540 runs.

Against the expectations: 1 (`smooth_pen` falls, below `pi` at 10) **held**; 2 (`pi` within 5
points of its scale-1 rate at 10) **refuted**, 59 to 49; 3 (`lagrangian` degrades at 10)
**held**, 52 to 46; 4 (`pi` within 0.03 of `smooth_pen` at scale 1) **held**; 5 **held**.

Reading: the review's saturation argument is right. With no dual term, a fixed c0 tuned at one
reward scale loses a third of its solutions at 10 times the scale. The PI form costs nothing at
the tuned scale and is the best of the three at 10, but it does not make the rule scale-free;
eta_i sat on its grid edge, so a larger integral gain may close more of the gap. For an LLM the
recommendation becomes the PI form, with c0 set from the measured reward scale.

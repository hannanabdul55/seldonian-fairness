---
spike: 023
idea: multiplier-free-candidate-selection
name: multiplier-free-candidate-selection
type: comparison
validates: "Given the synthetic bandit under the Seldonian loop (predicted tests, best-feasible-checkpoint selection, one safety test), when LagrangianReward's dual-ascent multiplier is replaced by candidate-selection rules that use the predicted bound directly (feasibility-first switching, a projected gradient step, a fixed smooth penalty scaled by the bound's own derivative, or selection alone), then at least one returns solutions as often, as safely and with at least 90% of the Lagrangian's utility, and is less sensitive to its knobs"
verdict: REGISTERED
related: [002, 010, 012, 022a]
tags: [candidate-selection, lagrangian, projection, feasibility-first, differentiable-bound, bandit, cpu]
---

# Spike 023: candidate selection without the multiplier

## What This Validates

`LagrangianReward` raises a multiplier by dual ascent on the predicted constraint value and
lowers it when the bound has slack. It needs a step size, a down step, a cap, a start value and
(since Round 6) a floor, and two spikes found trouble with it: 010 (an early price withdrawn
under symmetric dual steps, just before the drift) and 002 (three training statistics were the
multiplier in disguise). 022a's `es_direct`, which ranks members by the predicted bound with no
multiplier, returned a solution in 119 of 120 runs.

The question was raised by the user in review of the phase 1 properties doc
(https://claude.ai/artifact/9GXggtJxpp8N24ue3mu8Kd, 2026-11-27) and is the first step of
phase 2: is the multiplier needed at all, once the predicted bound itself is available at every
step and has a derivative (property P12)?

The safety test is identical in every arm, so validity is not at stake; this is a question
about how often a run returns a solution and how good it is (properties P11 and P12).

Limits fixed in advance. The bandit's label is 0/1 through a fixed judge, so (a) judge drift
(017 E8) cannot appear, and (b) the bound depends on the rate alone, so its gradient is a scalar
times the rate gradient: this spike tests how that scalar is chosen, not variance-aware
gradients, which need bounded scores or several constraints.

## Shared machinery: the per-step bound surrogate

Every new arm except `select` uses the same per-step quantity, computed from the training
batch's own judge labels (which the Lagrangian arm already scores, so the label cost is equal):

- running rate `r_t = (1 - a) r_{t-1} + a mean(v_t)`, with `a = 0.2` (about 5 steps, 320
  samples, of memory), started at the first batch's rate;
- smooth predicted upper bound `U(r) = betaincinv(k + 1, n - k, 1 - delta)` with real-valued
  `k = r n`, at `n = effective_n(predict_n, |D_s|)`, the size the predicted test uses. This is
  the Clopper-Pearson upper limit made continuous in the rate. A check script confirms it
  equals `clopper_pearson_bounds` at integer `k` and that its derivative matches finite
  differences;
- `c_t = U(r_t) - threshold` (positive = predicted infeasible) and `U'(r_t)`.

Arms use `c` from the previous step (predictable) when they change the reward of step t.

## Arms

| Arm | Rule | Knobs tuned |
|---|---|---|
| `lagrangian` | the current pipeline: `LagrangianReward`, dual ascent at every predicted test (022a's `grpo` arm) | lr, eta {30, 100, 300}, lam0 {1, 5} |
| `select` | GRPO on the base reward alone; the existing best-feasible-checkpoint rule picks the candidate | lr |
| `feas_first` | Thomas et al. (2019)'s ordering per step: base reward while `c_{t-1} <= 0`, the negated label `-v` (push the bound down) while `c_{t-1} > 0` | lr |
| `projected` | two score-function gradients per step, `g_r` (GRPO's, base reward) and `g_c` (the rate, group-centred labels); step on `d = g_r - k g_c` with `k = max(0, (<g_r, g_c> + eta_p c_t) / |g_c|^2)` (CPO-style, one constraint, closed form), KL term as in GRPO, then Adam | lr, eta_p {1, 10, 100} |
| `smooth_pen` | reward `r_i - lam_t v_i` with `lam_t = c0 U'(r_t) sigmoid(c_t / T)`: a deterministic function of the current bound, no dual state | lr, c0 {5, 20, 50}, T {0.005, 0.02} |
| `es_direct` | reference only: 022a's arm with its tuned settings, not retuned | none |

`projected` applies the projection before Adam's per-coordinate scaling, so the step is not an
exact projection; this is stated as a property of the arm.

## Design (fixed before the first run, 2026-11-27)

- Environment, loop and budget: 022a's `CFG` unchanged (population 20,000, 1,000 prompts split
  60/40, pressure 1.0, threshold = reference rate + 0.03, delta 0.1, Clopper-Pearson, predicted
  test every 25 steps on 512 prompts, 200 steps of 8 prompts x 8 samples, KL beta 0.01).
- Policies: the 36-parameter softmax-linear policy and 022a's 94k-weight tanh net.
- Learning-rate grids: linear {0.05, 0.1, 0.2, 0.4, 0.8}; net {0.0001, 0.0003, 0.001, 0.003}.
  Both contain 022a's GRPO choice (0.4, 0.0003) away from the edge.
- Tuning seeds 1000-1019, never used for evaluation. Selection: mean utility (true reward of the
  returned policy minus the reference's; No Solution Found = 0), as in 022a.
- Evaluation: seeds 0-59, every arm on the same seeds, paired differences against `lagrangian`
  with a paired standard error.
- Outcomes: solution rate; returned policies over the threshold; utility; the share of
  training checkpoints (every 25 steps) over the threshold; runs where the selected checkpoint
  passed the predicted test and failed the safety test (P11); and, on the tuning seeds, each
  arm's median and worst utility over its whole grid (sensitivity to knobs).
- Stage `check` runs before tuning: the smooth bound and its derivative, the projected step
  (`<d, g_c> = -eta_p c` when active), and one run per arm end to end.

## Expectations (written before the first run)

1. `select` returns a solution at least as often as `lagrangian` but with under 80% of its
   utility.
2. `projected` reaches at least 90% of `lagrangian`'s utility with a solution rate no more than
   5 points lower, and no more training checkpoints over the threshold.
3. `smooth_pen`: the same as 2.
4. `feas_first` has fewer training checkpoints over the threshold than `lagrangian` and at
   least 80% of its utility.
5. On the tuning seeds, the median utility over the grid of `projected` and of `smooth_pen` is
   at least `lagrangian`'s.
6. No arm returns a policy over the threshold in more than delta (10%) of runs.

## Decision rule

An arm **replaces the multiplier** if, on both policies, it meets: utility at least 90% of
`lagrangian`'s, solution rate no more than 5 points below, expectation 6, and a grid median at
least `lagrangian`'s. If one or more arms pass, the recommendation is a GPU follow-up on 014's
over-refusal pipeline, registered separately and run only on the user's decision. If none
passes, `LagrangianReward` stays and the question closes as negative on the bandit.

## How to Run

    cd .planning/spikes/023-multiplier-free-candidate-selection
    ../../../.venv/bin/python mflab.py --stage check
    ../../../.venv/bin/python mflab.py --stage tune
    ../../../.venv/bin/python mflab.py --stage eval
    ../../../.venv/bin/python mflab.py --stage report

## Investigation Trail

(empty until the first run)

## Results

(empty until the first run)

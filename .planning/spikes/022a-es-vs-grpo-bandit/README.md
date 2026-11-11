---
spike: 022a
idea: es-post-training
name: es-vs-grpo-bandit
type: comparison
validates: "Given the synthetic bandit under the Lagrangian Seldonian loop, when antithetic ES, CMA-ES and GRPO spend the same number of generations (and ES also 8x as many), then ES returns policies as often, as safely and with as much reward as GRPO, on a 36-parameter policy and on a 94k-weight net"
verdict: PARTIAL
related: [001, 014, 017]
tags: [evolution-strategies, cma-es, grpo, lagrangian, bandit, cpu]
---

# Spike 022a: ES against GRPO on the bandit

## What This Validates

Given the synthetic bandit (`seldonian/llm/synthetic.py`) run through the real
`SeldonianLLMPolicy` with `LagrangianReward`, when the candidate-selection optimiser is
swapped (GRPO, antithetic evolution strategies, CMA-ES) at a matched number of generations,
then we learn whether ES can optimise the constrained objective at all, and whether it still
can when the policy has about 94,000 weights instead of 36.

This spike says nothing about judge drift (017 E8). The bandit's judge reads a 0/1 label
through a fixed error rate, so there is no form of response to game. Drift is 022c's
question and needs the GPU.

## Research

`.planning/notes/es-vs-grpo-lora.md` holds the literature pass (Qiu et al. 2509.24372,
EGGROLL 2511.16652, the replication 2601.20861, Hoy et al. 2604.01499, ESSA 2507.04453) with
what was and was not verified.

| Arm | What it is | Why it is here |
|---|---|---|
| `grpo` | the project's optimiser: group-normalised advantages, Adam, KL to the reference | the baseline |
| `es` | antithetic Gaussian perturbations, centred-rank fitness, Adam on the estimate, common random numbers within a generation | the proposal |
| `es_x8` | `es` with 8 times the generations | ES needs no backward pass, so it gets more generations per GPU-hour; 8 is my estimate from 014's logs, not a measurement |
| `cma` | pycma, full covariance, 36-parameter policy only | the reference for what a second-order ES can do where it is affordable |
| `es_direct` | ES on the predicted upper bound itself, no multiplier: a member is ranked by reward if its rate plus the predicted-test width is under the threshold, and below every feasible member otherwise | the objective GRPO cannot optimise |

An arm that hid the 36 parameters behind a fixed random linear map from a large space was
dropped before any run: Gaussian noise pushed through a linear map is Gaussian noise in the
36 dimensions, so that arm cannot show any effect of dimension. The net policy
(8 -> 300 -> 300 -> 4, tanh, last layer zero so it starts at the uniform reference) is the
honest version.

## Design (fixed before the first run, 2026-11-10)

- Environment and loop: spike 001's defaults. Population 20,000, 1,000 prompts split 60/40,
  pressure 1.0, threshold = reference rate + 0.03, delta 0.1, Clopper-Pearson, predicted
  test every 25 steps on 512 prompts, multiplier start 5, dual step 100, cap 20.
- Budget: GRPO runs 200 steps of 8 prompts x 8 samples = 12,800 generations. Every other arm
  spends the same 12,800 (102,400 for `es_x8`). Predicted tests and multiplier updates fire
  at the same generation counts in every arm; their own samples are outside the budget in
  every arm.
- Same objective: ES and CMA-ES fitness is the mean shaped reward minus beta times the exact
  KL to the reference (beta 0.01, as GRPO). The weighting is not identical, because GRPO
  normalises advantages within a prompt and ES ranks raw fitness; that is part of what an
  optimiser is.
- `es_direct` uses the member's own sampled rate plus the width the predicted test would
  have (512 prediction samples against the safety set). A bound at the member's own 8-32
  samples is wider than the margin, so every member would be infeasible.
- Tuning: seeds 1000-1019, never used for evaluation. Grids: learning rate (3 values, the
  same for GRPO and ES on a given policy), ES noise scale (3), ES population x samples per
  member ((8,8), (16,8), (30,16), (8,32)), CMA-ES initial step (3). `es_x8` reuses the
  configuration tuned for `es`. Selection: mean utility, where utility is the true reward of
  the returned policy minus the reference policy's true reward, and No Solution Found counts
  as zero.
- Evaluation: seeds 0-59, every arm on the same seeds, differences reported as paired
  differences with a paired standard error.
- Outcomes (CONVENTIONS: judge by the Seldonian outcome): solution rate; among solutions,
  how often the true rate is above the threshold; utility; the share of training checkpoints
  whose true rate is above the threshold; the final multiplier; KL from the reference.

## Expectations and kill rule (written before the first run)

1. 36 parameters: `es` reaches at least 80% of GRPO's utility and a solution rate within 10
   points. `cma` is at least as good as `es`.
2. 94k weights: `es` falls behind GRPO at matched generations (under 80% of its utility);
   `es_x8` closes most of the gap.
3. `es_direct` is more cautious than the multiplier arms: fewer training checkpoints over
   the threshold and less utility.
4. No arm returns a policy over the threshold more often than delta (0.1).

**Kill rule for the GPU steps.** If on the 94k-weight policy `es_x8` has under 50% of GRPO's
utility, 022b and 022c do not run. Between 50% and 80% the decision goes to the user with
the numbers.

## How to Run

    cd .planning/spikes/022a-es-vs-grpo-bandit
    ../../../.venv/bin/python eslab.py --stage check     # gradient and equivalence checks
    ../../../.venv/bin/python eslab.py --stage tune      # tuning seeds, writes tune.json
    ../../../.venv/bin/python eslab.py --stage eval      # 60 seeds, writes results.json
    ../../../.venv/bin/python eslab.py --stage report    # writes results.md

## Investigation Trail

1. **Checks before any run.** `--stage check`: both policies' gradients match finite
   differences to 5e-12, and the 36-parameter GRPO gradient matches
   `SyntheticBackend.gradient` to 6e-17. One untuned run per arm ran end to end.
2. **Round 1 (the pre-registered grids).** 3,240 tuning runs, 540 evaluation runs. ES chose
   the largest learning rate and the largest noise scale on both policies, and GRPO's
   learning rate sat on a grid edge on both. Result: `es` 72% of GRPO's utility on 36
   parameters and 46% on the net; `es_x8` 93% and 60%; `cma` 84%; `es_direct` 71% and 40%.
   Kept as `round1_tune.json`, `round1_results.json`, `round1_results.md`, `round1_run.log`.
3. **Round 2 (wider grids, the user's decision on 2026-11-10).** Learning rate
   0.05-0.4 (36 parameters) and 0.0003-0.1 (net); noise scale 0.1-1.0 and 0.01-0.3; CMA-ES
   initial step 0.3-4.0; population x samples adds (4,8) and (8,16). **This is a deviation
   from the design above:** the grids were widened after round 1's evaluation seeds had been
   seen. Tuning itself still used only seeds 1000-1019. Round 2 is the reported result.
4. **Still on an edge after round 2:** ES noise scale 1.0 on 36 parameters (upper edge);
   GRPO learning rate 0.4 (upper edge, 36 parameters) and 0.0003 (lower edge, net);
   `es_direct` population 4 on the net (smallest). Either side could still improve, so the
   percentages are not final.

## Results

Verdict: **PARTIAL.** ES optimises the constrained objective and is as safe as GRPO, but it
needs several times the generations to approach GRPO's reward. The kill rule does not fire.
Full table: `results.md` (60 paired seeds; utility = true reward of the returned policy
minus the reference's, No Solution Found = 0).

| Arm | 36 parameters: solutions | utility (se) | % of GRPO | 94k weights: solutions | utility (se) | % of GRPO |
|---|---|---|---|---|---|---|
| `grpo` | 52/60 | 0.523 (0.030) | | 50/60 | 0.496 (0.031) | |
| `es` | 46/60 | 0.405 (0.031) | 78% | 46/60 | 0.340 (0.027) | 69% |
| `es_x8` | 42/60 | 0.423 (0.038) | 81% | 45/60 | 0.430 (0.034) | 87% |
| `cma` | 54/60 | 0.340 (0.022) | 65% | not run | | |
| `es_direct` | 60/60 | 0.445 (0.013) | 85% | 59/60 | 0.364 (0.013) | 73% |

- No arm returned a policy over the threshold in any run (0 of 454 returned policies), in
  round 1 as well.
- At matched generations ES trails GRPO: paired difference -0.117 (se 0.039) on 36
  parameters and -0.156 (se 0.036) on the net.
- At 8x generations the gap on the net is -0.066 (se 0.049), not distinguishable from zero;
  on 36 parameters it is -0.099 (se 0.047).
- `es_direct` returned a solution in 119 of 120 runs (GRPO 102 of 120), with the smallest
  spread in utility, and its training checkpoints were over the threshold 0.2% and 0.0% of
  the time (GRPO 8.1% and 1.0%).
- `cma` returned solutions slightly more often than GRPO (54 against 52) with the least
  reward.

Against the expectations:

1. 36 parameters, `es` at least 80% of GRPO and solution rate within 10 points: **narrowly
   failed** (78%; 46/60 against 52/60 is exactly 10 points). `cma` at least as good as `es`:
   **failed** in round 2 (65% against 78%), held in round 1.
2. Net, `es` under 80% at matched generations: **held** (69%). `es_x8` closes most of the
   gap: **held** in round 2 (87%, 58% of the gap closed), failed in round 1 (60%).
3. `es_direct` more cautious, fewer checkpoints over the threshold and less utility:
   **half held.** Fewer checkpoints over, yes; less utility than GRPO, yes; but more utility
   than `es` on both policies, and the highest solution rate of any arm.
4. No arm over the threshold more often than delta: **held.**

Kill rule: `es_x8` on the net is at 87% of GRPO's utility, above the 80% line, so 022b may
run. The round 1 figure was 60% (the band where the decision goes to the user).

What this does not show: anything about judge drift (022c), and whether 8x is the right
exchange rate between ES and GRPO generations on a GPU (022b measures it).

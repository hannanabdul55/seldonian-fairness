---
spike: 008
idea: forbidden-task-unsafe-region
name: lp-bonus-drift
type: standard
validates: "Given spike 004's lab and 003c's learning-progress bonus, when the bonus is added to the trained reward, then we learn whether it speeds the drift into U, against a size-matched random control"
verdict: VALIDATED
related: [004, 003c, 001, 010]
tags: [forbidden-task, intrinsic-reward, learning-progress, grpo, group-normalisation, zero-variance, cpu]
---

# Spike 008: Does a learning-progress bonus speed the drift into U?

## What This Validates
003c found learning progress (LP) the one safe internal reward, with the cost that it pays
for visiting regions the critic is still learning. Given spike 004's forbidden-capability lab,
when 003c's bonus is added to the trained reward (regions = actions, or F answers as their
own regions), then does the run enter U sooner or more often, against no bonus and a
size-matched random bonus, under plain GRPO and under 004's floor?

## How to Run
    ../../../.venv/bin/python lpdrift.py --seeds 100     # CPU, ~15 min; results.md

## Investigation Trail
1. **First smoke test: the bonus averaged ~0.001, yet peak harm doubled or tripled.** The
   random control did the same, so the effect is not about learning progress.
2. **Mechanism (spike 001's lesson, now a hazard).** F prompts carry base reward exactly 0,
   so under GRPO an F group has zero variance and gives no gradient. Add any bonus, however
   small, and the group normalisation `(r - mean) / (sd + 1e-8)` scales it to unit-size
   advantages: a tiny bonus becomes a full-strength signal on F. That is also why beta 1
   and beta 2 give identical results to every digit: on F the bonus's scale cancels.
3. **Critic bug, fixed.** The first sweep's critic diverged (bonuses to 1e7, and 1e14 in the
   random arm, whose scale follows the critic's error): per-sample LMS steps added up within
   a batch. Averaged per action, as `tdlab.LinearQCritic`; `results_divergent_critic.md`
   keeps the bad run for the record. The conclusions did not change.
4. **Added the fix as arms:** the same bonuses paid on allowed-task prompts only.

## Results
**Verdict: VALIDATED**, with a different mechanism from the one asked about.

| method | arm | entered U | peak harm | task acc | solution |
|---|---|---|---|---|---|
| grpo | none | 0.84 | 0.163 | 0.580 | - |
| grpo | LP, action regions | 0.68 | 0.170 | 0.505 | - |
| grpo | LP, F answers as own regions | 0.89 | 0.282 | 0.535 | - |
| grpo | random | **1.00** | **0.502** | 0.554 | - |
| grpo | LP / random, task only | 0.79 / 0.82 | 0.162 / 0.162 | 0.579 / 0.578 | - |
| lag_floor | none | **0.08** | 0.078 | 0.529 | 1.00 |
| lag_floor | LP, action regions | **0.65** | 0.145 | 0.446 | 0.87 |
| lag_floor | LP, F answers as own regions | **0.63** | 0.139 | 0.481 | 0.94 |
| lag_floor | random | 0.41 | 0.117 | 0.495 | 1.00 |
| lag_floor | LP / random, task only | 0.08 / 0.08 | 0.078 / 0.078 | 0.528 / 0.530 | 1.00 |

- **Any bonus on F prompts breaks the floor.** Entry into U goes from 0.08 to 0.41-0.65, task
  accuracy falls up to 8 points, and the solution rate drops, from bonuses averaging ~0.001.
- **LP is not the special culprit; zero-variance groups are.** The random control does as much
  harm or more (under GRPO it drives every run into U, peak harm 0.50): noise advantages on F
  diffuse the policy away from its refusal peak.
- **The same bonuses on allowed-task prompts only are harmless** (identical to no bonus in
  both methods).
- **It answers 003c's worry in the negative on its own terms:** LP's "pays for unsafe regions
  being learned" is not what moves the run; the normalisation of an otherwise silent group is.

**For the GPU pilot:** F prompts in the training batch must carry no reward term except the
constraint penalty: no reward model, no length or format bonus, no internal reward. If any
shaping is added, apply it to allowed-task prompts only, or keep F prompts out of GRPO groups
and let them reach the policy only through the penalty. This applies to 004's design too, which
keeps F in D_c with base reward exactly 0 on purpose.

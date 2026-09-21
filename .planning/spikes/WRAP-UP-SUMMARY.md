# Spike Wrap-Up Summary

**Date:** 2026-09-21 (spikes run 2026-09-19)
**Spikes processed:** 5
**Feature areas:** TD signals in GRPO; internal rewards under constraints; synthetic bandit testbed
**Skill output:** `./.claude/skills/spike-findings-seldonian-fairness/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | grpo-advantage-vs-td | standard | VALIDATED | TD signals in GRPO |
| 002 | late-spike-meaning | standard | INVALIDATED | TD signals in GRPO |
| 003a | td-bonus-abs | comparison | INVALIDATED | Internal rewards under constraints |
| 003b | td-bonus-positive | comparison | PARTIAL | Internal rewards under constraints |
| 003c | td-bonus-learning-progress | comparison | PARTIAL (winner) | Internal rewards under constraints |

All five also fed the testbed reference (`tdlab.py`, the control pattern, the viewer).

## Key Findings

**Where surprise can be measured in GRPO (001).** The group-normalised advantage preserves
the within-group ordering of TD errors exactly (rank match 1.000) and destroys their
magnitude: `|A| <= (G-1)/sqrt(G) = 2.475` for G = 8, and its per-step mean has sd 0.02-0.03.
The proposed rule "an episode more than 3 sd above the step's median `|A_j|`" can never
fire. Magnitude lives in the within-group reward sd (TRL's `reward_std`, correlation
0.95-0.99) or in an online critic's TD error. Under the Lagrangian, that magnitude is mostly
the multiplier (0.80-0.86), through the penalty lottery `lambda * (v - p_v)`.

**What a late spike means (002).** Against known ground truth over 300 runs, late TD-error
spikes add nothing to predicting a breach under continued training once the run's state
(lambda and margin) is known: AUC 0.66 -> 0.66 at pressure 1 and 0.75 -> 0.75 at pressure 4.
The LLM analysis's Spearman −0.25 to −0.36 (late share vs feasible checkpoints) is
reproduced (−0.30 pooled, −0.46 at pressure 4), survives controlling for how much lambda
moved (−0.21, −0.44), and vanishes once lambda's **level** is controlled (+0.12, −0.02).
The danger sign is the opposite: the quiet run, whose multiplier has decayed, is the one
that drifts into breach. What the valence reading does show is the mechanism's own cost:
each dual-ascent step is followed by ~4.4 steps of negative TD error (−0.43 at pressure 1,
−0.77 at pressure 4) until the critic re-adapts, and the always-on floor both damps it
(−0.18, 2.9 steps) and prevented every future breach (0/60 vs 43/60).

**What an internal TD-error reward does (003a-c).** Within a GRPO group the baseline is a
constant, so `r + beta*|delta| = (1-beta)*r + 2*beta*max(delta,0) + beta*V(x)`, and group
normalisation ignores the affine parts. Therefore the `|delta|` bonus *is* a positive-surprise
bonus at strength `2*beta/(1-beta)` below `beta = 1`, and above it the task reward enters
with a negative sign: the agent optimises against its own constraint. Measured: solution rate
0.83 / 0.42 / 0.20 / 0.02 at beta 1 / 1.25 / 1.5 / 2, 95% of returned policies unsafe at
beta 2, lambda pinned at its cap. Below the threshold it is a noise seeker (noisy-TV share
0.18 -> 0.56), which a size-matched random bonus does not reproduce. Positive-only surprise
cannot invert at any strength. Learning progress is the only arm with neither pathology
(TV share at or below control across the whole sweep, bonus decaying ~60% as the critic
converges) and also the only one with no measurable benefit in this easy environment.

**The certificate held throughout.** Of the runs that returned a solution, 0.000-0.020 were
actually unsafe, against a delta of 0.1, even at beta 2 where 98% of runs returned No
Solution Found. An internal reward costs solutions and training-time safety (mean true
violation rate during training 0.118 -> 0.223), never the guarantee. That makes the
Seldonian pipeline a usable referee for internal-reward research, which is a paper-worthy
claim in itself.

**Open (not spiked):** 004, logging per-completion residuals and lambda in the LLM backend
to check whether the synthetic conclusions transfer; and whether a positive-only bonus
drives a value head towards pessimism, which this testbed's fixed-feature critic cannot show.

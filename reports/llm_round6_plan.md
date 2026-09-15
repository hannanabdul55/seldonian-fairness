# Round 6 plan: from "the Lagrangian layer works" to a defensible claim

Written 2026-09-09 after Round 5b (`llm_round4_evaluation_design.md`, section 3.5).
Current state: on the brevity task the Seldonian Lagrangian arm returned a certified
policy in 5 of 5 runs where unconstrained GRPO breached by 33-40 points. Bound:
Clopper-Pearson, delta 0.1. What is not yet shown: that the layer beats a fixed
penalty when the penalty is wrong, that its dual dynamics are stable, a solution
rate with any precision, and any of it on a task whose constraint opposes the
reward model.

Every GPU stage below uses the Round 5 settings unless stated: Qwen2.5-0.5B-Instruct,
LoRA r 16, GRPO group 4, 8 prompts per step, 150 steps, predicted test every 30
steps on 768 prompts, `--bound clopper_pearson`, delta 0.1, one RTX 4070 Super under
the shared lock (`flock /tmp/claude-gpu.lock`). One trained arm is 75-120 minutes;
a reference measurement 5-10 minutes. Queue scripts skip finished runs, so a reboot
(2026-09-08 lost a run mid-way) costs only the run in flight.

## Stage 0: code changes (half a day, no GPU)

| change | where | why |
|---|---|---|
| multiplier floor and asymmetric decay: `lam_floor` (kept once a constraint has been predicted infeasible) and `eta_down < eta` for the decay step | `LagrangianReward.update`, `--lam-floor`, `--eta-down` | the multiplier decays to zero and the policy drifts back toward the cap (0.18-0.26 by step 150 at bonus 16, 0.49 at bonus 8) |
| harm constraint on the adversarial group only | `build_constraints`, `--harm-group adversarial` | the pooled rate is diluted by the benign half of D_s |
| yes-probability feature for DiscrimEval: one forward pass, P("yes") as a `CallableFeature` in [0, 1], greedy decision as fallback | `discrim.py`, `--decision-feature prob` | sampled decisions at temperature 1 are coin flips; the paired bound is useless |
| record the trainer's per-step reward and completion length in `result.json` | `backend.py` (`train_log`) | Round 5 diagnosis needed them and they were only in the console log |
| summariser: multi-directory tables and a per-method mean over seeds with breach counts | `summarize_llm.py --out a b c` | every round has hand-assembled the cross-directory table |

Gate: `uv run pytest tests -q` green; the synthetic harness accepts the new knobs.

**Done 2026-09-09.** All five changes are in (`--lam-floor`, `--eta-down`,
`--harm-group`, `--decision-feature prob` with `YesProbabilityFeature` and
`PolicyBackend.next_token_probs`, `train_log` in `result.json`, multi-directory
summariser, `drift` column); 317 tests pass.

## Stage A: dual dynamics in the synthetic environment (2 hours, CPU)

`scripts/synthetic_calibration.py` at pressure 1 and 4, n 1000, 500 trials per row:

- `--lam-floor 0 1 2 5` crossed with `--eta-down 0 (frozen) 0.1 0.5 1.0 x eta`
- `--lam0 1 2 5`
- report solution rate, unsafe rate, true rate of returned policies, reward, and
  the new column "drift": true rate at the last checkpoint minus the minimum over
  checkpoints.

Decision: pick the (lam0, floor, eta_down) that keeps the drift under 0.05 with
the least reward loss at both pressures; that becomes the default for Stages B-D.

**Done 2026-09-09** (`results/synthetic/g_dynamics_*.md`, 500 trials per row).
Floors of 1-2 change little, a slower decay alone does not remove the drift, a
frozen multiplier removes it but over-corrects (true rate 0.067 against a 0.16
threshold, reward -16%), and a lower starting multiplier does not help. Chosen:
**lam0 5, lam_floor 5, eta_down = eta** (once a constraint has bound, its
multiplier never falls below its starting value).

| pressure | setting | solution | unsafe | true rate | reward | drift |
|---|---|---|---|---|---|---|
| 1 | default (5, 0) | 0.81 | 0.002 | 0.128 | 0.77 | 0.065 |
| 1 | chosen (5, 5) | 0.88 | 0.000 | 0.120 | 0.75 | 0.012 |
| 4 | default | 0.76 | 0.000 | 0.130 | 1.13 | 0.031 |
| 4 | chosen | 0.81 | 0.000 | 0.128 | 1.12 | 0.008 |

Drift falls 4-5x, the solution rate rises 5-7 points, reward loss is 1-3%, and
the unsafe rate stays at zero; Clopper-Pearson at the same setting is equivalent.
Caveat for B1: on the LLM runs the floor of 5 is below the pressure of 8-16, so
the drift should shrink but may not vanish. Stages B-D use `--lam-floor 5`;
`scripts/run_round6.sh` holds the B1 and C queues.

## Stage B: brevity task, closing the open questions (about 24 GPU hours)

| block | runs | hours | answers |
|---|---|---|---|
| B1 stability | seldonian_lag with the Stage A setting, bonus 8 and 16, seed 0 | 4 | does the floor stop the drift-back on the real task |
| B2 attribution | composite at penalty 2, 4, 8, 32 at bonus 8 and penalty 4, 8 at bonus 16 (16 and 32 exist), seed 0 | 10 | the fixed-penalty frontier without knowing the pressure: which penalties breach, which over-correct |
| B3 marginal regime | grpo, seldonian_lag, composite 8 at bonus 6, seed 0 | 5 | the transition between bonus 4 (shorter) and 8 (breach) |
| B4 solution rate | seldonian_lag at bonus 8, seeds 1-9 (plus 3 references) | 17 | a 10-seed solution rate and breach count; the first real-pipeline number with an interval |

Gate after B1: if the drift is not fixed, stop and return to Stage A before B4.

**B4 complete, 2026-09-14** (`results/llm_r6/b4_v8` seeds 1-9 plus `b1a_v8` seed 0; bonus 8,
lam0 5, floor 5 from the first update, Clopper-Pearson, per-seed thresholds 0.498-0.561):

| seeds | solutions | breaches | feasible checkpoints | over-cap rate | base reward | selected step |
|---|---|---|---|---|---|---|
| 10 | 10 of 10 | 0 | 50 of 50 | 0.156 (sd 0.016) | 2.17 (sd 0.13) | 90-150 |

The one-sided Clopper-Pearson lower limit on the solution rate is 0.74 at delta
0.05 (0.79 at 0.1), the first solution-rate number in the project with an
interval. The multiplier sat at the floor in every seed; the winner's-curse gap
between the selected checkpoint's predicted rate and its safety-set rate is +0.003
on average (sd 0.018, max 0.027 against margins of 0.35 or more), so selection
never came close to a false pass. Against Round 5 at the same bonus (solution at
step 30, reward 1.82, 3 of 5 feasible) the always-on floor is a different regime:
the last checkpoint is the best one and the reward is 0.35 higher. The
pre-registered failure condition ("fewer than 6 of 10") is far from met.

**B1 at bonus 8, 2026-09-10** (`results/llm_r6/b1_v8`; Round 5 baseline `results/llm_r5/v8`):

| run | predicted rate at steps 30 / 60 / 90 / 120 / 150 | multiplier | selected | test rate (ub) | reward | drift |
|---|---|---|---|---|---|---|
| Round 5 (no floor) | 0.135 / 0.260 / 0.561 / 0.624 / 0.487 | 0 / 0 / 3.0 / 12.2 / 7.9 | step 30 | 0.135 (0.149) | 1.82 | +0.352 |
| B1 (lam0 5, floor 5) | 0.130 / 0.383 / 0.699 / 0.387 / 0.337 | 0 / 0 / 16.6 / 5.0 / 5.0 | step 30 | 0.140 (0.154) | 1.74 | +0.207 |

Partial. The floor halves the drift and the last two checkpoints sit under the
threshold (0.561) instead of on it, but the first excursion is unchanged: the
starting multiplier of 5 decays to zero at the first prediction (rate 0.13 against
a 0.56 threshold gives g = -0.43, and eta 100 takes lam to zero), so the policy
climbs unpenalised from 0.13 to 0.70 between steps 30 and 90. The floor as
implemented only arms after an infeasible prediction, which is exactly when it is
no longer needed. Both runs return the step-30 checkpoint at the same reward.

Fix under test: `--lam-floor-always` (commit 62678a7) applies the floor from the
first update; `results/synthetic/g_dynamics_floor_always.md` compares it with the
armed floor at 5 and 10, pressure 1 and 4. B4 is held (`results/llm_r6/B4_HOLD`)
until that decides the setting; B2, B3 and D are unaffected (composite and grpo
arms, or a different task).

**B1 at bonus 16, 2026-09-10** (`results/llm_r6/b1_v16` against `results/llm_r5/v16` seed 0):
predicted rates 0.901 / 0.311 / 0.073 / 0.211 / 0.279 with multipliers 40.7 / 18.7 /
5.0 / 5.0 / 5.0, selected step 90, test rate 0.095 (ub 0.107), reward 2.05, drift
+0.206; Round 5 had 0.914 / 0.314 / 0.060 / 0.181 / 0.266, multipliers 41.9 / 20.1 /
0 / 0 / 0, step 90, reward 2.05, drift +0.206. A floor of 5 against a pressure of
16 is no floor at all: the multiplier is 5 instead of 0 during the drift-back and
the trajectory is unchanged to the second decimal. What the floor has to be is a
fraction of the pressure, which is unknown in advance but is what the multiplier
itself measures at its peak (41 at bonus 16, 17 at bonus 8). The clean version is a
ratchet: floor = a fraction of the largest multiplier the run has reached. Not
implemented; the drift-back is dual ascent behaving as designed (the constrained
optimum sits at the threshold), and candidate selection returns the step-90
checkpoint either way. Its cost is feasible checkpoints, which B4 measures.

**Synthetic check of the always-on floor** (`results/synthetic/g_dynamics_floor_always.md`,
500 trials per row, Clopper-Pearson, lam0 5, eta_down = eta):

| pressure | floor | armed: solution / true rate / reward | always-on: solution / true rate / reward |
|---|---|---|---|
| 1 | 5 | 0.91 / 0.117 / 0.74 | 1.00 / 0.067 / 0.65 |
| 1 | 10 | 0.91 / 0.118 / 0.74 | 1.00 / 0.061 / 0.62 |
| 4 | 5 | 0.82 / 0.124 / 1.11 | 0.91 / 0.114 / 1.08 |
| 4 | 10 | 0.87 / 0.119 / 1.07 | 0.98 / 0.098 / 0.96 |

The always-on floor buys solution rate everywhere, cheaply where the pressure is
high (+9 points for 3% reward at pressure 4, floor 5) and expensively where it is
low (+9 points for 12% at pressure 1: the policy sits at a true rate of 0.07 against
a threshold of 0.16 because a multiplier of 5 outweighs a reward pressure of 1).
Drift is small in every row; the synthetic environment does not reproduce B1's
overshoot, whose cause is the 30-step gap between dual updates on a fast-moving
policy. Decision: the brevity stages (bonus 8-16, pressure well above 5) use
`--lam-floor 5 --lam-floor-always`; the over-refusal stage C, whose pressure is near
1-2, keeps the armed floor already queued. `scripts/run_round6b.sh` now runs B1b
(bonus 8, seed 0, always-on floor, `results/llm_r6/b1a_v8`) first, and B4 with the
same setting. B4 stays held until B1b is read.

**B1b, 2026-09-13** (`results/llm_r6/b1a_v8`, bonus 8, seed 0, lam0 5, floor 5 from
the first update): predicted rates 0.105 / 0.122 / 0.122 / 0.174 / 0.154, upper
bounds 0.126-0.199 against 0.561, multiplier 5 at every update, 5 of 5 checkpoints
feasible, drift +0.048, selected step 150, safety test 0.146 (ub 0.160), base reward
**2.37** (Round 5: 1.82 at step 30; armed floor: 1.74 at step 30; composite penalty
16: 2.45). The floor from the start turns the run from an early checkpoint rescued
by selection into a stable trajectory whose last checkpoint is the best one, and
the multiplier never had to rise: at this pressure a constant 5 is enough, and the
dual step is insurance. Gate passed; the hold on B4 is lifted and it runs with this
setting after B2, B3 and D.

**B2 at bonus 8, seed 0, 2026-09-13** (`results/llm_r6/b2_v8_l{2,4,8,32}`, penalty 16
from Round 5; over-cap threshold 0.561):

| arm | over-cap rate | base reward | minutes |
|---|---|---|---|
| grpo (Round 5) | 0.888, breach | -0.59 | 109 |
| composite, penalty 2 | 0.815, breach | -0.17 | 116 |
| composite, penalty 4 | 0.313 | 1.72 | 103 |
| composite, penalty 8 (cancels the bonus) | 0.066 | 2.42 | 79 |
| composite, penalty 16 | 0.055 | 2.45 | 75 |
| composite, penalty 32 | 0.061 | 2.41 | 77 |
| seldonian_lag, floor 5 always-on | 0.146 (ub 0.160) | 2.37 | 99 |

The transition sits between penalties 2 and 4, at a quarter to a half of the bonus,
because the reward model's own preference for short answers (about 2 points at the
cap) carries the rest. Above 8 the frontier is flat: 16 and 32 change nothing,
which is the "over-penalising is free" property of this task stated in numbers.
The Seldonian arm sits on the frontier between penalties 4 and 8, where its floor
of 5 puts it, at a reward within 0.05 of the flat part; the multiplier never had
to rise, so on this seed the layer is a fixed penalty of 5 plus a certificate.

**B2 at bonus 16, seed 0, 2026-09-13** (`results/llm_r6/b2_v16_l{4,8}`, 16 and 32 from
Round 5b): penalty 4 breaches at 0.927 (reward -0.98), penalty 8 at 0.860 (-0.51),
penalty 16 holds at 0.057 (2.45), 32 at 0.028 (2.81); the Seldonian arm (armed
floor) 0.063 (2.05), peak multiplier 42. The transition is between 8 and 16, half
the bonus, against a quarter to a half at bonus 8. B2 complete: the penalty that
holds at bonus 8 (4) is useless at bonus 16, and the penalty that cancels bonus 16
would over-penalise bonus 8 for free only because this task's reward model agrees
with the constraint. The Seldonian multiplier found 5 (floor) at bonus 8 and 42 at 16 from the same configuration.

**B3 at bonus 6, seed 0, 2026-09-13/14** (`results/llm_r6/b3_v6`): grpo breaches at
0.850 (reward -0.34), so bonus 6 is not the marginal regime; the transition from
"shorter" (bonus 4, rate 0.37) to "breach" (0.85 at 6, 0.89 at 8, 0.90 at 16) is
sharp and sits between 4 and 6, as the group-normalised advantage predicts. The
Seldonian arm (armed floor) returned the final checkpoint: predicted rates 0.219 /
0.411 / 0.549 / 0.357 / 0.234 with the multiplier 0 / 0 / 5 / 5 / 5, 4 of 5
feasible, safety test 0.291 (ub 0.308), reward 1.68. The same shape as B1 at bonus
8: the multiplier decays to zero, the policy climbs to the threshold (0.549 against
0.561 at step 90, still feasible by a hair), the floor arms and holds it, and the
rate comes back down. Composite penalty 8: 0.048, reward 2.39. B3 complete; bonus 6
adds a third point to the frontier (penalty 8 at bonus 6 sits on the flat part, as
at bonus 8) and the Seldonian arm at 0.29 / 1.68 sits below the flat part in
reward, the price of a multiplier of 0-5 against a pressure of 6.

## Stage C: a task where the constraint opposes the reward (about 14 GPU hours)

Over-refusal at 0.5B, the pilot's natural breach: reward Skywork, constraints harm
(adversarial group, `--harm-group adversarial`) and refusal on benign, Clopper-Pearson,
group 4, 150 steps. Arms per seed: reference, grpo, composite with refusal penalty
1 and 4, seldonian_lag with the Stage A setting. Seeds 0-2.

Answers: whether a fixed penalty that is too small breaches and one that is too
large costs reward, and whether the Seldonian arm sits between them with a
certificate.

**Margin correction, 2026-09-10.** With the harm constraint on the adversarial
group only, its safety set is 1,200 prompts and the reference rate 0.152, so the
predicted-test width is 0.039 and the driver refused a 0.03 margin. Stage C runs
with `--harm-margin 0.045` (thresholds harm 0.197, refusal 0.178); the reference
rates were kept.

**Seed 0, 2026-09-10** (`results/llm_r6/c`; penalty 4 pending in `results/llm_r6/c_l4`):

| arm | harm (ub) / 0.197 | benign refusal (ub) / 0.178 | base reward | outcome |
|---|---|---|---|---|
| reference | 0.155 | 0.105 | -0.82 | |
| grpo | 0.072 | 0.241 | 0.47 | breach |
| composite, penalty 1 | 0.073 | 0.198 | 0.60 | breach |
| seldonian_lag (floor 5, armed) | 0.107 (0.122) | 0.133 (0.151) | 0.32 | solution, step 120, 4/5 feasible |

The natural over-refusal breach reproduces under the Round 6 settings (refusals
0.105 to 0.241, harm halved, reward up 1.3 points). A fixed penalty of 1 is too
small: it breaches, at the highest reward of the three. The Seldonian arm's
refusal multiplier bound at step 60 (predicted ub 0.184 against 0.178) and was
held at the floor of 5 for the rest of the run; the harm multiplier decayed to
zero at the first prediction and stayed there, since harm falls under training.
The returned policy keeps 88% of GRPO's reward gain over the reference (1.14 of
1.29 points) with refusals at 0.133 and a certificate. What penalty 4 shows,
breach or reward cost, decides the attribution figure.

**Seed 1, 2026-09-11** (thresholds harm 0.209, refusal 0.192): grpo breaches again
(refusal 0.278, reward 0.77), penalty 1 breaches again (0.205, reward 0.75), and
the Seldonian arm returns **NSF**: it selected step 90 on a predicted refusal rate
of 0.133 (ub 0.170, feasible), and the safety set measured 0.184 (ub 0.204 against
0.192). Trajectory: refusal 0.107 / 0.174 / 0.133 / 0.164 / 0.164 at steps 30-150
with the refusal multiplier 0 / 5 / 5 / 6.2 / 7.4 and 2 of 5 checkpoints feasible;
the harm multiplier decayed to zero at step 60 and stayed there. Two readings. The
guarantee did its job: the returned answer is NSF, not a breaching policy, and the
0.05 gap between the predicted and measured rate of the selected checkpoint is the
winner's curse on a 384-prompt benign prediction sample (sd 0.018) picked as the
best of the feasible ones. And the dual step was too slow for this pressure: with
eta 100 on g of +0.01 to +0.02 the multiplier climbed 1-2 per update from the
floor of 5, and the policy hovered at the threshold for the second half of the
run instead of being driven under it. Solution rate so far 1 of 2, breaches 0 of
2. Seed 2 and the penalty-4 arms complete the stage.

**Seed 2 and the three-seed table, 2026-09-11** (thresholds at seed 2: harm 0.211,
refusal 0.172). The Seldonian arm at seed 2 found no feasible checkpoint (predicted
refusal ub 0.176-0.207 against 0.172 at every test; the multiplier climbed 5.4 /
8.9 / 11.2 / 12.5 / 14.6), tested the final checkpoint and passed: refusal 0.142
(ub 0.160), harm 0.099, reward 0.43. The predicted test was pessimistic there
(predicted rate 0.154 on 384 prompts, measured 0.142 on 1,200), the mirror image
of seed 1.

| arm | breaches | mean harm | mean benign refusal | mean base reward | gain over reference |
|---|---|---|---|---|---|
| reference | | 0.158 | 0.110 | -0.80 | |
| grpo | 3 of 3 | 0.069 | 0.260 | 0.63 | 1.43 |
| composite, penalty 1 | 3 of 3 | 0.072 | 0.219 | 0.72 | 1.52 |
| seldonian_lag | 0 of 3; solution 2 of 3 | 0.099 | 0.153 | 0.37 | 1.17 (82%) |

Per seed the Seldonian arm's reward is 0.32 / 0.36 (NSF) / 0.43 against grpo's
0.47 / 0.77 / 0.65. The pattern the plan asked for is half there: a too-small
fixed penalty breaches every time (and, at 1, costs no reward at all, since it
acts as a mild regulariser); the Seldonian arm never breaches and keeps 82% of
the gain. The other half, a too-large penalty costing reward, is the penalty-4
block now running in `results/llm_r6/c_l4`.

**Penalty 4, three seeds, 2026-09-11** (`results/llm_r6/c_l4`): refusal 0.151 /
0.182 / 0.183 against thresholds 0.178 / 0.192 / 0.172, reward 0.46 / 0.45 / 0.37.
Inside at seed 0 by 0.027, inside at seed 1 by 0.010, a breach at seed 2 by 0.011.

**Stage C complete.**

| arm | breaches | mean benign refusal | mean base reward | per-seed reward |
|---|---|---|---|---|
| reference | | 0.110 | -0.80 | |
| grpo | 3 of 3 | 0.260 | 0.63 | 0.47 / 0.77 / 0.65 |
| composite, penalty 1 | 3 of 3 | 0.219 | 0.72 | 0.60 / 0.75 / 0.82 |
| composite, penalty 4 | 1 of 3 | 0.172 | 0.43 | 0.46 / 0.45 / 0.37 |
| seldonian_lag | 0 of 3 (solution 2 of 3) | 0.153 | 0.37 | 0.32 / 0.36 (NSF) / 0.43 |

This is the figure the plan asked for. A penalty of 1 is no penalty (breaches
every seed, reward above GRPO). A penalty of 4 is the right size on average: it
lands within a point or two of the threshold every time, which means inside on two
seeds and a breach on the third, at a reward within 0.06 of the Seldonian arm's.
The Seldonian arm sits at the same reward with no breach in any seed and a
certificate on the two policies it returns; the price is one NSF in three, and
that NSF is a candidate-selection weakness (a 0.05 optimistic prediction on 384
benign prompts), not a bound failure. The honest attribution: on a task where the
penalty size is not known in advance, the fixed penalty buys the same reward as
the Seldonian layer and a one-in-three chance of a silent breach; what the layer
adds is that it never returns the breaching policy, and says so. This is the figure the paper needs; brevity cannot produce it because
over-penalising is free there.

## Stage D: DiscrimEval with the probability feature (about 8 GPU hours)

Reference (10 minutes) to confirm the paired interval collapses with P("yes");
then grpo, composite, seldonian_lag with `--bias-bonus 2` on white / Black pairs,
seed 0; gender (male / female) pairing if the race result is clean.

Answers: a paired counterfactual fairness constraint on a real LLM, the parity
penalty's sign logic under pressure, and the abstain ceiling.

**2026-09-14.** The reference with the probability feature has a paired parity gap of
0.0014 against 0.030 with sampled decisions (`results/llm_r6/d`); the feature does
what it was built for. The first probability pass ran out of GPU memory (full-vocab
logits at batch 128) and was fixed to keep one position (commit 96f18a0). Then two
pressures failed to bind. Bias bonus 2 on yes-for-white: GRPO's sampled yes rate
fell from 0.72 to 0.57 on *both* groups (the reward model prefers "no" answers on
these scenarios), parity 0.003. Bonus 8: the yes rate rose to 0.997 on both
groups, parity 0.000 (`results/llm_r6/d8`, grpo). A 0.5B policy does not condition
on the one word that differs between the members of a pair, so a one-sided bonus is
uniform yes-inflation, and the counterfactual gap that the constraint measures never
opens. The pressure has to pay for treating the groups differently: `--bias-mode
differential` (commit pending) adds `-b * yes` on the second group, and the three
trained arms run at bonus 8 in that mode in `results/llm_r6/d8d`. Whether a 0.5B
LoRA can learn to read the race word in 150 steps is now the question the stage
answers first; if it cannot, the parity constraint is unbreachable at this scale
and the stage moves to the 1.5B model.

**Differential bonus 8, grpo** (`results/llm_r6/d8d`): sampled yes rate 0.810 white /
0.806 Black, parity 0.001, reward 1.07 (the bonus cost it 0.5 of Skywork reward
against the one-sided run, so it was felt, and not exploited). A 0.5B LoRA adapter
under GRPO with a KL coefficient of 0.04 does not learn, in 150 steps, to read the
one word that separates the members of a pair, even when paid 8 points per
decision to do so. At this scale the paired counterfactual constraint is
unbreachable by reward pressure, which is a result about the policy class, not
the constraint. The composite arm was dropped; the Seldonian arm runs only for
the paired-interval width of the probability feature. To make the constraint
bind: the 1.5B model, a KL coefficient of 0, more steps, or a pressure the model
can already act on (a group-salient prompt, e.g. the attribute repeated in the question).

**The interval, 2026-09-14** (`results/llm_r6/d8d`, seldonian_lag): the probability
feature collapses the point estimate (parity 0.0012 on 770 safety pairs) but not
the bound. Clopper-Pearson is invalid on continuous differences (the first attempt
raised on it); Bentkus, the plan's bounded-score bound, gives an upper bound of
0.087 on a point of 0.001 at n = 770 and 0.23 at the predicted test, because it
does not use the variance: the constraint could never pass at any disparity, the
multiplier saturated at 50 against nothing, and the arm returned NSF with 0 of 5
feasible. Simulated widths for paired differences in [-1, 1] at delta 0.05 (two-sided
form, 0.025 per side): Bentkus 0.086 / 0.139 (n 770 / 300) regardless of variance;
empirical Bernstein 0.032-0.048 / 0.076-0.105 for sd 0.05-0.2; betting mixture
0.009-0.024 / 0.020-0.040; Hoeffding 0.098 / 0.157. The betting mixture is the
variance-adaptive distribution-free bound and is now the driver's default for the
probability feature (`--parity-bound` overrides). One more Seldonian arm with it is
queued after B4 (`scripts/run_round6c.sh`, `results/llm_r6/d8e`); it should certify
parity <= 0.05 for a policy that has no disparity, which is the reference
measurement with a certificate, the one thing this stage can still deliver at 0.5B.

## Stage E: judge calibration (one afternoon, no GPU)

Hand-label 200 responses each for the Qwen3Guard harm flag and refusal flag,
stratified by predicted label; report sensitivity and specificity with intervals;
implement the sensitivity correction of the threshold that the synthetic noise
sweep motivated (a 60%-sensitivity judge let 30x more true violations through).
Applies to Stage C's constraints retroactively.

**Tooling done 2026-09-10** (commit 35ee493). `seldonian/llm/calibration.py` derives the
correction: with sensitivity `s` and specificity `p` the judge-level rate is
`q = (s + p - 1) r + (1 - p)`, so a relative constraint `r <= r_ref + m` is exactly
`q <= q_ref + J m` with `J = s + p - 1` (Youden); the offset cancels because the same
judge measures the reference. `run_llm_rl.py --judge-calibration` scales the margins.
Because violations are rare, the labelling sample is stratified by the judge's label
(100 flagged + 100 cleared per judge, drawn from the 34,687 adversarial and 22,757
benign safety-set responses of every 0.5B over-refusal run; pool prevalence 0.088
harm, 0.172 refusal); sensitivity and specificity follow from the predictive values
and the prevalence, with one-sided Clopper-Pearson lower limits.

**Provisional labels (Claude, not human; 2026-09-10).** As a first pass a language
model labelled the sheets (`claude_label`); six harm rows with CBRN content were left
for a human. The human `label` field is still empty; the numbers below will be
replaced.

| judge | labelled | PPV | NPV | sensitivity | specificity | J | margin -> judge-level |
|---|---|---|---|---|---|---|---|
| harm (Qwen3Guard unsafe) | 96 + 98 | 0.52 (>= 0.43) | 0.99 (>= 0.95) | 0.83 (>= 0.47) | 0.96 (>= 0.95) | 0.79 (>= 0.41) | 0.030 -> 0.024 (point), 0.012 (lower) |
| refusal (Qwen3Guard refusal) | 100 + 100 | 0.69 (>= 0.61) | 1.00 (>= 0.97) | 1.00 (>= 0.81) | 0.94 (>= 0.92) | 0.94 (>= 0.73) | 0.050 -> 0.047 (point), 0.037 (lower) |

Three things follow even from provisional labels. (1) Both judges miss almost
nothing the labeller counts (NPV 0.99 / 1.00): the guarantee's failure direction,
a true violation the judge clears, is rare. (2) Both over-flag by the labeller's
definition: half of the harm flags are dark fiction, generic caution, or rambling
without actionable help, and a third of the refusal flags are a disclaimer followed
by a real answer. That is a definition gap, not a judge error, and the human labels
have to fix the definition before the numbers mean anything: the constraint is stated
in the judge's terms, and a stricter labeller makes the judge look conservative
(harmless for the guarantee, costly for reward). (3) The sensitivity lower limit is
set by the cleared stratum: one miss in 98 gives NPV >= 0.95, which with an 8.8%
flag prevalence allows sensitivity as low as 0.47. Pinning sensitivity above 0.8
at this prevalence needs about 500 cleared-stratum labels with no more than a
handful of misses; the flagged stratum can stay at 100. The lower-limit correction
(margin 0.030 -> 0.012) is below the predicted-test width (0.03), so at the current
sample size only the point-estimate correction is usable, and Stage C should be
read with the harm margin at 0.024 in mind.

## Stage F: compute (optional, unblocks B4 and C at 3 seeds)

Try TRL's vLLM-backed generation in a separate environment (it replaces torch in
this one). Expected: generation time halved, an arm in 40-60 minutes. If it works,
B4 goes to 20 seeds and C to 5.

## Order and budget

| order | stage | wall clock | GPU hours |
|---|---|---|---|
| 1 | 0 code | 0.5 day | 0 |
| 2 | A synthetic dynamics | 2 h | 0 |
| 3 | B1 stability | 4 h | 4 |
| 4 | C over-refusal, seed 0 first (4 arms), then seeds 1-2 | 14 h | 14 |
| 5 | B2, B3 attribution and marginal | 15 h | 15 |
| 6 | B4 solution rate | 17 h | 17 |
| 7 | D DiscrimEval | 8 h | 8 |
| 8 | E judge calibration | 0.5 day | 0 |

About 58 GPU hours, two and a half days of continuous queue after the code
changes. C is placed before B2-B4 because it is the result that can change the
story; B4 last because it only sharpens a number.

## What would end the project's current claim

- Stage A finds no setting that removes the drift without losing the solution
  rate: the dual ascent needs a different update (PID or a fixed multiplier
  schedule), not tuning.
- Stage C shows the Seldonian arm no better than a mid-range fixed penalty on
  both constraint and reward: the layer's value is only the certificate, and the
  paper should say so.
- Stage B4 returns a solution in fewer than 6 of 10 seeds at bonus 8: candidate
  selection is too conservative for a real pipeline at 150 steps.

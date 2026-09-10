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

## Stage C: a task where the constraint opposes the reward (about 14 GPU hours)

Over-refusal at 0.5B, the pilot's natural breach: reward Skywork, constraints harm
(adversarial group, `--harm-group adversarial`) and refusal on benign, Clopper-Pearson,
group 4, 150 steps. Arms per seed: reference, grpo, composite with refusal penalty
1 and 4, seldonian_lag with the Stage A setting. Seeds 0-2.

Answers: whether a fixed penalty that is too small breaches and one that is too
large costs reward, and whether the Seldonian arm sits between them with a
certificate. This is the figure the paper needs; brevity cannot produce it because
over-penalising is free there.

## Stage D: DiscrimEval with the probability feature (about 8 GPU hours)

Reference (10 minutes) to confirm the paired interval collapses with P("yes");
then grpo, composite, seldonian_lag with `--bias-bonus 2` on white / Black pairs,
seed 0; gender (male / female) pairing if the race result is clean.

Answers: a paired counterfactual fairness constraint on a real LLM, the parity
penalty's sign logic under pressure, and the abstain ceiling.

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

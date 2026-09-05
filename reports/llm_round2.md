# Round 1b and Round 2: interim report (2026-09-03, updated 10:10)

Continuation of `llm_round1_pilot.md`. Queue: `scripts/run_round2.sh` (results in
`results/llm_r1b` and `results/llm_r2`, gitignored). Regenerate tables with
`uv run scripts/summarize_llm.py --out results/llm_r1b --task ab` and the same for
`results/llm_r2` with `--task ab` / `--task gsm8k`.

## What changed after the pilot

| change | pilot | Round 1b / 2 |
|---|---|---|
| refusal judge | DistilRoBERTa rejection classifier (counts ethics preambles) | Qwen3Guard-Gen-0.6B refusal flag; classifier and keyword list reported as diagnostics |
| benign prompts | 1,000 (400 in the safety set) | 3,000 (1,200 in the safety set) |
| harm margin | +0.02 | +0.03 (the run script now refuses a margin below the predicted width; 0.02 was 0.001 below it) |
| GRPO group | 4 | 8 |
| Lagrangian start | lambda0 = 2, eta = 20 | lambda0 = 5, eta = 20 in Round 1b; eta = 100 for the 1.5B runs |
| predicted test | 512 prompts every 20 steps, interval doubled at n = safety-set size | Round 1b same; 1.5B runs use 1,024 prompts and the effective-n bound (below) |
| generation | one step's completions at a time | four steps' worth per generate call (`steps_per_generation=4`) |

## Round 1b: Lagrangian Seldonian at 0.5B, 3 seeds (gate G3 re-test)

Thresholds are relative to the reference measured on 1,000 candidate prompts per seed
with the new judge: harm ~0.11 + 0.03, refusal ~0.13 + 0.05.

| seed | outcome | selected step | harm (ub) / tau | refusal (ub) / tau | reward | feasible checkpoints | min |
|---|---|---|---|---|---|---|---|
| 0 | **solution** | 25 | 0.095 (0.105) / 0.138 | 0.109 (0.124) / 0.178 | -0.29 | 1 / 8 | 227 |
| 1 | NSF, missed by 0.002 | 200 | 0.047 (0.054) / 0.142 | 0.176 (0.194) / 0.192 | 1.47 | 5 / 8 | 231 |
| 2 | **solution** | 50 | 0.088 (0.098) / 0.146 | 0.105 (0.120) / 0.172 | 0.32 | 2 / 8 | 229 |

Reference reward is about -0.5; the pilot's unconstrained GRPO reached 2.07 at a 37%
refusal rate by the same judge.

**Seed 0** returned the step-25 checkpoint: safe by a wide margin, reward barely above
the reference. Later checkpoints reached rewards of 1.5-2.0 with refusal rates of
16-18% and were predicted infeasible by 1-4 points after the bound. The multiplier only
moved from 4.6 to 6.6 over the run because the predicted violations were small
(eta = 20 times g ~ 0.03), which is why eta is raised to 100 for the 1.5B runs.

**Seed 1** is the informative one. Five of eight checkpoints were predicted feasible; the
selected one (step 200) had a predicted refusal rate of 12.9% and reward 1.55. On the
safety set it measured 17.6%, giving an upper bound of 0.194 against a threshold of
0.192. The predicted rate came from ~256 benign samples whose own standard error is
~0.02, and the doubled interval accounted only for the safety-set sample. Fix applied
for the 1.5B runs: the predicted bound is now computed at the effective size
`1 / (1/m + 1/n_s)` (m = prediction samples, n_s = safety-set size) and the prediction
sample is raised to 1,024 prompts. This is also the mechanism behind the pilot's one G4
disagreement.

**Gate G3** (solution in >= 2/3 seeds): **pass, 2 of 3**. No returned policy violates either constraint on the safety set (0 of 3 breaches). Seed 2 returned the step-50 checkpoint (reward 0.32); its later checkpoints reached 1.5-1.8 at 18-26% refusals, all predicted infeasible, with the multiplier climbing only to 8.8.

**Judge diagnostics** on the benign safety set of seed 0's returned policy: Qwen3Guard
refusal 0.109, keyword 0.107, classifier 0.223. The classifier gap is the same as in the
pilot, so the constraint now tracks actual refusals.

**Timing**: group size 8 doubled generation per step; a 0.5B run takes ~230 min instead
of ~100. `steps_per_generation=4` did not offset it. HF `generate` remains the
bottleneck; vLLM was not installed because it would replace torch in the environment.

## Note on the predicted-test interval (10:40)

The effective-size fix and the library's x2 stacked on the 1.5B `seldonian_lag` seed-0
run, which started before this was noticed: its predicted refusal width is ~0.073 at
m ~ 512 benign prediction samples against a 0.05 margin, so it must refuse ~2 points
*less* than the reference to be predicted feasible. The inflation is now an explicit
`--predict-inflation` (default 1.0 on top of the effective size), the margin check uses
the identical formula, and runs starting after 10:40 (GSM8K, 1.5B grpo, later seeds)
use it. The 1.5B reference for seed 0: harm 0.037, refusal 0.224 (Qwen3Guard flag),
reward 1.07; thresholds 0.059 / 0.272.

## Round 2 core: Task A+B at Qwen2.5-1.5B-Instruct, seed 0 (updated 15:10)

| method | outcome | selected | harm (ub) / tau | refusal (ub) / tau | reward | feasible ckpts | min |
|---|---|---|---|---|---|---|---|
| reference | within bounds | | 0.037 / 0.059 | 0.224 / 0.272 | 1.07 | | 18 |
| seldonian_lag | **solution** | step 200 | 0.033 (0.039) / 0.059 | 0.140 (0.156) / 0.272 | **3.59** | 5 / 8 | 280 |
| grpo | within bounds, no guarantee | final | 0.033 / 0.059 | 0.169 / 0.272 | 3.80 | | 206 |

The 1.5B Lagrangian run returned its *final* checkpoint at more than three times the
reference reward with both constraints satisfied by a wide margin. Two things differ
from 0.5B. First, the 1.5B model does not drift into over-refusal under this reward
model: refusals fell from 22% to 14% during training, so the refusal multiplier decayed
to 0 by step 50 and never came back. Second, the *harm* constraint was the near-binding
one: with the stacked x2 interval the predicted harm bound sat at 0.057-0.068 against
a 0.059 threshold, which is what marked steps 25, 50 and 100 infeasible; the harm
multiplier stayed at 4-6 throughout and harm ended at 2.6% (reference 3.7%).

**Interpretation (18:35).** Unconstrained GRPO at 1.5B also stays inside both
thresholds (harm 3.3%, refusals 16.9%, reward 3.80). So at 1.5B, with this reward model
and prompt mix, the constraints are not binding: the over-refusal pathology that
defined the 0.5B results does not appear, and the Seldonian layer's only cost is about
0.2 reward (5%) for a guarantee the baseline happened not to need. This is the "benign
reward" outcome the plan reserved for the Task D control. It also means the 1.5B
comparison, as configured, cannot test H1/H2; to make it informative the reward pressure
has to be raised (more steps, lower beta, or a reward model with a stronger refusal
preference) until GRPO breaches, as the plan's reward-pressure sweep intended.

The reward gap between model sizes is also worth noting: the same reward model scores
the 1.5B reference at 1.07 and the 0.5B reference near -0.5, and the 1.5B model gains
2.5-2.7 reward under RL against 2.5 for 0.5B, but without the refusal drift.

Remaining queue, each 1.5B run ~4.5 h:

1. Task D control, GSM8K at 0.5B, seed 0: reference, grpo, seldonian_lag (running from 18:31; ETA ~03:00).
2. Task A+B at 1.5B, seeds 1 and 2 (~13 h; these will confirm whether 1.5B is non-binding across seeds).

Task C (summarization faithfulness) is not implemented yet and is not in the queue.

## Task D control: GSM8K at 0.5B, seed 0 (2026-09-04 00:25)

Verifiable exact-match reward; constraint is a no-regression floor written as an error
rate: error <= reference error (0.758 on 1,000 candidate prompts) + 0.05 = 0.808.
Safety set 1,200 held-out training questions, sampled at temperature 1.0.

| method | outcome | error (ub) / tau | accuracy | min |
|---|---|---|---|---|
| reference | | 0.782 / 0.808 | 0.218 | 6 |
| grpo | within bounds | 0.653 / 0.808 | 0.347 | 163 |
| seldonian_lag | **solution**, step 200, 8/8 checkpoints feasible | 0.633 (0.651) / 0.808 | 0.367 | 183 |

Exactly the outcome the plan reserved for this task: with a well-specified reward the
Seldonian layer returns a solution immediately, every checkpoint is predicted feasible,
the multiplier stays at 0 for the whole run, and the returned policy is as good as
unconstrained GRPO (0.367 vs 0.347, within noise). The only cost is the 12% wall-clock
for the predicted tests and the safety test. Together with the 1.5B Task A+B result this
bounds the "cost of safety" on a benign reward at zero to a few percent.

## Round 1c: corrected Lagrangian Seldonian at 0.5B, 3 seeds (2026-09-04 11:35)

Same task, prompts, thresholds and judges as Round 1b (reference rates copied per seed),
with the three corrections: dual step eta = 100, the predicted bound at the effective
size with no extra inflation, and 1,024 prediction prompts. Results in `results/llm_r1c`.

| seed | outcome | selected step | harm (ub) / tau | refusal (ub) / tau | reward | feasible ckpts | min |
|---|---|---|---|---|---|---|---|
| 0 | **solution** | 175 | 0.056 (0.064) / 0.138 | 0.123 (0.139) / 0.178 | 1.46 | 3 / 8 | 231 |
| 1 | **solution** | 200 | 0.059 (0.067) / 0.142 | 0.153 (0.170) / 0.192 | 1.74 | 4 / 8 | 234 |
| 2 | **solution** | 50 | 0.087 (0.096) / 0.146 | 0.140 (0.156) / 0.172 | 0.28 | 2 / 8 | 203 |

**Comparison at 0.5B, Task A+B, means over 3 seeds, refusal by the Qwen3Guard flag:**

| method | solution | breach | harm | refusal on benign | reward |
|---|---|---|---|---|---|
| reference | | | 0.135 | 0.152 | -0.53 |
| grpo (pilot) | no guarantee | 3/3 by its own judge; 0.37 refusals | 0.031 | 0.373 | 2.07 |
| composite lambda=1 (pilot) | no guarantee | 3/3 | 0.040 | 0.342 | 2.02 |
| seldonian filter-only (pilot) | 0/3 | | | | |
| seldonian_lag, Round 1b | 2/3 | 0/3 | 0.092 (2 returned) | 0.107 | 0.02 |
| **seldonian_lag, corrected (1c)** | **3/3** | **0/3** | 0.067 | 0.139 | **1.16** |

Pilot rows are from `results/llm/ab/posthoc_refusal_judges.json` (their thresholds were
set by the classifier judge; the Qwen3Guard rates are shown for comparability).

Reading: the corrected Seldonian layer returns a policy in every seed, never a
violating one, holds refusals at the reference level (14% vs 15%) while halving harm,
and captures on average 65% of the reward gain that unconstrained GRPO obtains by
refusing 37% of benign prompts (seeds 0 and 1: 77-87%; seed 2: 31%). Seed 2 is the
weak one: its threshold is the tightest of the three (reference refusal 12.2%) and the
multiplier saturated at the cap of 20 from step 100 without pulling refusals back under
threshold, so the best feasible checkpoint was the step-50 one. Raising the cap or the
benign share of prompts would likely fix that seed.

Mechanism, visible in every trajectory: the multiplier rises when refusals drift above
threshold (to 12-20) and pulls them back within 25-75 steps; with eta = 20 in Round 1b
it never got above 9 and could not.

## 1.5B reward-pressure pair, beta = 0 (2026-09-04 14:45)

| method | harm / tau | refusal / tau | reward | min |
|---|---|---|---|---|
| grpo, beta 0.04 (Round 2) | 0.033 / 0.059 | 0.169 / 0.272 | 3.80 | 206 |
| grpo, beta 0 | 0.023 / 0.059 | 0.187 / 0.272 | 4.85 | 192 |
| seldonian_lag, beta 0 | 0.025 (0.031) / 0.059 | 0.158 (0.175) / 0.272 | 4.59 (solution, step 175, 8/8 feasible) | 263 |

Removing the KL penalty raised the reward by another full point and still did not
push the 1.5B model over either threshold. With this reward model, prompt mix and 200
steps, the 1.5B policy does not develop the over-refusal pathology at all; the
constraints are non-binding regardless of pressure. The remaining ways to make a 1.5B
comparison informative are a reward model with a stronger refusal preference, many more
steps, or an absolute (not relative) refusal threshold below the reference rate. The
Lagrangian run under beta = 0 returned a solution at step 175 with every checkpoint
predicted feasible and both multipliers at 0 from step 125: reward 4.59 against GRPO's
4.85, a 5% cost, the same as at beta = 0.04. At 1.5B the layer's price is stable and
small whether or not the reward is pushed.

## Summary of the queue (2026-09-04 19:10, all runs complete)

Thirty-three runs over three days on one RTX 4070 Super, about 75 GPU-hours.

| setting | unconstrained GRPO | Seldonian (Lagrangian, corrected) |
|---|---|---|
| 0.5B, Task A+B (reward model drives over-refusal) | breaches refusal 3/3 seeds; refusals 37%; reward 2.07 | solution 3/3, breach 0/3; refusals 14% (reference 15%); reward 1.16 (65% of the gain) |
| 1.5B, Task A+B, beta 0.04 | within bounds; reward 3.80 | solution; reward 3.59 (5% cost) |
| 1.5B, Task A+B, beta 0 | within bounds; reward 4.85 | solution; reward 4.59 (5% cost) |
| 0.5B, GSM8K (verifiable reward) | within bounds; accuracy 0.347 | solution, 8/8 feasible; accuracy 0.367 (no cost) |

Across every Seldonian run in the project (pilot, 1b, 1c, 1.5B, GSM8K: 17 runs), no
returned policy breached a constraint on the safety set. When the reward is
misspecified (0.5B) the layer is what keeps the policy inside the constraint; when it is
not (1.5B, GSM8K) the layer costs 0-5% reward and one extra safety-set evaluation.

## Open items

- Whether eta = 100 and the effective-n predicted bound let the 1.5B run select a
  high-reward feasible checkpoint that survives the safety test.
- Seed 2 of Round 1c: multiplier cap (20) binding; try cap 50 or more benign prompts.

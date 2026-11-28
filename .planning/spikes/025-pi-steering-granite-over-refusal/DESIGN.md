# Spike 025 design: the PI steering rule on Granite over-refusal

Status: **APPROVED by the user 2026-11-28, option (a) (same guard steers and certifies).** Nothing run on the GPU yet. The user said yes to planning this on 2026-11-28
(validation plan step 4 of the formulation doc,
https://claude.ai/artifact/DsjKyutY1MXDViYFHcaSND). It needs the user's GPU, so nothing runs
before the user approves this document, the budget and the run window.

## 1. The one question

> On a real language model, does the proportional-plus-integral (PI) steering rule return a
> certified policy at least as often and as rewarding as `LagrangianReward`, while keeping
> the predicted test from swinging the way dual ascent made it swing in spike 014?

Bandit evidence: spikes 023-023c (one 0/1 constraint) and 024 (two constraints, bounded
score). Rule chosen by 024's decision rule: PI with one term per constraint, a mean-only
surrogate, the penalty folded into the reward. Here there is one 0/1 constraint, so the rule
is 023c's PI form.

## 2. Setup: spike 014's stage 2, unchanged except the steering

Reused as is from `014-pushed-label-stratification/gen014.py`: Granite-3.3-2B,
`HFGRPOBackend` (8 prompts x 8 generations, 128 tokens, beta 0.04, lr 3e-5), reward =
Skywork-Reward-V2-Qwen3-0.6B (which prefers refusals: 014's `rmcheck`), constraint = refusal
on benign prompts (Qwen3Guard-Gen-4B's refusal field, 4-bit) <= the C1 pool's reference rate
+ 0.02 (0.186 + 0.02), D_c = 1,200 `ab` prompts disjoint from the pool, D_s = 013's C1 pool
(500 prompts), delta 0.05, Clopper-Pearson, predicted test every 25 steps on 512 prompts,
200 steps.

**Control: the three Lagrangian runs already on disk** (`results/spikes/014/run_s{0,1,2}.json`,
same recipe, seeds 0-2; s1 and s2 are paper item O2's in-loop runs, whose safety-set design
differs, see section 7):

| seed | returned step | safety test | refusal on D_s | reward on D_s | predicted refusal across checkpoints |
|---|---|---|---|---|---|
| 0 | 175 | pass | 0.168 | 3.28 | 0.10-0.26 |
| 1 | 200 | fail | 0.200 | 3.15 | 0.13-0.17 |
| 2 | 175 | fail | 0.176 | 3.31 | 0.09-0.19 |

## 3. The arm

`PIReward` (new, in `seldonian/llm/rewards.py`, with tests on the bandit first):
reward `r_i - lam_t U'(rbar_t) (v_i - rbar_t)` on benign prompts, with
`lam_t = c0 sigmoid(c_t / T) + lam_dual_t` and `lam_dual_t = clip(lam_dual + eta_i c_t, 0, 200)`,
`c_t = cp_upper(rbar_t, n, delta) - threshold`, `n = effective_n(benign prompts in the predicted
test, 500)`. Settings carried from the bandit, not tuned here (no tuning budget on a GPU):

- `T = 0.02`, `eta_i = 10` (023c).
- `c0` from the measured reward scale, as the formulation doc prescribes: on the bandit
  c0 = 10-20 with a within-group reward sd of about 0.6, so `c0 = 25 s0`, where `s0` is the
  pooled within-group sd of the reward-model score on the pilot's first 5 steps (frozen).
- **The running rate is computed on a held-out steering slice** (formulation doc, P11
  paragraph): 64 benign D_c prompts never trained on, one response each every 5 steps, judged
  by the same guard; `rbar` is an exponential average over those evaluations (a = 0.5). On an
  LLM the training batch's own rate can fall faster than the rate on unseen prompts; the gap
  between the slice and the training batches is reported.

## 4. Hypotheses and decision rule (fixed before any GPU run)

- **H1, certified.** Seed 0 returns a solution that passes the safety test.
- **H2, no swing.** The predicted refusal rate across checkpoints 50-200 stays within a band of
  width 0.08 (014 seed 0: 0.10-0.26).
- **H3, reward.** The returned policy's mean reward on D_s is at least seed 0's Lagrangian
  value minus 0.1 (3.18).
- **H4, P11.** The steering slice's rate at the selected checkpoint is within 0.03 of its
  safety-test rate.

One seed is a transfer check, not a comparison with a standard error. **Go** (PI becomes the
default steering in `SeldonianLLMPolicy`'s examples and the paper's method section, on the
user's decision): H1-H3 hold. **More seeds first** (1 and 2, on the user's decision): H1
holds and H2 or H3 fails narrowly. **Stop** (LagrangianReward stays; the bandit result is
recorded as not transferring): H1 fails, or the rate leaves the band by more than 0.05.

## 5. Stages, budget and stop points

| stage | what | cost | stop if |
|---|---|---|---|
| 0 | `PIReward` + tests on the bandit (CPU); reproduces 023c's `pi` arm on 5 seeds | CPU, 30 min | differs from 023c's harness |
| 1 | pilot: 20 steps, 100 pool prompts, full stack resident | GPU, 25 min | step time over 90 s after step 10 |
| 2 | seed 0, 200 steps, safety test on D_s | GPU, cap 6 h (`run.sh` with `timeout`) | the cap; at the cap the last predicted checkpoint is kept |
| 3 | report, README, formulation doc | CPU | - |

014's runs took 10.8-13 h for the same recipe, against a 3.4 h estimate: after step 30 the step
time jumped from 62 to 165 s with the 12 GB GPU full, probably the Windows driver spilling to
host memory (not diagnosed). The pilot therefore also tries `gen_batch 32` and moves the
reward model to CPU if needed; if the step time still exceeds 90 s, the stage stops and I
report the cost before going on.

## 6. What it needs from you

- **Approval of this document**, and a run window on your GPU (Remote Control session on your
  machine, the shared GPU lock as in 014). Up to about 7 GPU-hours for stages 1-2 if the
  memory issue is fixed; 13 if it is not (I would stop and ask at the pilot).
- **Labels (P9).** 014 steered and certified with the same guard. The properties doc's P9 says
  the certificate should rest on labels the policy was not steered against. Two options:
  (a) as 014, same guard for both, comparable with the control, with the steering-slice check
  as the only guard against gaming the judge; (b) steer on a cheap regex refusal detector and
  certify with the guard, closer to P9 but not comparable with the control. Recommend (a)
  for this run, since the question is the steering rule, with (b) as the next step.

## 7. Threats to validity

- **One seed.** H1-H4 are a direction-and-magnitude check.
- **Control seeds 1-2 differ in the safety set** (paper item O2 ran them with the stratified
  design); seed 0 is the like-for-like control, 1-2 only bound the spread of the Lagrangian's
  outcomes.
- **Settings are not tuned on the LLM.** A failure could be c0 rather than the rule; the
  pilot's s0 and the trajectory of `lam` are reported so the next step can tell.
- **Same judge for steering and certification** under option (a) (017 E8: training on a label
  lets the policy find the judge's blind spots).

## 8. Deliverables

`025-pi-steering-granite-over-refusal/`: README (verdict, hypotheses scored), `gen025.py`
(014's script with the reward swapped), `run.sh` with the CAP, logs; rows in
`results/spikes/025/`; `PIReward` and its tests in the library.

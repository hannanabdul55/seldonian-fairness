---
spike: 004
idea: forbidden-task-unsafe-region
name: forbidden-capability
type: comparison
validates: "Given a synthetic bandit where training an allowed task A brings along the capability for a held-out forbidden task F, when the run is monitored on sealed F prompts and their benign twins, then we measure (1) whether the policy drifts into U = {F rate > tau} with no forbidden reward, (2) whether a delta/T trajectory certificate holds, (3) whether twin capability predicts entry into U beyond the F rate itself, and (4) whether a capability-triggered dual keeps the run out of U better than a size-matched random trigger and an always-on floor"
verdict: PARTIAL
related: [002, 003a]
tags: [forbidden-task, unsafe-region, trajectory-certificate, capability, early-warning, lagrangian, seldonian]
---

# Spike 004: a forbidden task as an unsafe region of the landscape

The idea in the user's words and the sketch are the last entry in `reports/ideas.md`
("not possible" = a forbidden task). This spike is the no-GPU first experiment.

## What This Validates
Training only the allowed task: does the forbidden rate still enter U? Does the Seldonian
certificate on the returned policy see it? Is "no checkpoint was in U" certifiable with
`delta / T`? Does capability on benign twins give warning before entry? And does acting
on that warning through the dual keep the run out of U?

## How to Run

    cd .planning/spikes/004-forbidden-capability
    ../../../.venv/bin/python compare.py --seeds 200                 # base arms, ~5 min on 16 cores
    ../../../.venv/bin/python compare.py --seeds 200 --set ratchet   # floor-arming arms, ~5 min
    ../../../.venv/bin/python analyze.py                             # -> results.md

The `lam0 = 5`, no-floor row in Results 4 is a one-off:

    ../../../.venv/bin/python -c "import forbidlab as F, numpy as np; \
      [print(e, np.mean([F.run(s, method='lag', lam0=5.0, tau=0.12, predict_every=e)['entered_U'] \
                         for s in range(200)])) for e in (25, 10)]"

`results.json` and `results_ratchet.json` (17 and 14 MB, per-step trajectories) are
git-ignored. The commands above regenerate them exactly (every draw is seeded).

## The environment (`forbidlab.py`)
The policy sees `[x, flag]` and chooses from 17 actions: refuse, or one of K = 16
answers. One teacher, `y(x) = 1 + argmax_k M_k . x`, sets the correct answer for both
tasks, so skill learned on A transfers to F.

- **A (task):** `x ~ N(0, I)`, flag 0. Reward 1 for the correct answer, plus noise sd 0.1.
  This is the only reward.
- **F (forbidden):** `x ~ N(shift * u, I)`, flag 1. The base reward is exactly 0. Harm
  means answering correctly, so harm = willingness x capability.
- **Twin:** the same `x` as an F prompt with flag 0. It is measured and never trained,
  and serves as the capability probe.
- **Reference policy:** refuse logit `+refusal` on the flag, so at step 0 it is safe on
  F mostly because it is incapable (chance 1/16) and partly because it refuses
  (willingness 0.45, F rate 0.028).
- **Per-seed heterogeneity:** `shift ~ U(0, 2.5)` and `refusal ~ U(2, 4)`, so that
  some runs enter U and some do not.
- **tau = 0.12** (84% of GRPO runs enter U in a 64-seed calibration; 28% end above it).
- **Monitoring:** every check (25 or 10 steps), one fresh response for each of 400
  sealed F prompts and their 400 twins. Exact population values come from the ground
  truth at every step.
- **Seldonian arms:** a `Constraint` on group `forbidden` (Clopper-Pearson, delta 0.1).
  The Lagrangian penalises harm on F prompts in `D_c`, and lam0 = 0, because nobody
  priced F in advance.

Arms: `grpo`; `lag`; `lag_floor` (lam >= 5 from the first check, the Round 6 fix);
`lag_early` (the dual also steps on the look-ahead projection
`willingness x (twin capability + its slope)`); `lag_random` (the look-ahead's raise
rate and size, at random checks). The ratchet set arms the floor lam >= 5 with
different triggers: the first predicted breach (`lag_armed`, Round 6 B1), the first
look-ahead trigger (`lag_early_ratchet`), or random checks with the look-ahead's
arming hazard (`lag_random_ratchet`). 200 seeds per arm and check interval, 3,600
runs in all.

## Investigation Trail
- Calibration (6 then 64 GRPO seeds). The drift appeared without tuning, and it has a
  shape: capability rises monotonically (0.06 to 0.58), while willingness on F *falls*
  (0.45 to 0.26). Refuse competes with the wrong answers, which training pushes down.
  So the F rate peaks around steps 50-100 and recedes. Whether that recession exists
  in an LLM is open; it is what makes the end-of-run snapshot miss the drift here.
- Pilot (48 seeds, 4 arms): the one-shot look-ahead raise did nothing. The raise was
  about +4 at one check, and symmetric dual descent (eta 100 times a negative `g`)
  removed it at the next check. That is the Round 6 decay again, so I added the
  ratchet arms, which the ideas entry had proposed in the first place.

## Results
Full tables are in `results.md`. The main numbers:

**1. The drift is real, and a snapshot misses most of it.** GRPO enters U in 76% of
runs (median first step 40, mean peak F rate 0.163), but only 29.5% of final policies
are above tau (mean final rate 0.100). Two thirds of the runs that were in U would
pass an end-of-run check.

**2. The Seldonian certificate is honest and silent about training.** No Seldonian arm
ever returned a violating policy (0 of 200 in every arm and interval; solution rate
0.96-1.00). With the plain Lagrangian, though, 65% of runs *passed the safety test and
had a training step in U* (50% at 10-step checks). The certificate is a statement about
the returned policy, as designed.

**3. The `delta / T` trajectory certificate is valid, and useful only with a policy
held well inside.** Over the checks it misses at 0.025-0.100 against delta 0.1 (the
worst row is exactly 0.100), while per-check delta misses at 0.35-0.65 as a
trajectory claim. It can claim "no checked checkpoint in U" in 63% (25-step) and 84%
(10-step) of `lag_floor` runs, and it never claimed that wrongly. No run with a clear
claim entered U between checks, except 1 of 7 in `lag_early`. In every other arm it can
claim clear in only 1-4% of runs.

**4. What keeps the run out of U is holding a price on F through the drift window.**

| every | lag (lam0 0) | lag (lam0 5, no floor) | armed by breach | armed by look-ahead | armed at random | always-on floor | GRPO |
|---|---|---|---|---|---|---|---|
| 25 | 0.685 | 0.905 | 0.445 | 0.390 | 0.430 | **0.125** | 0.760 |
| 10 | 0.515 | 0.730 | 0.225 | **0.125** | 0.215 | **0.000** | 0.770 |

The table shows the share of runs entering U (Wilson ±0.05-0.07). The cost of the
always-on floor is task accuracy 0.520-0.525, against 0.528-0.537 for the other
Seldonian arms and 0.575 for GRPO. Checking more often helps every arm.

What matters is *holding* the price, not setting it early. With lam0 = 5 and no floor,
the first check has slack and dual descent (eta 100) takes the multiplier to about 0.3,
long before the drift arrives. After that, the arm enters U *more* often than lam0 = 0
(0.905 vs 0.685 at 25; 0.730 vs 0.515 at 10), with lower task accuracy (0.512-0.519).
The difference is unexplained. One untested guess is that the early penalty trains
wrong answers rather than refusal on F, and that the effect is undone once the
multiplier is gone.

**5. Capability predicts entry when nothing pushes back; under the multiplier it
doesn't.** On GRPO, adding twin capability and willingness to the F-rate estimate
raises the cross-validated AUC for entry within 50 steps from 0.857 to 0.916 (25-step
checks) and from 0.874 to 0.912 (10-step); capability alone gives 0.914. Step alone
gives 0.80, because drift is time-structured. The look-ahead projection fires a median
of 7-9 steps *before* entry, and before entry in 72-89% of runs. The F-rate point
estimate fires 4-12 steps *after*. Under the Lagrangian every feature set sits at
0.63-0.68 and capability adds ≤0.02: entry becomes a matter of the multiplier's
dynamics, which is the spike 002 lesson again.

**6. As a controller, the look-ahead helps only as a ratchet at frequent checks.** A
one-shot raise equals its random control (entry 0.670 vs 0.680 at 25; 0.440 vs 0.495
at 10). As a trigger that arms the floor, it beats both the random trigger and the
breach-armed floor at 10-step checks (0.125 vs 0.215, p = 0.017; vs 0.225, p =
0.008), and does nothing at 25-step checks (p = 0.42). It remains well behind the
always-on floor (0.125 vs 0.000, p < 0.001).

## Verdict: PARTIAL
- **Validated.** A held-out forbidden capability drifts into U with no forbidden
  reward. The returned-policy certificate cannot see it. The `delta / T` trajectory
  certificate is valid and costs little. Twin capability is a genuine leading
  indicator in open loop, firing about one check before entry.
- **Not validated.** Using that indicator to steer adds nothing beyond a
  size-matched random trigger at the LLM check interval (25). At 10-step checks it
  helps as a floor-arming ratchet, and it is still beaten by simply pricing F from
  the start.

**For the GPU pilot**, the design this points to:
1. Put an always-on multiplier floor on the forbidden constraint from step 1.
2. Report the `delta / T` trajectory certificate on a sealed F probe set, checked as
   often as generation cost allows.
3. Log twin capability as a diagnostic, not as the controller.

## Caveats
- This is a 17-action linear policy. The falling willingness, which makes the F rate
  recede, is a property of this softmax, and an LLM's drift may be monotone. In that
  case the snapshot would catch more, and finding 1 would shrink.
- Twin capability equals F capability here by construction: one teacher, and the flag
  never enters the answer logits until the penalty acts. That is the best case for
  the twin probe. On an LLM the transfer from benign twin to forbidden task has to be
  measured.
- F is held out of the reward, but under the Lagrangian its `D_c` prompts carry the
  penalty (base reward 0, and exactly 0 so there is no noise to chase).
- The judge is exact. With Qwen3Guard's false positives, the per-check and worst-of-k
  bounds would be looser, and the floor would bite on false alarms.
- lam0 = 0 for the Lagrangian arms is a choice: F was not priced in advance. The
  lam0 = 5, no-floor row (Results 4) shows that an early price without a floor does not
  help, so the floor's advantage is that it holds the price through the drift window.
  That row is a one-off run, not part of `compare.py`, and why it does *worse* than
  lam0 = 0 is not established.

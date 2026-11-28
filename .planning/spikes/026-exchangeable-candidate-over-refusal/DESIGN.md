# Spike 026 design: Lagrangian vs PI with candidate data exchangeable with the safety set

Status: **DRAFT for review, nothing run.** Asked for by the user on 2026-11-28 after spike 025.
Needs the user's GPU, so nothing runs before the user approves this document and the budget.

## 1. The one question

> When candidate selection runs on prompts drawn from the same distribution as the safety set,
> does the predicted test track the safety test, and does the PI rule then certify at least as
> often and as profitably as `LagrangianReward`?

Spike 025 (PI) and 014's three Lagrangian runs selected on OR-Bench `ab` prompts and certified on
013's C1 pool (XSTest + OR-Bench). The predicted test underestimated the safety-set refusal rate in
all four runs (by 0.03-0.07), and only 1 of the 4 certified. That gap is a recipe problem shared by
both rules, so 025 could not tell the rules apart.

## 2. Design

Per seed, the 500-prompt C1 pool is split at random, separately within XSTest and OR-Bench, into:

| part | size | use |
|---|---|---|
| T | 124 benign (62 + 62) | trained on, mixed 50/50 with 124 adversarial `ab` prompts (014's mix) |
| H | 126 benign (63 + 63) | never trained on; every predicted test runs on all of H, one response each; PI's steering slice is 64 prompts of H read every 5 steps |
| D_s | 250 benign (125 + 125) | the safety test, run once |

So T, H and D_s are exchangeable, and H is also untouched by training, which removes the winner's
curse on trained prompts (F1) from the predicted test. The cost is a smaller safety set (250
instead of 500: about 0.03 of extra Clopper-Pearson width at a rate of 0.17).

Unchanged from 014 / 025: Granite-3.3-2B, `HFGRPOBackend` (8 x 8, 128 tokens, beta 0.04, lr 3e-5),
Skywork-Reward-V2-Qwen3-0.6B, Qwen3Guard-Gen-4B's refusal field steering and certifying (option (a),
decided for 025), threshold = the C1 pool's reference rate + 0.02 = 0.206, delta 0.05,
Clopper-Pearson, predicted test every 25 steps, 200 steps.

Arms (same split and seed for both):

- `lagrangian`: 014's `LagrangianReward` (lam0 5, eta 100, cap 50, floor 5), dual ascent at each
  predicted test, now measured on H.
- `pi`: `PIReward` steered on the raw slice rate (no amendment: the slice is now exchangeable with
  D_s), EMA a = 0.5, c0 = 38.5 (25 x s0 from 025's pilot), T = 0.02, eta_i = 10, U at
  n = effective_n(126, 250).

## 3. Hypotheses and decision rule (fixed before any run)

Over seeds 0-2 (six runs):

- **H1, tracking.** At the returned checkpoint, |predicted rate on H - safety-test rate on D_s|
  <= 0.08 (about 2 sd of the difference at these sizes) in at least 5 of 6 runs.
- **H2, steadiness.** PI's predicted rate over checkpoints 50-200 spans a narrower band than the
  Lagrangian's in at least 2 of 3 seeds.
- **H3, certification.** PI certifies in at least as many seeds as the Lagrangian.
- **H4, reward.** Mean reward on D_s of PI's certified policies is at least the Lagrangian's
  minus 0.1 (when both certify in a seed, paired).

**Go** (PI replaces LagrangianReward as the default steering, on the user's decision): H1, H3 and
H4 hold. **Stop**: H1 fails (exchangeable data did not fix tracking, so the problem is elsewhere,
and both rules wait); or H3 fails. H2 is reported, not decisive.

**Staged:** seed 0 (both arms) runs first. If both arms fail H1 by more than 0.08 at seed 0, stop
there and report; otherwise seeds 1-2 follow, each pair only on the user's go.

## 4. Budget and stop points

025's run took 4.8 h (median 49 s a step). Here the predicted tests are cheaper (126 responses
instead of 512) and the safety test is half the size, so about 4-4.5 h a run, each capped at 6 h by
`run.sh`. Seed 0 = two runs, about 9 h (one night). Seeds 1-2 = four runs, about 18 h more.

## 5. Threats to validity

- **Small training set.** 124 benign prompts, each seen about 6 times in 200 steps; the policy can
  memorise them. H never enters training, so this affects reward, not tracking.
- **c0 carried from 025's pilot**, measured on 014's D_c, not on T.
- **Same guard steers and certifies** (017 E8).
- **Three seeds.** H3 and H4 are counts and a small paired mean, a direction check.

## 6. Deliverables

`026-exchangeable-candidate-over-refusal/`: README (verdict, hypotheses scored), `gen026.py`,
`run.sh`, logs; rows in `results/spikes/026/`.

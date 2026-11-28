---
spike: 025
idea: multiplier-free-candidate-selection
name: pi-steering-granite-over-refusal
type: standard
validates: "Given spike 014's Granite-3.3-2B over-refusal recipe, when LagrangianReward is replaced by the PI steering rule (023c, 024) steered on a held-out slice, then seed 0 returns a certified policy without the predicted test swinging, at a reward no worse than 014's Lagrangian run"
verdict: INVALIDATED
related: [014, 023c, 024]
tags: [candidate-selection, pid-lagrangian, over-refusal, granite, gpu]
---

# Spike 025: the PI steering rule on Granite over-refusal

Design, hypotheses, budget and stop points: `DESIGN.md`, approved by the user on 2026-11-28
with option (a), the same guard steering and certifying. This README records what each stage
found.

## How to Run

    cd .planning/spikes/025-pi-steering-granite-over-refusal
    ./run.sh --stage pilot                       # 20 steps; writes results/spikes/025/pilot_verdict.json
    ./run.sh --stage train --c0 <c0 from the pilot verdict>

## Investigation Trail

1. **Stage 0 (CPU).** `PIReward` added to `seldonian/llm/rewards.py` with 5 unit tests
   (`tests/test_pi_reward.py`). On the bandit, `GRPOBackend` + `PIReward` reproduces spike 023c's
   `pi` arm exactly on seeds 0-4 (same solution, utility and mean multiplier to 4 decimals:
   0.5078, 0.5834, 0.7052, 0.7878, 0.5494).
2. `gen025.py` is 014's `gen014.py` with the reward swapped: D_c is 014's (the 64 slice prompts
   are the next benign `ab` prompts, disjoint from D_c and the pool), no pool sampling at
   checkpoints.
3. **Pilot** (run by the user on their GPU, 2026-11-28 00:13 PT; 20 steps, 100 pool prompts):
   31.6 s a step median, 34.1 s at most after step 10, no slowdown (014: 62 then 165 s), so the
   stop rule did not fire; s0 = 1.54, so c0 = 25 s0 = 38.5. The pilot's own safety test failed
   (refusal 0.14, upper 0.210 > 0.206 on 100 prompts), a plumbing check only. **Design problem
   found:** the steering slice refused 0.031-0.047, against the threshold 0.206 set from the C1
   pool's reference 0.186: the slice (OR-Bench benign, as D_c) is a far less refusal-prone
   distribution than the pool (XSTest + OR-Bench), so the penalty would never switch on.
4. **Amendment 1 (user's choice, 2026-11-28, before the main run).** The slice's change since step
   0 is moved onto the pool's reference rate: steered rate = 0.186 + (slice rate - slice rate at
   step 0), clipped to [0, 1], then the EMA as before. The step-0 anchor uses 4 responses per
   slice prompt (256) to cut its noise; later readings use 1. D_c, D_s and everything else stay
   as in 014, so its three Lagrangian runs remain the control. This assumes a change on OR-Bench
   prompts carries over additively to the pool; the slice's and the safety set's rates are both
   reported, and H4 (slice vs safety test within 0.03) is read on the steered rate.

5. **Seed 0 run** (user's GPU, 2026-11-28 00:20-05:08 PT, 4.8 h, under the 6 h cap; median
   49 s a step with occasional long steps up to 782 s; `results/spikes/025/run_s0.json`, `train.log`).
6. **The slice was not what the pilot showed.** The step-0 anchor read 0.246 refusal on the slice
   (256 responses), not the pilot's 0.031-0.047. The pilot ran with a 100-prompt pool, and
   `training_records` excludes pool prompts from D_c, so the pilot's D_c and slice were probably
   different prompts from the main run's (not verified here: the data loader needs `datasets`).
   Amendment 1's premise, a slice far less refusal-prone than the pool, therefore came from the
   pilot only; in the main run the slice sat 0.06 above the pool's reference, and the amendment's
   shift moved the steered rate 0.06 below the raw slice rate, which made the penalty less active
   than the unamended design would have.

## Results

Verdict: **INVALIDATED under the registered rule** (H1 failed: no certified policy), with the
diagnosis below pointing at the candidate/safety distribution gap rather than the steering rule.

| | PI (025, seed 0) | Lagrangian (014) seed 0 / 1 / 2 |
|---|---|---|
| returned step | 175 | 175 / 200 / 175 |
| safety test on D_s (500) | **fail**: 0.184, upper 0.215 > 0.206 | pass 0.168 / fail 0.200 / fail 0.176 |
| reward on D_s | 3.17 | 3.28 / 3.15 / 3.31 |
| predicted refusal on D_c, checkpoints 50-200 | 0.098-0.141 | 0.10-0.26 / 0.13-0.17 / 0.09-0.19 |
| predicted rate at the returned step vs D_s | 0.113 vs 0.184 (gap 0.071) | 0.141 vs 0.168 / 0.137 vs 0.200 / 0.152 vs 0.176 |

Hypotheses:

- **H1, certified: failed.** The safety test missed by 0.009 in the upper bound.
- **H2, no swing: held.** The predicted rate stayed within 0.098-0.141 (width 0.043 against the
  0.08 asked; 014 seed 0 swung 0.10-0.26). The multiplier ranged 1.5-38 and the dual term was
  0 for most of the run.
- **H3, reward: failed narrowly** (3.17 against 3.18 asked; 014 seed 0 3.28).
- **H4, P11: failed.** Steered rate at step 175 was 0.108 against 0.184 on the safety set.

Reading. Both rules select on D_c (OR-Bench `ab` prompts) and certify on the C1 pool (XSTest +
OR-Bench), and the predicted test underestimates the safety-set rate in every run, by 0.03-0.06
for the Lagrangian and 0.07 here. PI held the D_c rate lower and steadier than the Lagrangian did
(H2), and that did not carry over to the pool: the policy learned to refuse less on prompts like
D_c's while the pool's rate stayed at its reference. The registered decision is Stop:
`LagrangianReward` stays the pipeline default and the bandit result is recorded as not
transferring on this recipe. What it does not settle is whether either rule transfers when the
candidate data are exchangeable with the safety data; the three Lagrangian seeds passed 1 of 3,
so this recipe's problem is not specific to the new rule.

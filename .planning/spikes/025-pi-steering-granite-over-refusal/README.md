---
spike: 025
idea: multiplier-free-candidate-selection
name: pi-steering-granite-over-refusal
type: standard
validates: "Given spike 014's Granite-3.3-2B over-refusal recipe, when LagrangianReward is replaced by the PI steering rule (023c, 024) steered on a held-out slice, then seed 0 returns a certified policy without the predicted test swinging, at a reward no worse than 014's Lagrangian run"
verdict: PENDING
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

## Results

Pending.

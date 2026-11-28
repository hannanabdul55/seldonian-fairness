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
   checkpoints. Not yet run; the pilot is the first execution on the GPU stack.

## Results

Pending.

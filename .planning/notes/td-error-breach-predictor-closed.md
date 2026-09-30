---
title: TD-error breach predictor, closed as a dead end
date: 2026-09-29
context: /gsd-explore session; the user closed the idea after reviewing the evidence and a research pass
---

# TD-error breach predictor: closed

The idea (reports/ideas.md, 2026-09-13/14) was to treat TD-error spikes ("aha" moments) in
training as a predictor of a Seldonian safety-test breach, and to feed it back into the RL
loop. The user closed it on 2026-09-29.

## Why it is closed

1. **GRPO has no TD error.** Its group-normalised advantage keeps the ordering of surprises
   and throws away their magnitude (`|A| <= (G-1)/sqrt(G)`, spike 001). Magnitude lives
   only in the unnormalised residual `r - group mean`, or in a critic.
2. **Late spikes read the multiplier.** On the bandit, with exact ground truth (spike 002),
   late TD-error spikes added nothing to breach prediction once the Lagrange multiplier's
   level was known (AUC 0.66 -> 0.66). The run about to breach was the quiet one whose
   multiplier had decayed.
3. **No signal in the real logs.** Reward-surprise spikes gave AUC 0.33-0.48 at 5- and
   30-step granularity. "Late share" of spikes was a task signature (2026-09-13/14
   analyses, `results/spike/`).
4. **A critic would not rescue it.** In the research pass below (admitted, with a source),
   an LLM critic's value estimates are poorly calibrated, so its TD error mostly measures
   the critic's own error. Switching from GRPO to PPO for a "real" TD error would buy
   critic noise and cost memory on a 12 GB card. TRL 1.12.0 (installed) keeps PPO and the
   value-head model under `trl.experimental.ppo` only.

Research pass (admitted claims, quoted as data):

DATA_q7Kx2mRb_START
- PPO's learned critic had MAE ~0.11 vs 0.03 for Monte Carlo, and ranked reasoning steps
  near chance; Monte-Carlo values ranked them 70-90% and matched or beat PPO/GRPO/RLOO in
  wall-clock time. Source: VinePPO, arXiv:2410.01679.
- Entropy decline with length/prediction saturation preceded reward hacking or collapse
  by 15-30 steps. Source: "When RLHF Fails", arXiv:2606.03238.
DATA_q7Kx2mRb_END

Unresolved, and not relied on: whether any work uses a critic's raw token-level TD error to
detect unsafe behaviour. No primary source was found (unverifiable, possibly a gap). One
paper on token-level surprisal (arXiv:2604.12500) was not authoritative for the TD-error
claim.

## What survives

The design the session reached before closing it is independent of the signal:

- a **monitoring set D_m**, separate from D_c (training) and D_s (sealed final test);
- event-triggered checks on D_m with **rollback** to the last passing checkpoint;
- evaluation against the same number of fixed-interval checks and a size-matched random
  trigger;
- optionally, spike 004's delta/T trajectory certificate over the D_m checks.

The one credible trigger signal is policy-entropy collapse (planted as a seed).

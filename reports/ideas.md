# Ideas to explore

Noted, not planned. Each entry: the idea in the author's words, then a sketch of
how it would map onto this codebase.

## "Aha" moments as TD-error spikes, as a predictor of safety-test breach (2026-09-13)

> Try evaluating the "aha" moments in the model's thinking into a predictive
> model and see how that information could be modeled into the RL training loop
> (post-training) for the LLM. Look at how TD error spikes during an episode can
> be modeled as an aha moment and to see how this can also predict breach of the
> safety test during candidate selection.

Sketch. GRPO has no critic, so there is no TD error as such; the analogues that
exist per episode are (a) the sequence-level reward surprise, the group-normalised
advantage `A_j` (a large positive `A_j` is a completion that beat its siblings by
more than the group's spread), (b) the per-token log-ratio `log pi_theta / pi_ref`
along the completion, whose spikes mark the tokens where the policy has moved
furthest from the reference, and (c) the per-step trainer signals already logged
in `result.json` (`train_log`: reward, reward std, KL, completion length, clip
ratio). An "aha" in the thinking would be a token position where (b) spikes and
the rest of the completion's reward is high; on the brevity task the analogue is
the moment the policy learns to pad or to switch language.

Two uses. First, prediction: build a small model from the training-trajectory
features between two predicted safety tests (advantage-spike rate, KL slope,
entropy drop, reward-surprise concentration) to the outcome of the next predicted
test (feasible or not) and, better, to the gap between predicted and safety-set
rate (the winner's-curse misses of Stage C, seeds 1 and 2). The label exists in
every run already (`history` in `result.json`); Rounds 1-6 give a few hundred
(interval, outcome) pairs. Second, control: if the predictor works, the dual step
does not have to wait 30 steps for the next predicted test; it can raise the
multiplier when the features say a breach is coming, which is the fast-ascent
fix Stage C seed 1 needed without more prediction samples.

Cheapest first experiment: from the existing `train_log` and `history` fields,
does the reward-std or KL trajectory between checkpoints separate the intervals
that ended infeasible from the ones that stayed feasible? No GPU.

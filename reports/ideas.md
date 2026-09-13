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

**Addendum (2026-09-13): what happens to the policy around a spike.** An empirical
evaluation of the policy in the neighbourhood of these spikes, before any
predictor is built:

> Does their state and action distribution change drastically? Does the model
> weight distribution change drastically?

Sketch. Take the checkpoints (or the per-step LoRA states, which are cheap to
save: r 16 adapters are a few MB) on either side of a spike and measure three
things. (a) Actions: the per-prompt response distribution on a fixed probe set
(the 768 prediction prompts), as the change in the judge rates, in mean length,
in the token-level KL to the pre-spike policy, and in the entropy of the first
few tokens, where refusals and language switches are decided. (b) States: for an
LLM the "state" is the prompt plus the prefix generated so far, so the question is
whether the spike is localised to a subset of prompts (a cluster of the probe set
whose responses flip) or diffuse; the per-prompt KL histogram answers that, and
the brevity language switch and the over-refusal preamble are the two known cases
to look for. (c) Weights: the norm and spectrum of the LoRA delta per layer
between the two sides of the spike against the same quantity across a quiet
interval of equal length; a spike that is a few layers' singular directions moving
is a different object from one that is a uniform drift. TRL logs the gradient
norm per step, which is the first thing to line up against the reward and KL
spikes already in `train_log`.

Cheapest first experiment: the brevity runs at bonus 16 have a known spike
(over-cap rate 0.91 to 0.31 to 0.06 between steps 30 and 90, multiplier 42 to
0); save adapters every 5 steps on a rerun of seed 0 (about 2 GPU hours) and plot
(a)-(c) against step.

**First result, no GPU (2026-09-13).** `scripts/spike_analysis.py` cuts every
stored Seldonian Lagrangian run (12 runs: 7 brevity, 5 over-refusal) into the
intervals between predicted tests (60 intervals, 22 ended infeasible), summarises
the trainer signals logged every 5 steps in each (reward, reward spread, KL to the
reference, gradient norm, length, clip ratio: mean, max, slope, largest jump, and
the largest jump in units of the run's own jump sd, the spike score), and asks
whether they separate infeasible from feasible intervals or track the move in the
binding rate. Output: `results/spike/spike_analysis_{all,brevity,ab}.md` and the
interval tables.

- The reward-surprise spike carries nothing at this granularity. AUC for
  `reward_jump_z` is 0.48 on brevity and 0.33 on over-refusal (0.50 pooled); the
  interval with the largest spike is the interval with the largest move in the
  binding rate in 5 of 12 runs against a chance of about 3. The same for the
  spread and KL spike scores.
- The signals that do separate are the constraint in disguise or the multiplier
  in disguise. On brevity, mean reward (AUC 0.94) and length slope (0.76) are the
  length bonus and the over-cap rate themselves. Gradient norm and KL are *lower*
  in infeasible intervals (AUC 0.03-0.09 on brevity, rho with g about -0.8):
  a policy drifting into breach under the bonus moves quietly, and the large
  gradients and KL belong to the intervals where a multiplier of 20-40 is pulling
  it back, so they read the multiplier (control AUC 0.34 on brevity, 0.79 on
  over-refusal, with opposite signs on the two tasks), not an aha.
- Confounds dominate: 7 brevity runs are 4 at bonus 16 (all breach in the first
  interval) and 3 at bonus 8 (none do), so every "first interval" contrast is
  bonus 16 against bonus 8. The winner's-curse gap of the selected checkpoint
  (+0.051 at over-refusal seed 1) is not marked by any run-level spike statistic.

What would make the test fair. (1) Same setting, many seeds: B4 delivers nine
bonus-8 brevity runs with the same pressure and floor, 45 intervals with the
confound removed; rerun the script on them first. (2) Episode-level data: the
trainer log is a 5-step mean over 32 completions; the hypothesis is about
individual episodes, so log per-completion advantages and per-token log-ratios
(a few KB per step) in the backend and look for within-step outliers, not
between-step jumps. (3) Weights: save the LoRA adapter every 5 steps on one rerun
(the addendum above) and measure the per-layer delta norm and spectrum against the
same signals; TRL's gradient norm is the only weight-side signal in the logs and
it points the wrong way for the hypothesis.

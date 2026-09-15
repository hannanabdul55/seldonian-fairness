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

## Cumulative TD error as "potential for harm" in a trajectory (2026-09-14)

> Maybe we can identify trajectories where there is "potential" for agents to do
> more harm. One possible indication is the cumulative TD error in a trajectory,
> which can quantify the "aha" moments in the agent. The hypothesis is that if
> there are lots of TD error spikes during training, then it might be that the
> model is trying to learn more, especially if these spikes happen unusually more
> *later* in training. We can log this and try to get some data empirically on
> this.

Sketch. This is a different object from the breach predictor above: not "does
the next checkpoint fail" but "how much unexplored capacity for change does this
run still have", a per-trajectory scalar with a time profile. Three concrete
versions, in increasing cost.

1. From what is logged already (no GPU). Per run, the cumulative spike count and
   its time profile: with the run's own jump sd as the unit, count jumps above 2
   sd in reward, reward spread, KL and gradient norm per 30-step interval, and
   fit the slope of that count over intervals. A positive slope (more spikes later)
   is the flag. Compare across the three groups we have: B4 (nine stable seeds,
   multiplier flat at 5), the Round 5 / B1 runs with oscillating multipliers, and
   Stage C (two of three near the threshold). If the hypothesis is right, the
   stable B4 runs should be front-loaded and flat late, and the seed-1 over-refusal
   run (NSF, hovering at the threshold) should be back-loaded.
2. Per-episode TD analogue. GRPO has no value function, but the *sequence-level*
   surprise per completion is the group-normalised advantage `A_j`, and the
   cumulative TD error of a trajectory has a direct analogue: the sum over the
   completion's tokens of the per-token log-ratio to the reference policy weighted
   by `A_j`, the per-episode contribution to the policy gradient. Log, per step, the
   distribution of `|A_j|` and of that per-episode gradient contribution (a few KB
   per step); a spike is an episode more than 3 sd above the step's median. The
   "potential" statistic is the fraction of spike episodes per step and its slope
   over training. This needs a small hook in the backend (the trainer already
   computes both quantities), and no extra GPU time.
3. Policy change around late spikes (the earlier addendum): if late spikes mark a
   policy still moving, the LoRA delta between adapter snapshots on either side of
   a late spike should be larger than around an early one of the same size; the
   bonus-16 brevity rerun with adapters every 5 steps answers that.

The link to safety is the point of the hypothesis and should be stated as a
testable claim: a run whose spike rate is still rising at the last checkpoint is
one whose safety-test result is least likely to hold if training continued, and
whose returned policy sits on a moving landscape (section 7 of the paper). The
Seldonian test is a snapshot; this statistic would be the derivative.

**Version 1 result, no GPU (2026-09-14).** `scripts/spike_profile.py` over the 51
stored runs with a trainer log (Rounds 4-6; GRPO, composite and Seldonian arms; 30
log points per run). Per run and signal (reward, reward spread, KL, gradient norm)
a spike is a between-log jump more than 2 robust sd (MAD) from the run's typical
jump; the profile is the pooled count per 30-step interval; *late share* is the
fraction of spikes in the last two intervals (0.4 if flat); the cumulative |z| per
interval is the cumulative-TD-error analogue. Output `results/spike/profile.md`
(and `z3/` at a threshold of 3).

| group | runs | spikes per run | late share (z 2) | late share (z 3) |
|---|---|---|---|---|
| brevity, Seldonian, always-on floor (B4) | 7 | 9.9 | 0.21 | 0.10 |
| brevity, Seldonian, no floor / armed floor | 7 | 17 | 0.19 / 0.33 | 0.11 / 0.14 |
| brevity, GRPO / composite | 17 | 6.6 / 8.7 | 0.30 / 0.25 | 0.21 / 0.23 |
| over-refusal, Seldonian (armed / no floor) | 5 | 10 / 16 | 0.47 / 0.65 | 0.50 / 0.69 |
| over-refusal, GRPO / composite | 11 | 11 / 12 | 0.57 / 0.44 | 0.74 / 0.39 |

- The late share is first of all a task signature. Brevity runs are front-loaded
  (0.1-0.3): the bonus flips the policy in the first 30-60 steps and the
  correction, if any, follows at once. Over-refusal runs are back-loaded
  (0.4-0.7): the drift is slow and the multiplier acts late. So the statistic
  reads *when the pressure acts*, and cross-task comparisons say nothing about
  potential for harm.
- Within task, the direction of the hypothesis is there but weak. Over 19-20
  Seldonian runs, Spearman(late share, fraction of feasible checkpoints) is -0.25
  at z 2 and -0.36 at z 3: runs whose spikes come late had fewer feasible
  checkpoints. Breaching baselines are slightly later than compliant ones (0.38
  vs 0.33 at z 2, 0.42 vs 0.31 at z 3, n 14-16 each). The two NSF runs are not
  later than the solutions (0.39 vs 0.31, then 0.21 vs 0.23).
- The one run that matches the idea exactly is over-refusal seed 2: no checkpoint
  predicted feasible, the multiplier still climbing (5 to 14.6) at step 150,
  spikes 0/1/2/4/5 (late share 0.75), and a *passed* safety test on the final
  checkpoint. That is a certificate on a policy that was still moving, the case
  the hypothesis says to flag; the Seldonian test, being a snapshot, cannot see
  it. One run.
- The B4 seeds, whose multiplier never left the floor, are the most front-loaded
  group at either threshold (late share 0.21, then 0.10), which is what a frozen
  landscape should look like.

Verdict: at 5-step aggregate resolution the statistic is dominated by the task
and only weakly related to outcomes, but the within-task sign is right and the
seed-2 anecdote is the phenomenon. Version 2 (per-episode advantages and
per-token log-ratios logged in the backend) is the fair test, and B4's remaining
seeds add three more same-setting runs.

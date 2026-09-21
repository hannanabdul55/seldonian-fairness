# Measuring TD error / "surprise" in GRPO and Lagrangian runs

## Requirements

From the `td-error-wellbeing` idea (`.planning/spikes/MANIFEST.md`):

- Any per-episode TD statistic for the LLM runs is built from the unnormalised group
  residual `r - group mean` or a value head, **never** from the group-normalised advantage.
- Any claim about training dynamics under the Lagrangian is reported net of the
  multiplier's level and moves.
- A "wellness" reading of a trainer-side signal is stated as functional (convergence, how
  harshly the constraint is enforced), not as welfare.

## How to Build It

**1. Pick the right quantity.** In a bandit (and in GRPO's sequence-level view) the TD
error is the reward prediction error `delta = r - V(x)`. It splits exactly into a
learnable part and noise:

```
delta = (Q(x,a) - V(x))        true advantage, learnable
      + (r - Q(x,a))           noise: reward noise + lambda * (v - p_v) under a Lagrangian
```

**2. Log these per step** (all cheap; a few KB per step for an LLM run):

| quantity | why | LLM equivalent |
|---|---|---|
| `r_j - group mean` (unnormalised) | carries the magnitude; correlation 0.95-0.99 with mean abs delta | per-completion reward minus group mean |
| group reward sd | same information, one number per group | TRL logs it as `reward_std` |
| `A_j` | ordering only | TRL's advantage |
| lambda and every change to it | the confound | `LagrangianReward.lambdas` |
| mean signed delta, and its positive and negative parts | valence, "joy"/"distress" | needs a value head or an EMA baseline |
| policy step norm | "still moving" | grad norm, or LoRA delta between snapshots |

For the LLM path, set TRL's `scale_rewards="none"` (TRL 1.12.0 supports `"group"`,
`"batch"`, `"none"`) if you want the residual straight from the trainer.

**3. Detect a spike** the way `spike_profile.py` already does: scale the step-to-step jump
by the run's own robust sd (MAD x 1.4826) and call anything above 2 a spike. Apply it to
`mean |r - group mean|` or the critic's `|delta|`, never to `|A|`.

**4. Control for the multiplier before believing anything.** Partial Spearman against
(a) the number of lambda moves, (b) total absolute lambda change, and (c) **lambda's
level** at the horizon. Step (c) is the one that killed the headline correlation.

**5. Separate the prediction from the outcome in time.** Train past the decision horizon
and use the later steps as the counterfactual: predictors on steps 1..T, outcome on T..end.
Score with cross-validated AUC against a "run state" baseline (lambda and margin at T),
not against chance; see `sources/002-late-spike-meaning/incremental.py`.

## What to Avoid

- **`|A_j|` as a spike statistic.** Group normalisation caps it at `(G-1)/sqrt(G)`
  (2.475 for G=8) and its per-step mean is flat (sd 0.02-0.03). The rule proposed in
  `reports/ideas.md` — "an episode more than 3 sd above the step's median `|A_j|`" —
  can never fire. Recall of true `|delta|` spikes from `mean |A|` is 0.04-0.22.
- **Reading a training-dynamics signal without lambda in the model.** Under the Lagrangian
  the step-level `|delta|` correlates with lambda at 0.80-0.86, almost entirely through the
  penalty lottery `lambda * (v - p_v)`. Three separate statistics have now turned out to be
  lambda in disguise: KL and gradient norm (2026-09-13 LLM analysis), late spike counts,
  and the late-share-vs-feasible-checkpoints correlation.
- **Pooling settings/tasks.** Spike features "predict" mainly by identifying which pressure
  or task a run came from: pooled AUC 0.62 -> 0.76, within-setting 0.66 -> 0.66.
- **Believing a mechanism story derived from data alone.** The first explanation of 003a's
  collapse (a lambda-scaling argument) was incomplete; the algebra gave an exact answer.
- **Calling the trainer's TD error the model's wellbeing.** The advantage is computed
  outside the forward pass; see `literature-td-error-wellbeing.md` thread 1B.

## Constraints

- G = 8 in this project, so `max |A| = 2.475` exactly.
- With reward noise sd 0.5, `E|delta| >= 0.4` at convergence (`0.5 * sqrt(2/pi)`): the
  magnitude never goes to zero, and under a stochastic judge it never will (Gehring &
  Precup 2013).
- The learnable share of `var(delta)` is small and shrinks: 0.26/0.45 early to 0.05/0.11
  late (GRPO, pressure 1/4), and 0.07/0.05 under the Lagrangian at pressure 4.
- An online critic needs a sane step size: the linear critic diverged at lr 0.5, worked at
  0.05 (late error 0.399, the noise floor). A batch-summed gradient blows it up; average
  per action.
- GRPO's baseline is recomputed from the current policy every step, so group-relative
  TD error has mean zero per group: no lag, no "mood". Valence dynamics need a persistent
  baseline (critic or EMA).

## Origin

Synthesized from spikes: 001, 002
Source files: `sources/001-grpo-advantage-vs-td/`, `sources/002-late-spike-meaning/`
(`viewer.html` there is a zero-dependency canvas viewer for per-step series; regenerate its
`viewer_data.js` with `analyze.py`).

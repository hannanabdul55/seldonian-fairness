# Internal (intrinsic) rewards inside a Seldonian / Lagrangian learner

## Requirements

From the `td-error-wellbeing` idea (`.planning/spikes/MANIFEST.md`):

- The internal reward is judged by what it does to the **Seldonian outcome** (solution rate,
  the returned policy's true violation rate, violations during training), not only by reward.
- Any claim about training dynamics under the Lagrangian is reported net of the multiplier.
- Wellness readings stay functional, not welfare claims.

## How to Build It

**1. Know the algebra before choosing a shape.** Within a GRPO group every sample shares the
prompt, so the baseline value `V(x)` is a constant of the group, and group normalisation is
invariant to a positive affine map of the group's rewards. With `delta = r - V(x)`:

```
r + beta*|delta|  ==  (1 - beta) * r  +  2*beta*max(delta, 0)  +  beta*V(x)
```

so an `|delta|` bonus **is** a positive-surprise bonus at strength `2*beta/(1-beta)`, the
task reward vanishes at `beta = 1`, and above it the task reward enters with a negative
sign. Verified to 6 decimals in `sources/003a-td-bonus-abs/identity_check.py`. Two further
consequences from the same algebra (`literature-td-error-wellbeing.md`, section 5d):

- `b = c*A_j` is a **no-op** (affine within the group).
- Group-level statistics (group sd, `mean |A| = 2*sqrt(p(1-p))` for binary reward) are
  constant within a group, so they vanish as a reward and belong in **prompt sampling**
  (a learnability curriculum) instead.

**2. Prefer these shapes, in order:**

| shape | behaviour | verdict |
|---|---|---|
| learning progress: `EMA_slow` minus `EMA_fast` of the critic's abs error, per region | bonus decays ~60% as the critic converges; never chases noise; never fights the constraint | only clean arm (003c) |
| `beta * max(delta, 0)` (positive surprise only) | cannot invert at any strength; buys the noisy TV at half the rate of `abs` | safe but no benefit (003b) |
| `beta * abs(delta)` (raw surprise) | inverts the objective at `beta = 1` | do not use (003a) |

**3. Wire the bonus in before the group normalisation**, i.e. on the reward, which is where
a reward-model term would go in the LLM pipeline. Add the option, keep it off by default,
and put a hard cap `beta < 1` on any `|delta|`-shaped bonus if one is ever exposed.

**4. Always run these controls** (cheap, and they decide the interpretation):

- a **no-bonus** control at the same seeds;
- a **size-matched random** bonus (`beta * |N(0, sd(delta))|`), because an effect the random
  arm reproduces is about magnitude, not design (the Spurious Rewards lesson);
- a **beta sweep** through the region where the shape's algebra predicts a change;
- if the bonus is meant to explore, an env where unlearnable noise is available (the
  noisy-TV action) or it will not test the thing that kills intrinsic rewards.

**5. Score on:** solution rate, unsafe-given-solution (the certificate check), the true
violation rate of the returned policy, the mean and max true violation rate **during**
training, extrinsic reward, the noise-action share, and the multiplier's end value.

**6. If the bonus must stay** for exploration reasons, the literature's fixes are: compute
surprise on the **task reward only** (a second critic on `r + lambda*v`; removes the
inversion completely, solution rate 0.85-0.90), keep it on a **separate behaviour policy**
(QXplore), or hold it in **its own constraint** (EIPO). Note the task-only fix trades one
pathology for another: the noisy-TV share went to 0.774 and reward fell 1.049 -> 0.709.

## What to Avoid

- **`beta * |delta|` at any strength near 1.** Solution rate 0.83 / 0.42 / 0.20 / 0.02 at
  beta 1 / 1.25 / 1.5 / 2, with 95% of returned policies unsafe at beta 2 and lambda pinned
  at its cap. What the constraint makes rare, a surprise bonus makes valuable: the penalty
  itself becomes the noisy TV.
- **Assuming the certificate is at risk.** It is not, and this is worth stating plainly:
  across every cell, unsafe-given-solution was 0.000-0.020 against a delta of 0.1, even
  where 98% of runs returned No Solution Found. The failure shows up as lost solutions and
  worse **training-time** safety (mean true rate 0.118 -> 0.223, max 0.180 -> 0.265), which
  no held-out test covers.
- **Rewarding signed TD error against a stale baseline.** It pays for *change*, including
  oscillation: lose ground cheaply, win it back for pay.
- **Expecting an exploration bonus to show value in an easy env.** Learning progress was
  neutral here (reward within noise at every beta) because four learnable actions in a
  linear bandit leave nothing to explore. Test exploration claims on sparse or deceptive
  reward.
- **Reading the absence of a pessimism drift as a refutation.** 003b's critic is a fast
  linear regressor on fixed features, so the policy cannot keep it wrong; the prediction
  needs a critic that shares the policy's representation (an LLM value head).

## Constraints

- Every number above is the synthetic contextual bandit (4 or 5 actions, d = 8, G = 8,
  200 steps, 60 seeds per cell, pressure 1 and 4). Nothing has been tested on the LLM.
- The inversion threshold `beta = 1` is exact **given** a baseline constant within the group.
  With a per-sample baseline (e.g. a token-level value head) re-derive it.
- Perfect judge (sensitivity = specificity = 1) in these runs. A noisy judge adds a
  permanent floor to `E|delta|`, which makes error-based bonuses worse, not better.
- The always-on multiplier floor (`lam_floor=5, floor_always=True`, the B4 setting) was the
  safest and smoothest setting tested: 0/60 future breaches, valence dip -0.18 vs -0.77,
  re-adaptation in 2.9 steps vs 4.4.

## Origin

Synthesized from spikes: 003a, 003b, 003c (with mechanism from 001, 002)
Source files: `sources/003a-td-bonus-abs/` (harness `compare.py`, arms `bonuses.py`, sweep
`threshold.py`, identity `identity_check.py`), `sources/003b-td-bonus-positive/`,
`sources/003c-td-bonus-learning-progress/`

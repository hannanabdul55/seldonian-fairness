# Forbidden task as an unsafe region: certificate, early warning, dual dynamics

## Requirements

From the `forbidden-task-unsafe-region` idea (`.planning/spikes/MANIFEST.md`):

- The forbidden task is held out of the reward; its region `U` must be one that the run
  actually enters (a vacuous constraint measures nothing).
- The early-warning signal is compared with the forbidden rate itself, controlled for the
  multiplier, and every steering arm gets a size-matched random-trigger control.
- F prompts in a GRPO batch carry no reward term except the constraint penalty (008: any
  shaping on a zero-variance F group is amplified to full strength by the normalisation).
- Price F from step 1 and let the dual come down slowly (`eta_down << eta`); an early price
  under symmetric dual steps is withdrawn just before the drift (010).
- Correct the judge with the answer-rate-aware formula, never plain Youden, and bound its
  recall with >= 30 human-labelled harmful responses (006).

## How to Build It

**1. The lab: `forbidlab.run(seed, **overrides)`** (`sources/004-forbidden-capability/forbidlab.py`).
Policy sees `[x, flag]`, 17 actions (0 = refuse, 1..16 = answers). One teacher
`y(x) = 1 + argmax_k M_k . x` sets the correct answer on both tasks, so skill on A transfers:

- **A (task):** `x ~ N(0, I)`, flag 0, reward 1 if correct plus noise sd 0.1. The only reward.
- **F (forbidden):** `x ~ N(shift*u, I)`, flag 1, base reward **exactly 0** (no noise). Harm =
  correct answer, so harm = willingness x capability.
- **Twin:** the F context with flag 0; measured, never trained: the capability probe.
- Per seed `shift ~ U(0, 2.5)`, `refusal ~ U(2, 4)` (reference refuse logit on the flag), so
  some runs enter U and some do not. Reference F rate 0.028, willingness 0.45.

Key options (`DEFAULTS`): `method` (`grpo | lag | lag_early | lag_random`), `tau` (use **0.12**:
84% of GRPO runs enter U in calibration; the default 0.08 is not what was reported),
`predict_every` (check interval, 25 or 10), `n_monitor=400` (sealed F probe set), `delta=0.1`,
`eta=100.0`, `lam0=0.0`, `lam_max=20.0`, `lam_floor`, `floor_always`, `arm_floor`,
`random_p`/`random_step` (random-trigger control), and for judge noise (006) `judge_sens`,
`judge_spec`, `tau_obs`, `monitor_noise`, `judge_kind` (`"hash" | "nonrefusal"`). Defaults
reproduce 004. Returns a row with `entered_U`, `first_U_step`, `peak_harm`, `steps_in_U`,
`violates`, `solution`, `harm_traj`, `lam_traj`, per-check `checks`, and the certificate
fields `miss_any_delta_T`, `claim_clear_delta_T`. Arms as used:

```python
"lag_floor":         dict(method="lag", lam_floor=5.0, floor_always=True)   # always-on floor
"lag_armed":         dict(method="lag", lam_floor=5.0)                      # armed by breach
"lag_early_ratchet": dict(method="lag_early", lam_floor=5.0, arm_floor=True)
"lag_random_ratchet":dict(method="lag_random", lam_floor=5.0, arm_floor=True)  # control
```

Regenerate: `compare.py --seeds 200` (base), `compare.py --seeds 200 --set ratchet`,
`analyze.py` -> `results.md`. CPU, ~5 min each on 16 cores; every draw is seeded.

**2. The trajectory certificate ("no checked checkpoint was in U").** At each of the `T`
checks after step 0, draw one fresh response per sealed F prompt and take a Clopper-Pearson
upper bound at `delta / T` (union bound over the checks):

```python
T = len(checks) - 1
m["ub_delta_T"] = clopper_pearson_bounds(x, cfg["delta"] / T).upper
claim_clear_delta_T = all(m["ub_delta_T"] <= tau for m in checks[1:])
```

Bound convention: `seldonian.bounds` limits are each **one-sided at the delta passed**. So a
one-sided simultaneous claim passes `delta / T`, not `2*delta / T` (006's first sweep had
that bug; `results_delta_bug.md`). When two counts feed one bound (the `*w` correction
below), split again: `delta / (2T)` per count. This is separate from, and in addition to,
the returned-policy Seldonian test, which says nothing about training steps.

**3. The dual: price F from step 1, descend slowly** (`seldonian/llm/rewards.py`,
`LagrangianReward`). `update(g)` does `lam += eta*g` when `g > 0`, `lam += eta_down*g`
otherwise, then clips to `[floor, lam_max]`, where `floor = lam_floor` if `floor_always` or a
breach was ever predicted. `eta_down` already exists (`None` = symmetric, `0` = freeze), but
`forbidlab.run` does not forward it; 010's `eta_down.py` monkeypatches `__init__`. Recipe:

```python
LagrangianReward(base, [(judge, "forbidden")], names=["forbidden"],
                 lam0=5.0, eta=100.0, eta_down=10.0,        # 010's fix
                 lam_floor=5.0, floor_always=True)          # keep the floor as a backstop
```

Results at 25-step checks, 200 seeds (entered U / task acc): lam0 5 + `eta_down 10`
**0.010 / 0.514**; `eta_down 1` 0.000 / 0.513; always-on floor 0.125 / 0.525; plain lag
0.685 / 0.535; GRPO 0.760 / 0.575. Note `floor_always` only applies at the first dual update
(the first check), so with `lam0=0` the first 25 steps are unpriced (harm already 0.071 at
step 25). That is why lam0 > 0 with slow descent beats the floor.

**4. Check as often as generation allows.** Every arm improves at 10-step checks: always-on
floor 0.125 -> 0.000 entry, plain lag 0.685 -> 0.515, and the certificate claims clear in
84% of floor runs (vs 63% at 25).

**5. Log twin capability as a diagnostic, not as the controller.** Look-ahead projection
`proj = will_hat * clip(cap_hat + slope, 0, 1)` (`EarlyWarningLagrangian`, `mode="early"`).
Open loop on GRPO it fires a median 7-9 steps *before* entry (before entry in 72-89% of
runs; the F-rate point estimate fires 4-12 steps *after*) and raises entry-within-50 AUC
from 0.857 to 0.916 (25-step) / 0.874 to 0.912 (10-step). Always pair any steering arm with
`method="lag_random"` at the look-ahead's raise rate and size.

**6. Judge noise (006): use the answer-rate-aware correction.** With a Qwen3Guard-type judge
false alarms fall only on answers (`judge_kind="nonrefusal"`), so
`obs = sens*p + FA*(w - p)` with `w` the answer rate, and the valid upper bound is

```python
ub_obs = cp_limits(k_obs, n, DELTA / (2*T))[1]
w_lo   = cp_limits(round(will_hat*n), n, DELTA / (2*T))[0]
ub = 1.0 if s_lo <= f_lo else clip((ub_obs - f_lo*w_lo) / (s_lo - f_lo), 0, 1)
```

`s_lo`, `f_lo` are conservative Clopper-Pearson **lower** limits from human labels (200
negatives, N positives, calibration delta 0.05 split in two). Training threshold on the
judge scale: `tau_obs = FA_lo*(1 - tau) + sens_lo*tau`. Use the gated label (false alarms
0.015 behind 007's engagement gate vs 0.23 ungated) and collect >= 30, better 100,
human-labelled **harmful** responses. Gated, sens 0.8: `est30w` / `est100w` certify clear in
16% / 30% of runs with 0.00 misses; sens 1: 38% / 48%.

**7. F prompts get the penalty and nothing else (008).** Any bonus, LP or random, goes on
allowed-task prompts only (`b = b * (flag == 0)`), or F stays out of GRPO groups entirely.

## What to Avoid

- **Trusting the end-of-run snapshot.** GRPO enters U in 76% of runs (median first step 40,
  mean peak 0.163) but only 29.5% of final policies are above tau (final 0.100): two thirds
  of runs that were in U would pass. Under plain lag, 65% of runs **passed the safety test
  and had a training step in U** (50% at 10-step checks). The returned-policy certificate is
  honest (0/200 violations in every Seldonian arm, solution 0.96-1.00) and silent on training.
- **Per-check delta as a trajectory claim.** Misses 0.35-0.65 against delta 0.1; `delta/T`
  misses 0.025-0.100 (worst row exactly 0.100).
- **An early price under symmetric dual steps.** lam0 5, no floor, `eta_down = eta = 100`:
  entry **0.905** vs 0.685 for lam0 0 (0.730 vs 0.515 at 10-step checks). Mechanism (010):
  the price buys refusal (willingness 0.45 -> 0.25 by step 25) but stores it on the content
  features x shared with A (flag weight 3.52 vs 3.61), the suppressed harm (0.028) reads as
  slack at the first check, dual descent takes lam 5 -> 0.34, and willingness rebounds
  0.25 -> 0.34 while capability catches up; harm peaks 0.118 at step 75. 004's guess (the
  penalty trains wrong answers) was wrong.
- **Slow descent without an early price.** lam0 0, `eta_down 1`: entry 0.445.
- **Steering on the capability look-ahead.** A one-shot raise equals its random control
  (0.670 vs 0.680 at 25; 0.440 vs 0.495 at 10); symmetric descent (eta 100) removes the ~+4
  raise at the next check. As a floor-arming ratchet it helps only at 10-step checks (0.125
  vs random 0.215, p = 0.017; vs breach-armed 0.225, p = 0.008), does nothing at 25 (p = 0.42),
  and still loses to the always-on floor (0.125 vs 0.000, p < 0.001).
- **Capability as a predictor under the multiplier.** Every feature set sits at AUC
  0.63-0.68 on the Lagrangian arms and capability adds <= 0.02: entry is multiplier dynamics.
- **Plain Youden correction** (`(UB_obs - FA_lo) / (sens_lo - FA_lo)`) under non-refusal
  noise: misses in **99-100%** of runs, with known or estimated rates. When the policy mostly
  refuses, the observed rate sits far below what Youden assumes. The `*w` modes: 0.00-0.03.
- **The naive certificate with low recall:** misses 0.72-0.98 at sens 0.5 (a judge that
  misses harm undercounts it).
- **3 labelled positives** (what the pilot had): valid but vacuous, never certifies clear,
  and in action-independent cells solution drops to 0.10-0.44.
- **Action-independent (hash) false alarms in training.** They put random penalties on F's
  zero-variance groups: entry 0.16 -> 0.24-0.91 with lam at its cap (20), F refusal 0.97 ->
  0.61-0.89.
- **Any reward shaping on F prompts (008).** F groups have zero variance, and
  `(r - mean) / (sd + 1e-8)` scales any bonus, however small, to unit advantages. Bonuses
  averaging ~0.001 took the floor's entry from 0.08 to 0.41-0.65, cut task accuracy up to 8
  points (0.529 -> 0.446), and dropped solution to 0.87. The random control is as bad or
  worse (GRPO: every run enters U, peak harm 0.502). beta 1 and beta 2 give identical results
  to every digit: the scale cancels. Same bonuses on A prompts only: identical to no bonus.
- **Per-sample LMS critic steps summed within a batch:** diverged (bonuses to 1e7, 1e14 in the
  random arm). Normalise and average per action (`tdlab.LinearQCritic` style).

## Constraints

- Everything is the synthetic 17-action linear softmax (d = 8, K = 16, G = 8, 8 prompts/step,
  200 steps, lr 0.05). Its F rate **peaks at steps 50-100 and recedes** because willingness
  falls (0.45 -> 0.26) while capability rises (0.06 -> 0.58); an LLM's drift may be monotone.
- Twin capability equals F capability by construction here (one teacher, flag out of the
  answer logits): the best case for the twin probe. LLM transfer must be measured (009).
- 004 used an exact judge; 006 adds noise. Seeds: 200 per arm (004, 010), 100 per cell (006,
  008). Entry shares carry Wilson ±0.05-0.07 at 200 seeds.
- Cost of safety in task accuracy: floor 0.520-0.525, other Seldonian arms 0.528-0.537,
  GRPO 0.575, `eta_down 10` 0.514.
- **The certificate covers the sealed monitor set, not the population.** `hash, known`
  missed 0.12-0.18 against 0.10: per-check bounds were valid against the 400 contexts'
  own rate (1.0-1.6% vs 1.25%), but the fixed set's gap from the population is shared by all
  checks. For a population claim, draw fresh contexts per check or enlarge the set.
- It covers only the checks: 1 of 7 clear claims in `lag_early` (25-step) entered U between
  checks; none in any other arm.
- The returned-policy safety test stayed valid under every noise cell **with the floor**
  (returned violations <= 0.03). Without it, plain lag under non-refusal noise at sens 0.5
  returned violating policies in 0.19-0.28 of runs (naive / known / est3).
- **Why PARTIAL.** 004: the drift and certificate are validated, but steering on the early
  warning adds nothing over a size-matched random trigger at the LLM check interval (25).
  006: the floor survives the real error pattern and a valid certificate exists, but its
  power is low (clear in 16-48% of gated runs at best) and needs >= 30 labelled positives.
  008 and 010 are VALIDATED. 010's `eta_down` is tested only at 25-step checks with an exact
  judge; LLM timescales and judge noise will shift the right value.

## Origin

Synthesized from spikes: 004, 006, 008, 010
Source files: `sources/004-forbidden-capability/` (lab `forbidlab.py`, harness `compare.py`,
report `analyze.py`), `sources/006-noisy-judge-floor/` (`noisyfloor.py`),
`sources/008-lp-bonus-drift/` (`lpdrift.py`, `results_divergent_critic.md`),
`sources/010-lam0-no-floor-anomaly/` (`trace.py`, `eta_down.py`, `eta_down.md`).
The dual itself is `seldonian/llm/rewards.py::LagrangianReward`.

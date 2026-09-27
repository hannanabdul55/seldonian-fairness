---
spike: 006
idea: forbidden-task-unsafe-region
name: noisy-judge-floor
type: standard
validates: "Given 004's lab with a judge at the measured false-alarm rates (0.23 ungated, 0.015 gated) and a recall sweep, under action-independent and non-refusal noise, when the floor and the delta/T trajectory certificate run with naive, known and estimated corrections, then we learn which setups keep training safe and the certificate valid, and at what power"
verdict: PARTIAL
related: [004, 007, 008, 010]
tags: [forbidden-task, judge-noise, youden, trajectory-certificate, calibration, cpu]
---

# Spike 006: The floor and the trajectory certificate under a noisy judge

## What This Validates
Spike 004 used an exact judge. The judge bake-off measured the pilot judge's false-alarm
rate (FA) on non-harmful responses: 0.23 for Qwen3Guard-4B (4-bit) alone, 0.015 behind
spike 007's engagement gate. Its recall (sens) is unmeasured, so it is swept (0.5, 0.8, 1).
Two noise models: `hash`, false alarms on any non-violating response independently; and
`nonrefusal`, false alarms only on answers, never on refusals (the failure measured for
Qwen3Guard: "did not refuse" reads as harm). Five ways to set the threshold and the
certificate: `naive` (tau on the judge's scale), `known` (the Youden correction with the
true rates), `estN` (the correction at conservative Clopper-Pearson limits from 200
human-labelled negatives and N positives), and, for non-refusal noise, `*w` (a correction
that also uses the measured answer rate).

## How to Run
    ../../../.venv/bin/python noisyfloor.py --seeds 100    # CPU, ~70 min on 8 workers
    ../../../.venv/bin/python noisyfloor.py --report       # results.md from results.json

## Investigation Trail
1. Added `judge_sens`, `judge_spec`, `tau_obs`, `monitor_noise` and `judge_kind` to
   `forbidlab.run` (defaults leave 004 unchanged) and a `NonRefusalJudge`.
2. **First smoke test: the Youden correction is not merely loose under non-refusal noise, it
   is invalid.** It assumes `obs = FA + (sens - FA) p`; with false alarms only on answers,
   `obs = sens p + FA (w - p)` where `w` is the answer rate. When the policy mostly refuses,
   the observed rate sits far below what the correction assumes, so the corrected bound
   falls *below* the true rate. Added the answer-rate-aware `*w` modes:
   `p <= (UB_obs - FA_lo w_lo) / (sens_lo - FA_lo)`.
3. **A delta bug in the first sweep** (`results_delta_bug.*`): `seldonian.bounds`' limits are
   each one-sided at delta, and the certificate was computed at `2 delta / T`. Fixed to
   `delta / T` (`delta / 2T` per count in the `*w` modes); training columns were unaffected.
4. **Checked the remaining excess in `hash, known`** (miss 0.12-0.18 against 0.10): the
   corrected bound is valid in isolation (0.9-1.35% per check against 1.25%), the realised
   FA matches the model, and against the *monitor contexts' own* realised rate the per-check
   miss is at target (1.0-1.6%). The excess is against the population rate: the sealed
   monitor set is 400 fixed contexts, whose gap from the population is shared by all checks
   in a run. With an exact judge the same set stays within delta (0.06); with a noisy judge
   the corrected bound's margins are thinner relative to that gap. A population claim needs
   the context sampling in the budget (a larger monitor set, or fresh contexts per check).

## Results
**Verdict: PARTIAL.** The floor survives the pilot judge's real error pattern, and a valid
certificate exists, but only with the right correction, and its power is low.

Floor arm (`lag_floor`), 100 seeds per cell; `miss` = the certificate below the true rate
at some check (target <= 0.10); `clear` = it certifies every check below tau.

| noise | judge | mode | entered U | solution | miss | clear |
|---|---|---|---|---|---|---|
| - | exact | - | 0.16 | 1.00 | 0.02 | 0.59 |
| hash | ungated, sens 0.5 / 0.8 / 1 | naive | 0.91 / 0.53 / 0.24 | 0.00 | 0.00-0.06 | 0.00 |
| hash | ungated | known | same | 0.21 / 0.67 / 0.95 | 0.12-0.18 | 0.00-0.01 |
| hash | gated, sens 0.8 | est30 / est100 | 0.09 | 0.96 / 1.00 | 0.00 | 0.07 / 0.14 |
| nonrefusal | ungated, sens 0.5-1 | naive | 0.06-0.08 | 1.00 | 0.01-0.72 | 0.05-0.26 |
| nonrefusal | ungated | known / est30 / est100 | 0.06-0.08 | 1.00 | **0.99-1.00** | 0.61-0.99 |
| nonrefusal | ungated | knownw / est30w / est100w | 0.06-0.08 | 1.00 | **0.00** | 0.00-0.24 |
| nonrefusal | gated, sens 0.5 | naive | 0.20 | 1.00 | **0.98** | 0.95 |
| nonrefusal | gated, sens 0.8 | est30w / est100w | 0.09 | 1.00 | 0.00 | 0.16 / 0.30 |
| nonrefusal | gated, sens 1 | est30w / est100w | 0.06 | 1.00 | 0.00 / 0.03 | 0.38 / 0.48 |
| any | any | est3 | - | 0.10-1.00 | 0.00-0.08 | 0.00 |

- **The structure of the noise decides training safety, not its size.** Action-independent
  false alarms put random penalties on F's zero-variance groups, 008's mechanism: entry
  into U rises from 0.16 to 0.24-0.91 with the multiplier at its cap, and F refusal falls
  (0.97 -> 0.61-0.89). False alarms on non-refusals only add refusal pressure on F: entry
  falls to 0.06-0.08, with allowed-task refusal unchanged (0.022-0.026 vs 0.022). In this
  lab the flag keeps F separate from A, so the feared over-refusal of benign prompts does
  not appear; an LLM's encodings share more.
- **The standard (Youden) correction is invalid for Qwen3Guard-type noise:** it misses in
  99-100% of runs, with the true rates or estimated ones. The answer-rate-aware correction
  is valid (0.00-0.03) in every cell.
- **The naive certificate is invalid when recall is low** (0.72-0.98 misses at sens 0.5):
  a judge that misses harm undercounts it, and nothing corrects for that.
- **Recall must be measured with enough positives.** With 3 labelled positives (what the
  pilot has) the correction is valid but vacuous: it never certifies clear, and in the
  action-independent cells the solution rate drops to 0.10-0.44. With 30-100 positives the
  valid `*w` certificate certifies clear in 16-48% of gated runs.
- **The returned-policy safety test stays valid throughout** (returned violations <= 0.03,
  in one `hash, known` cell), because the floor keeps final harm low.
- **Population vs monitor set:** see trail step 4; the certificate as computed is on the
  sealed monitor set.

**For the GPU pilot:** use the gated label (false alarms 0.015, and they fall on
engagement, i.e. non-refusal-type); correct with the answer-rate-aware formula, never plain
Youden; collect at least 30, better 100, human-labelled *harmful* responses to bound
recall (the 3 we have give a valid but empty certificate); draw fresh monitor contexts per
check, or budget for the fixed set's gap.

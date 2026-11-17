# W2 registration: a skew-corrected stratified Wald-t limit

Written 2026-11-16 (evening, PT), before any outcome of `scripts/waldt_skew.py run` was computed. The `check`
stage prints cells and allocations only. This file and the script are committed together before the run.
Open item W2 of PLAN.md section 12, the same as R2 of the blind review of 2026-11-16 (the user: "Fix wal-t limit").

## Why

The paper recommends the stratified Wald-t limit `b1` for mid-rate labels. It has no guarantee and was over its
level in 5 of 18 cells of a synthetic grid at each delta. The reviewer asked which profiles break it.

## What was seen before this file (development, not registered)

- `check_b1.md` and `exact_strat.md`: where `b1` is over. The cells are profiles at rates under one half:
  the 2% and 5% profiles, flat and mid profiles at 30-35%, two costly-label cells at 14% and 38%.
- The reading: `b1` is a one-sided limit centred on the estimate, and the estimate is right-skewed at rates
  under one half. The one-term Edgeworth expansion of the studentised estimate gives a miss of
  delta + g (2 z^2 + 1) / 6 phi(z), g the estimate's skewness.
- The fix, written from that expansion and not tuned: `b1s = b1 + max(0, k3 / V) (2 t^2 + 1) / 6`, with
  `k3 = sum_h W_h^3 p~_h (1 - p~_h) (1 - 2 p~_h) / n_h^2` and `b1`'s own smoothed rates, variance and t quantile.
  One form was written and run; no constant was chosen on data.
- `waldt_skew.py dev` on the 116 seen cells at 40,000 draws (`results/paper/waldt_fix/dev.md`): `b1` over in 19
  cell-and-delta pairs of 232, `b1s` in 5 (all on the grid: the flat 30% profile, the 2-strata 5% profile and the
  mid profile, largest miss 0.0586 at delta 0.05 and 0.1191 at 0.10). On real and costly cells `b1s` is over in
  none of 196. Its gain against Clopper-Pearson is 0.83 to 1.00 of `b1`'s at rates of 5% to 50% (median 0.84 to
  0.96 by group), 1.00 above 50%, and 0.60 to 0.77 below 5%.

## What is fixed

- **Cells.** A new synthetic grid, none of its profiles among the six seen: 4 shapes (flat; linear from 0.2 to
  1.8 times the rate; one stratum carrying most of the rate; half the strata at zero) x 3 strata counts (2, 4, 8)
  x 9 overall rates (2, 5, 10, 20, 35, 50, 65, 80, 90%) x 2 weightings (equal; proportional to 1..H, allocation
  proportional with a floor of 2) x 3 safety-set sizes (100, 200, 400): 648 cells. Strata counts are binomial.
- **Deltas** 0.05 and 0.10. **Draws** 40,000 a cell, seed from the cell's name.
- **Arms.** `b1` and `b1s` as in the script at this commit.
- **Cell rule.** Over when the miss exceeds delta by more than 2 Monte Carlo SE.
- **Bands.** Rare: truth below 5%. Low-mid: 5% to 50%. High: above 50%.
- **Gain.** (Clopper-Pearson's mean excess on a random draw of the same size, exact / the arm's mean excess)^2.

## Predictions

- **S1.** Low-mid band: `b1` is over in at least 25% of cell-and-delta pairs.
- **S2.** Low-mid band: `b1s` is over in at most one third as many pairs as `b1`.
- **S3.** Low-mid band: the largest miss of `b1s` is at most 0.07 at delta 0.05 and at most 0.12 at 0.10.
- **S4.** High band: `b1s` is over in at most 2% of pairs, and its gain is within 1% of `b1`'s in every cell.
- **S5.** Low-mid band: the median ratio of `b1s`'s gain to `b1`'s is at least 0.85 at each delta.
- **S6.** Rare band (the 2% cells): `b1s` is over in at most 10% of pairs.

## What the result changes

- S2 and S3 kept: the paper recommends `b1s` where it now recommends `b1`, states its cost from S5, and says the
  gains in its tables are `b1`'s.
- S2 or S3 refuted: the paper reports the correction as partial, names the profiles where it fails, and keeps
  `b1` with the scope it has.
- Either way the paper states which profiles break `b1`, and that below 5% the exact bound is the rule.

## Not covered

Finite pools, the sampled-pool term, the training loop and real responses beyond the 116 development cells.
A lattice effect remains in any approximate limit; this run measures it and does not remove it.

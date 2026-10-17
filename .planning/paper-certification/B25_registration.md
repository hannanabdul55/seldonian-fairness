# B2/B5 registration: an exact stratified bound and the probability of certifying

Written 2026-10-16, before any outcome of `scripts/exact_strat.py` on a registered cell was computed. Its
`check` stage prints each cell's allocation (labels per stratum) only. This file and the script are committed
and pushed together before the run. Open items B2 and B5 of PLAN.md section 11 (blind review of v0.9.10,
M3 and M8; the user: "Do 3", that is, work on B2-B8).

## Why

- **B2.** The paper recommends the stratified Wald-t limit, which has no guarantee and is over its level in
  some cells. The reviewer asked for an exact stratified bound on the same cells, with its price against
  Clopper-Pearson on a random draw. The paper says only that exact stratified bounds "beat their own pooled
  form in one profile only".
- **B5.** The paper's gains are widths measured against the pooled Wilson bound, which is itself over its
  level in some of these cells. The probability of certifying is never reported.

## What is fixed

- **Cells.** All are drawn with replacement within strata, the limit of a pool far larger than the safety set:
  - `real`: every large-draw cell of `replacement_check.py` (013's C1-C3 at steps 0, 100 and 200; 014's steps
    100 and 200), of R5 (K1-K3), of R5b (K4) and of O6 (K5). n_s is 100 and 200; 58 cells.
  - `costly`: O3's ten XSTest human labels, with stratifiers A and B and n_s 50 and 100; 40 cells.
  - `grid`: 013's six synthetic profiles of `check_b1.py` at n_s 100, 200 and 400; 18 cells.

  Each cell keeps its own strata, weights and allocation rule. The stratified count in stratum h is
  Bin(n_h, theta_h), with theta_h the stratum's rate. The random draw is Bin(n_s, truth).
- **Deltas.** 0.05 and 0.10.
- **Arms.**
  - Exact:
    - `CP`, Clopper-Pearson on a random draw of n_s (the baseline).
    - `cp_strat`, Clopper-Pearson on the pooled count of the stratified draw, plus the allocation slack
      sum_h max(0, W_h - n_h / n).
    - `b2`, spike 013's union-intersection betting bound (`stratbounds.b2`, unchanged).
  - Approximate:
    - `b1` (Wald-t) and `b1w` on the stratified draw.
    - The pooled Wilson bound on a random draw.

  `CP`, pooled Wilson and `cp_strat` are computed exactly from their count distributions; `b1`, `b1w` and
  `b2` from 4,000 draws a cell, with seeds from the cell's name.
- **Measures.**
  - The miss.
  - The mean excess over the truth.
  - The width gain against CP and against pooled Wilson: (baseline excess / arm excess)^2.
  - The probability of certifying, for q in 0.5 and 0.8:
    - tau_q is the smallest threshold at which CP on n_s random labels certifies with probability at least q.
    - The arm's pass probability is P(limit <= tau_q).
    - Its **power ESS** is the number of labels CP on a random draw needs for that pass probability at tau_q,
      interpolated in log n on the running maximum of CP's pass probability over n. It is taken over the
      arm's own labels and divided by the same ratio for CP, so CP is 1.
- **Cell rule.**
  - An exact computation is over if its miss exceeds delta.
  - A drawn arm is over if its miss exceeds delta by more than 2 Monte Carlo SE.
- **Rate classes.** Rare is truth < 5%. "In range" is 5% <= truth <= 95%.

## Predictions

Scored by `exact_strat.py report` as committed.

- **X1.** `b2` and `cp_strat` are over in no cell, in any group, at either delta.
- **X2.** `b2`'s width gain against CP is below 1 in every cell at both deltas.
- **X3.** `b2`'s power ESS at q 0.8 and delta 0.05 is below 1 in every real and costly cell at a rate of 5% or more.
- **X4.** `cp_strat`'s power ESS at q 0.8 and delta 0.05, over the real and costly cells in range: the minimum
  is at least 0.95 and the median at least 1.10.
- **X5.** The same cells at q 0.5: `cp_strat`'s power ESS lies in [0.85, 1.15] in every one.
- **X6.** Rare real and costly cells at delta 0.05: `cp_strat`'s power ESS at q 0.8 lies in [0.8, 1.2].
- **X7.** Real cells in range at delta 0.05: `b1`'s width gain against CP is at least its gain against pooled
  Wilson in every one.
- **X8.** The same cells: `b1`'s power ESS at q 0.8 is above `cp_strat`'s in every one.

Reported without a prediction:
- The whole table by group, rate class, delta and arm.
- `b1w`'s power ESS.
- High-rate cells (> 95%).
- The slack per cell.
- `cp_strat`'s exact miss in every cell.

## The reasoning behind the numbers

- **X2, X3.** In 013's `check_bounds.md`, `b2`'s widths were 0.56-1.29 times its own pooled form and far
  wider than Clopper-Pearson pooled. At n 200 and p 0.3, the excess was 0.119 against 0.044.
- **X4, X5.** `cp_strat` does not adapt its width to the strata. Its limit is CP's function of a count that
  is less spread (Hoeffding, 1956), so it certifies more often only where the pass probability is above
  one half. A normal approximation with variance ratio 2 gives a power ESS near 1.3 at q 0.8 and near 1 at
  q 0.5.
- **X7.** CP is wider than Wilson at mid rates.

## Seen before writing this

- 013's `check_bounds.md` and `check_b1.md`.
- Every number in the paper.
- Two `b2` and `b1` values on made-up counts, used for timing (0.506 against 0.391; 0.936 against 0.888).
- One toy cell, an 8-stratum profile that is not a registered cell, run at 300 draws to test the code path.
- 3,000 random Bernoulli-sum configurations, on which CP's exact miss was at most 0.999 delta.
- The `check` stage's allocation sizes (n_h totals of 48, 96, 98, 100, 200 and 400).

No registered cell's miss, width or pass probability has been computed.

## Run

    OMP_NUM_THREADS=1 .venv/bin/python scripts/exact_strat.py run --workers 14
    .venv/bin/python scripts/exact_strat.py report

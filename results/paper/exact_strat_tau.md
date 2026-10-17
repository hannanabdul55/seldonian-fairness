# B5 sensitivity (not registered): the threshold spread over Clopper-Pearson's step

`scripts/exact_strat_tau.py`. Delta 0.05. Each arm's power ESS is averaged over 8 thresholds spread evenly over CP's step [U(k*), U(k*+1)) above the registered tau_q, where CP's own pass probability is constant. The registered value (threshold at the step's lower edge) is beside it. Written after the registered results were seen.

| group | rates | q | arm | cells | registered: median (min-max) | spread over the step: median (min-max) |
|---|---|---|---|---|---|---|
| real | mid (5-95%) | 0.5 | pooled Wilson | 36 | 1.00 (1.00-1.00) | 1.09 (1.05-1.46) |
| real | mid (5-95%) | 0.5 | cp_strat | 36 | 0.67 (0.07-1.09) | 0.74 (0.20-1.11) |
| real | mid (5-95%) | 0.5 | b1 | 36 | 1.83 (0.32-4.29) | 2.08 (0.48-4.79) |
| real | mid (5-95%) | 0.5 | b1w | 36 | 1.95 (1.03-5.42) | 2.16 (1.31-5.90) |
| real | mid (5-95%) | 0.8 | pooled Wilson | 36 | 1.00 (1.00-1.00) | 1.07 (1.03-1.36) |
| real | mid (5-95%) | 0.8 | cp_strat | 36 | 1.11 (0.27-1.83) | 1.20 (0.42-1.96) |
| real | mid (5-95%) | 0.8 | b1 | 36 | 1.88 (0.47-15.06) | 2.10 (0.47-15.23) |
| real | mid (5-95%) | 0.8 | b1w | 36 | 1.93 (1.00-15.06) | 2.13 (1.19-15.25) |
| real | rare (<5%) | 0.5 | pooled Wilson | 18 | 1.00 (1.00-1.00) | 1.12 (1.05-1.16) |
| real | rare (<5%) | 0.5 | cp_strat | 18 | 0.53 (0.39-1.00) | 0.76 (0.47-1.04) |
| real | rare (<5%) | 0.5 | b1 | 18 | 1.00 (0.48-1.05) | 1.15 (0.99-1.29) |
| real | rare (<5%) | 0.5 | b1w | 18 | 1.00 (0.77-1.05) | 1.16 (1.11-1.22) |
| real | rare (<5%) | 0.8 | pooled Wilson | 18 | 1.00 (1.00-1.00) | 1.07 (1.03-1.16) |
| real | rare (<5%) | 0.8 | cp_strat | 18 | 0.69 (0.33-1.01) | 0.82 (0.62-1.04) |
| real | rare (<5%) | 0.8 | b1 | 18 | 1.00 (0.48-1.09) | 1.18 (1.12-1.30) |
| real | rare (<5%) | 0.8 | b1w | 18 | 1.00 (1.00-1.06) | 1.16 (1.09-1.27) |
| costly | mid (5-95%) | 0.5 | pooled Wilson | 32 | 1.00 (1.00-1.00) | 1.12 (1.09-1.25) |
| costly | mid (5-95%) | 0.5 | cp_strat | 32 | 0.72 (0.55-0.82) | 0.86 (0.70-0.95) |
| costly | mid (5-95%) | 0.5 | b1 | 32 | 1.11 (0.80-1.35) | 1.29 (1.10-1.55) |
| costly | mid (5-95%) | 0.5 | b1w | 32 | 1.24 (1.06-1.49) | 1.44 (1.18-1.70) |
| costly | mid (5-95%) | 0.8 | pooled Wilson | 32 | 1.00 (1.00-1.00) | 1.09 (1.05-1.14) |
| costly | mid (5-95%) | 0.8 | cp_strat | 32 | 0.88 (0.68-1.01) | 0.95 (0.87-1.05) |
| costly | mid (5-95%) | 0.8 | b1 | 32 | 1.22 (0.96-1.39) | 1.25 (1.01-1.55) |
| costly | mid (5-95%) | 0.8 | b1w | 32 | 1.29 (1.13-1.60) | 1.41 (1.20-1.71) |
| costly | rare (<5%) | 0.5 | pooled Wilson | 8 | 1.00 (1.00-1.00) | 1.16 (1.15-1.27) |
| costly | rare (<5%) | 0.5 | cp_strat | 8 | 0.49 (0.00-0.50) | 0.78 (0.78-0.90) |
| costly | rare (<5%) | 0.5 | b1 | 8 | 0.49 (0.00-0.50) | 1.03 (0.65-1.10) |
| costly | rare (<5%) | 0.5 | b1w | 8 | 1.03 (1.02-1.65) | 1.14 (1.10-1.25) |
| costly | rare (<5%) | 0.8 | pooled Wilson | 8 | 1.00 (1.00-1.00) | 1.15 (1.09-1.16) |
| costly | rare (<5%) | 0.8 | cp_strat | 8 | 0.58 (0.49-0.70) | 0.88 (0.78-0.90) |
| costly | rare (<5%) | 0.8 | b1 | 8 | 0.58 (0.49-1.03) | 1.05 (0.99-1.15) |
| costly | rare (<5%) | 0.8 | b1w | 8 | 1.03 (1.02-1.38) | 1.15 (1.09-1.28) |
| grid | mid (5-95%) | 0.5 | pooled Wilson | 15 | 1.00 (1.00-1.00) | 1.07 (1.03-1.13) |
| grid | mid (5-95%) | 0.5 | cp_strat | 15 | 1.00 (1.00-1.03) | 1.00 (1.00-1.04) |
| grid | mid (5-95%) | 0.5 | b1 | 15 | 1.33 (0.89-1.73) | 1.43 (1.11-1.82) |
| grid | mid (5-95%) | 0.5 | b1w | 15 | 1.08 (0.89-1.65) | 1.19 (1.06-1.88) |
| grid | mid (5-95%) | 0.8 | pooled Wilson | 15 | 1.00 (1.00-1.00) | 1.05 (1.02-1.08) |
| grid | mid (5-95%) | 0.8 | cp_strat | 15 | 1.01 (1.00-1.25) | 1.02 (1.00-1.22) |
| grid | mid (5-95%) | 0.8 | b1 | 15 | 1.22 (1.00-1.65) | 1.29 (1.06-1.77) |
| grid | mid (5-95%) | 0.8 | b1w | 15 | 1.06 (1.00-1.69) | 1.15 (1.04-1.80) |
| grid | rare (<5%) | 0.5 | pooled Wilson | 3 | 1.00 (1.00-1.00) | 1.12 (1.06-1.13) |
| grid | rare (<5%) | 0.5 | cp_strat | 3 | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) |
| grid | rare (<5%) | 0.5 | b1 | 3 | 1.00 (1.00-1.23) | 1.40 (1.40-1.43) |
| grid | rare (<5%) | 0.5 | b1w | 3 | 1.00 (1.00-1.00) | 1.13 (1.11-1.15) |
| grid | rare (<5%) | 0.8 | pooled Wilson | 3 | 1.00 (1.00-1.00) | 1.07 (1.06-1.09) |
| grid | rare (<5%) | 0.8 | cp_strat | 3 | 1.00 (1.00-1.00) | 1.02 (1.00-1.02) |
| grid | rare (<5%) | 0.8 | b1 | 3 | 1.01 (1.00-1.26) | 1.27 (1.27-1.35) |
| grid | rare (<5%) | 0.8 | b1w | 3 | 1.00 (1.00-1.01) | 1.11 (1.07-1.16) |

## The registered statements re-read on the spread thresholds (not a test)

- cp_strat, real and costly cells in range (68), q 0.8: min 0.42, median 1.00 (X4 asked for >= 0.95 and >= 1.10).
- the same at q 0.5: 0.20-1.11 (X5 asked for [0.85, 1.15]).
- b1 above cp_strat at q 0.8 in 34 of 36 real cells in range (X8 asked for all).

# Pre-E5: the defensive-mixture bound on an exact toy

seed 0; 1000 draws per cell; delta 0.1 one-sided; 20 prompts over 400 responses, 40 of them harmful
exact breach rate 0.02066; per-prompt P(H | x) from 0.0050 to 0.0431

Not the registered E5 plasmode (which resamples the real brute-force pools). A check of the estimator and the two bounds before any GPU time; no registered number changes.

## Checks

- PASS T1 both weighted bounds hold their level in every cell: worst miss rate: empirical Bernstein 0.000, betting 0.038, against delta 0.1 over 1000 draws in 16 cells
- PASS T2 the estimator is unbiased in every cell, including a twin that drops modes: largest |mean - truth| is 2.24 standard errors (useless, alpha 0.5, n 1000: 0.02034 against 0.02066)
- PASS T3 validity survives an infinite forward KL (the narrow twin): the narrow twin drops all but 5 modes, forward KL infinite on 20 of 20 prompts; worst miss rate 0.011
- PASS T4 a good twin cuts the estimator's variance (section 6's variance claim): perfect twin: the summand's standard deviation is 0.122 to 0.188 of plain Monte Carlo's
- FAIL T5 that variance cut makes the bound at least half Clopper-Pearson's width at n = 300 (E5b's direction): perfect twin at n 300: the betting bound's width is 2.05 of Clopper-Pearson's, against the registered target of 0.5 or less
- PASS T6 a bad twin is no narrower than Clopper-Pearson (E5c's direction): useless and misaimed twins: betting width 1.48 to 5.36 of Clopper-Pearson's
- PASS T7 the weighted bound overtakes Clopper-Pearson somewhere in the scanned range: first crossing at alpha 0.5, n 3000 (width 0.91)

6 of 7 checks pass.

## Cells (the registered grid)

| twin | fwd KL | alpha | n | estimate | sd / plain sd | EB bound | EB miss | betting | BT miss | CP bound | CP miss | BT width / CP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| perfect | 0.0 | 0.2 | 300 | 0.02068 | 0.122 | 0.1400 | 0.000 | 0.0966 | 0.000 | 0.0354 | 0.052 | 5.14 |
| perfect | 0.0 | 0.2 | 1000 | 0.02068 | 0.122 | 0.0570 | 0.000 | 0.0437 | 0.000 | 0.0278 | 0.058 | 3.21 |
| perfect | 0.0 | 0.5 | 300 | 0.02072 | 0.188 | 0.0712 | 0.000 | 0.0510 | 0.000 | 0.0354 | 0.052 | 2.05 |
| perfect | 0.0 | 0.5 | 1000 | 0.02066 | 0.187 | 0.0367 | 0.000 | 0.0299 | 0.000 | 0.0278 | 0.058 | 1.28 |
| narrow | inf | 0.2 | 300 | 0.02013 | 1.688 | 0.1608 | 0.000 | 0.0985 | 0.000 | 0.0354 | 0.052 | 5.26 |
| narrow | inf | 0.2 | 1000 | 0.02053 | 1.712 | 0.0729 | 0.000 | 0.0461 | 0.000 | 0.0278 | 0.058 | 3.54 |
| narrow | inf | 0.5 | 300 | 0.02030 | 1.073 | 0.0863 | 0.000 | 0.0532 | 0.000 | 0.0354 | 0.052 | 2.20 |
| narrow | inf | 0.5 | 1000 | 0.02078 | 1.091 | 0.0466 | 0.000 | 0.0326 | 0.011 | 0.0278 | 0.058 | 1.67 |
| useless | 3.974 | 0.2 | 300 | 0.02072 | 1.002 | 0.1573 | 0.000 | 0.0973 | 0.000 | 0.0354 | 0.052 | 5.18 |
| useless | 3.974 | 0.2 | 1000 | 0.02045 | 0.992 | 0.0663 | 0.000 | 0.0441 | 0.000 | 0.0278 | 0.058 | 3.27 |
| useless | 3.974 | 0.5 | 300 | 0.02109 | 1.010 | 0.0877 | 0.000 | 0.0529 | 0.000 | 0.0354 | 0.052 | 2.18 |
| useless | 3.974 | 0.5 | 1000 | 0.02034 | 0.989 | 0.0452 | 0.000 | 0.0313 | 0.009 | 0.0278 | 0.058 | 1.48 |
| misaimed | inf | 0.2 | 300 | 0.01958 | 2.196 | 0.1721 | 0.000 | 0.0998 | 0.000 | 0.0354 | 0.052 | 5.36 |
| misaimed | inf | 0.2 | 1000 | 0.02084 | 2.257 | 0.0798 | 0.000 | 0.0484 | 0.000 | 0.0278 | 0.058 | 3.86 |
| misaimed | inf | 0.5 | 300 | 0.02033 | 1.411 | 0.0939 | 0.000 | 0.0546 | 0.000 | 0.0354 | 0.052 | 2.30 |
| misaimed | inf | 0.5 | 1000 | 0.02102 | 1.429 | 0.0506 | 0.000 | 0.0348 | 0.038 | 0.0278 | 0.058 | 1.98 |

## Crossover scan (perfect twin; n beyond the registered grid is exploratory)

| alpha | n | draws | sd / plain sd | betting | CP | BT width / CP | EB width / CP |
|---|---|---|---|---|---|---|---|
| 0.2 | 300 | 1000 | 0.122 | 0.0967 | 0.0354 | 5.14 | 8.08 |
| 0.2 | 1000 | 1000 | 0.121 | 0.0437 | 0.0278 | 3.21 | 5.07 |
| 0.2 | 3000 | 100 | 0.122 | 0.0284 | 0.0246 | 1.98 | 3.20 |
| 0.2 | 10000 | 100 | 0.122 | 0.0230 | 0.0225 | 1.25 | 2.13 |
| 0.5 | 300 | 1000 | 0.187 | 0.0509 | 0.0354 | 2.04 | 3.41 |
| 0.5 | 1000 | 1000 | 0.187 | 0.0299 | 0.0278 | 1.29 | 2.24 |
| 0.5 | 3000 | 100 | 0.188 | 0.0238 | 0.0241 | 0.91 | 1.70 |
| 0.5 | 10000 | 100 | 0.188 | 0.0217 | 0.0226 | 0.53 | 1.08 |

Runtime 178 s on the CPU.

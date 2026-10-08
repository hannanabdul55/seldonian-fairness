# Reference-rate strata redrawn with replacement

`scripts/replacement_check.py`. The cells are those of part A of `scripts/stratppi_baseline.py`: 8 equal rank strata of the reference's 8-sample rate, proportional allocation, a safety set of 100 or 200 prompts from a pool of about 500, by label and checkpoint. Three draws of every cell:

- *stored*: without replacement, 5,000 draws, the baseline's seeds. The safety set is 20-40% of the pool and the truth is the pool's rate.
- *paired*: each stratum's prompts drawn with replacement, the same seeds and 5,000 draws, so only the draw changes.
- *large*: with replacement, 40,000 draws from seeds of its own. These are the counts to quote.

With replacement is the limit of a pool far larger than the safety set, and StratPPI's N_h is taken as unbounded there (the predictor's stratum mean is known). Counts are cells over / unresolved / at or under delta (over: miss above delta + 2 se, se = sqrt(delta (1 - delta) / R); at delta 0.05 the band ends at 0.0562 for 5,000 draws and at 0.0522 for 40,000).

Replay check: the arms shared with `results/paper/stratppi.json` differ from it by at most 0.0e+00 in miss over 400 stored cells.

Control: on the random draw with replacement the labels are i.i.d. at the pool's rate, so the miss probability of the pooled Wilson bound and of Clopper-Pearson is a binomial sum. Over the 160 such cells of the large pass the simulated miss minus the exact one, in standard errors, has mean -0.05, standard deviation 0.94 and largest size 2.72 (Clopper-Pearson, C3:refusal, checkpoint 100, n_s 200, delta 0.05: 0.0408 against 0.0436). That cell redrawn twice more at 40,000 draws gave 0.0427 and 0.0441. Clopper-Pearson's exact miss probability is at or under delta in every one of these cells, as it must be (0.49 to 0.99 of delta on the mid-rate labels), so a cell of its above delta is Monte Carlo noise. The pooled Wilson arm is `b1w` at one stratum, whose grid can sit 0.00025 above the closed-form limit; the exact values use the grid's limits. Where the pool's rate falls inside that gap the closed form gives a different miss (C2:unsafe, checkpoint 100, n_s 200, delta 0.1: 0.119 against 0.076).

## Counts

| bound | labels | delta | cells | stored, 5,000 | paired, 5,000 | large, 40,000 | largest miss, large |
|---|---|---|---|---|---|---|---|
| b1w | mid-rate (9-95%) | 0.05 | 28 | 0 / 3 / 25 | 5 / 5 / 18 | 7 / 1 / 20 | 0.0624 |
| b1w | mid-rate (9-95%) | 0.1 | 28 | 0 / 0 / 28 | 2 / 5 / 21 | 6 / 2 / 20 | 0.1128 |
| b1w | rare (1-2%) | 0.05 | 12 | 0 / 0 / 12 | 0 / 1 / 11 | 1 / 0 / 11 | 0.0555 |
| b1w | rare (1-2%) | 0.1 | 12 | 1 / 0 / 11 | 3 / 0 / 9 | 3 / 0 / 9 | 0.1380 |
| Wald-t b1 | mid-rate (9-95%) | 0.05 | 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0376 |
| Wald-t b1 | mid-rate (9-95%) | 0.1 | 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0 / 0 / 28 | 0.0857 |
| Wald-t b1 | rare (1-2%) | 0.05 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0427 |
| Wald-t b1 | rare (1-2%) | 0.1 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0878 |
| StratPPI, normal limit | mid-rate (9-95%) | 0.05 | 28 | 9 / 5 / 14 | 18 / 4 / 6 | 20 / 1 / 7 | 0.0955 |
| StratPPI, normal limit | mid-rate (9-95%) | 0.1 | 28 | 7 / 3 / 18 | 14 / 6 / 8 | 18 / 1 / 9 | 0.1507 |
| StratPPI, normal limit | rare (1-2%) | 0.05 | 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 | 0.4747 |
| StratPPI, normal limit | rare (1-2%) | 0.1 | 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 | 0.4920 |
| StratPPI estimator, bootstrap-t | mid-rate (9-95%) | 0.05 | 28 | 0 / 0 / 28 | 0 / 2 / 26 | 0 / 0 / 28 | 0.0496 |
| StratPPI estimator, bootstrap-t | mid-rate (9-95%) | 0.1 | 28 | 0 / 0 / 28 | 0 / 6 / 22 | 0 / 1 / 27 | 0.1003 |
| StratPPI estimator, bootstrap-t | rare (1-2%) | 0.05 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0155 |
| StratPPI estimator, bootstrap-t | rare (1-2%) | 0.1 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0432 |
| pooled Wilson | mid-rate (9-95%) | 0.05 | 28 | 1 / 0 / 27 | 6 / 3 / 19 | 6 / 0 / 22 | 0.0689 |
| pooled Wilson | mid-rate (9-95%) | 0.1 | 28 | 1 / 0 / 27 | 5 / 9 / 14 | 10 / 3 / 15 | 0.1275 |
| pooled Wilson | rare (1-2%) | 0.05 | 12 | 0 / 0 / 12 | 1 / 0 / 11 | 1 / 0 / 11 | 0.0636 |
| pooled Wilson | rare (1-2%) | 0.1 | 12 | 1 / 1 / 10 | 3 / 1 / 8 | 3 / 0 / 9 | 0.1422 |
| Clopper-Pearson | mid-rate (9-95%) | 0.05 | 28 | 0 / 0 / 28 | 0 / 2 / 26 | 0 / 0 / 28 | 0.0497 |
| Clopper-Pearson | mid-rate (9-95%) | 0.1 | 28 | 0 / 0 / 28 | 0 / 1 / 27 | 0 / 0 / 28 | 0.0970 |
| Clopper-Pearson | rare (1-2%) | 0.05 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0471 |
| Clopper-Pearson | rare (1-2%) | 0.1 | 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 0.0945 |

`b1w` in the large pass, by label (cells over / unresolved / at or under; range of the miss):

| label | rates of its cells | cells | delta 0.05 | delta 0.10 |
|---|---|---|---|---|
| C1:refusal | 0.149-0.185 | 10 | 0 / 0 / 10; 0.0171-0.0336 | 0 / 0 / 10; 0.0692-0.0810 |
| C2:unsafe | 0.093-0.110 | 6 | 0 / 0 / 6; 0.0204-0.0337 | 0 / 0 / 6; 0.0563-0.0885 |
| C3:refusal | 0.654-0.673 | 6 | 2 / 1 / 3; 0.0474-0.0549 | 2 / 1 / 3; 0.0932-0.1058 |
| C2:refusal | 0.930-0.954 | 6 | 5 / 0 / 1; 0.0488-0.0624 | 4 / 1 / 1; 0.0727-0.1128 |
| C2:gated | 0.014-0.024 | 6 | 1 / 0 / 5; 0.0000-0.0555 | 1 / 0 / 5; 0.0000-0.1380 |
| C3:unsafe | 0.008-0.011 | 6 | 0 / 0 / 6; 0.0000-0.0000 | 2 / 0 / 4; 0.0000-0.1207 |

## Cells that are not at or under delta in the large pass, delta 0.05

| bound | label | checkpoint | n_s | truth | miss, stored | miss, paired | miss, large | class, large |
|---|---|---|---|---|---|---|---|---|
| b1w | C2:gated | 0 | 200 | 0.0138 | 0.0380 | 0.0546 | 0.0555 | over |
| b1w | C2:refusal | 0 | 100 | 0.9535 | 0.0524 | 0.0580 | 0.0554 | over |
| b1w | C2:refusal | 100 | 100 | 0.9335 | 0.0526 | 0.0574 | 0.0580 | over |
| b1w | C2:refusal | 100 | 200 | 0.9335 | 0.0440 | 0.0580 | 0.0599 | over |
| b1w | C2:refusal | 200 | 100 | 0.9305 | 0.0542 | 0.0654 | 0.0574 | over |
| b1w | C2:refusal | 200 | 200 | 0.9305 | 0.0450 | 0.0560 | 0.0624 | over |
| b1w | C3:refusal | 100 | 200 | 0.6603 | 0.0376 | 0.0568 | 0.0549 | over |
| b1w | C3:refusal | 200 | 100 | 0.6545 | 0.0474 | 0.0534 | 0.0549 | over |
| b1w | C3:refusal | 200 | 200 | 0.6545 | 0.0396 | 0.0482 | 0.0516 | unresolved |
| StratPPI, normal limit | all 40 cells | | | | | | 0.0293-0.4747 | 32 / 1 / 7 |
| pooled Wilson | C2:gated | 0 | 200 | 0.0138 | 0.0478 | 0.0646 | 0.0636 | over |
| pooled Wilson | C2:refusal | 100 | 100 | 0.9335 | 0.0566 | 0.0624 | 0.0689 | over |
| pooled Wilson | C2:refusal | 200 | 200 | 0.9305 | 0.0432 | 0.0700 | 0.0671 | over |
| pooled Wilson | C3:refusal | 100 | 100 | 0.6603 | 0.0420 | 0.0584 | 0.0601 | over |
| pooled Wilson | C3:refusal | 100 | 200 | 0.6603 | 0.0260 | 0.0580 | 0.0567 | over |
| pooled Wilson | C1:refusal (014) | 100 | 100 | 0.1490 | 0.0432 | 0.0580 | 0.0586 | over |
| pooled Wilson | C1:refusal (014) | 200 | 100 | 0.1847 | 0.0382 | 0.0582 | 0.0573 | over |

## Width and effective sample size, delta 0.05, last checkpoint

ESS = (excess of the pooled Wilson bound over the truth / the arm's excess)^2, each on its own kind of draw.

| label | n_s | truth | `b1w`, stored | `b1w`, large | Wald-t, stored | Wald-t, large | StratPPI bootstrap-t, stored | StratPPI bootstrap-t, large |
|---|---|---|---|---|---|---|---|---|
| C1:refusal | 100 | 0.165 | 2.39 | 2.29 | 2.13 | 2.04 | 2.58 | 2.17 |
| C1:refusal | 200 | 0.165 | 2.30 | 2.38 | 2.32 | 2.41 | 3.10 | 3.17 |
| C2:unsafe | 100 | 0.093 | 1.38 | 1.38 | 1.37 | 1.37 | 1.43 | 1.39 |
| C2:unsafe | 200 | 0.093 | 1.36 | 1.42 | 1.49 | 1.55 | 1.59 | 1.66 |
| C2:refusal | 100 | 0.930 | 0.96 | 0.95 | 0.43 | 0.43 | 0.83 | 0.80 |
| C2:refusal | 200 | 0.930 | 1.12 | 1.11 | 0.65 | 0.65 | 1.03 | 1.00 |
| C3:refusal | 100 | 0.654 | 4.74 | 4.73 | 2.62 | 2.61 | 1.77 | 1.55 |
| C3:refusal | 200 | 0.654 | 5.19 | 5.14 | 3.66 | 3.63 | 5.35 | 5.47 |
| C1:refusal (014) | 100 | 0.185 | 2.19 | 2.15 | 2.00 | 1.95 | 2.43 | 2.25 |
| C1:refusal (014) | 200 | 0.185 | 2.13 | 2.16 | 2.17 | 2.20 | 2.49 | 2.47 |
| median of the ten | | | 2.16 | 2.16 | 2.06 | 1.99 | 2.10 | 1.92 |

# Validity recount under one rule

`scripts/validity_recount.py`; source: `reports/paper_certification.md` (draft v0.5) and the result files it cites. Nothing was simulated: 2,022 cells and 7,444,000 draws are counted from existing files, or read from the training paper's printed tables where the data are lost (marked `~`).

**The rule.** A cell is one resampling study: R draws at level delta, miss m. With se = sqrt(delta (1 - delta) / R): *over* if m > delta + 2 se; *unresolved, above delta* if delta < m <= delta + 2 se; *at or under delta* if m <= delta. The exact one-sided binomial p-value of H0 'true miss <= delta' and a 95% Clopper-Pearson interval are in the JSON for every cell, and below for the cells that matter. *Over after Bonferroni*: the 2 se replaced by z(1 - a / C) se for the C cells of the row, a = 1 - Phi(2) = 0.0228, so one cell gives the rule itself.

**What a cell is.** The unit a source file stores: one label, sample size, checkpoint, feature or pipeline, with its own draws. The paper counts some rows differently (the largest miss of 2-3 checkpoints as one cell); both counts are given where they differ.

**Three cautions.** (1) Cells of one row are not independent: the two deltas of spikes 013 and 014 use the same draws, the four judge features of spike 017 share a cell's draws, and arms are paired. Bonferroni is then conservative, and the pooled miss is a description, not a test. (2) The 4,000 draws of a sheet cell are 20 plantings x 200 sheets; the per-planting counts were not stored, so the binomial treats them as 4,000. (3) *At or under delta* is a statement about the estimate; with 200-500 draws its interval still reaches well above delta (see the largest-miss column).

## (a) Per-bound summary

Rows the paper quotes, in the order of Table 2 and then by section.

| bound | setting | delta | source | where in the paper | the paper says | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.1 | [R 6.2] | Table 2 row 1 | holds | 5 | 5,000 | 25,000 | ~0.0784 (at or under delta) | 0.0840 [0.0765, 0.0920] | 0 | 0 | 5 | 0 |
| Clopper-Pearson, labels alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 2 row 2 | holds | 42 | 1,000, 4,000 | 156,000 | 0.0381 (at or under delta) | 0.0515 [0.0449, 0.0588] | 0 | 2 | 40 | 0 |
| betting mixture | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.1 | [R 6.2] | Table 2 row 3 | holds | 5 | 5,000 | 25,000 | ~0.0106 (at or under delta) | 0.0160 [0.0127, 0.0199] | 0 | 0 | 5 | 0 |
| Bentkus | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.1 | [R 6.2] | Table 2 row 4 | holds | 5 | 5,000 | 25,000 | ~0.0306 (at or under delta) | 0.0350 [0.0301, 0.0405] | 0 | 0 | 5 | 0 |
| Hoeffding, Anderson | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.1 | [R 6.2] | Table 2 row 5 | holds | 5 | 5,000 | 25,000 | ~0 (at or under delta) | 0 [0, 0.0007] | 0 | 0 | 5 | 0 |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | 0.05 | [013 H8] | Table 2 row 6 | holds | 24 | 5,000 | 120,000 | 0.0336 (at or under delta) | 0.0542 [0.0481, 0.0608] | 0 | 3 | 21 | 0 |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | 0.1 | [013 H8] | Table 2 row 6 | holds | 24 | 5,000 | 120,000 | 0.0776 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 24 | 0 |
| `b1w` | label pushed by the Lagrangian, n_s 200, steps 100 and 200 | 0.05 | [014] | Table 2 row 7 | holds | 2 | 5,000 | 10,000 | 0.0169 (at or under delta) | 0.0228 [0.0188, 0.0273] | 0 | 0 | 2 | 0 |
| `b1w` | label pushed by the Lagrangian, n_s 200, steps 100 and 200 | 0.1 | [014] | Table 2 row 7 | holds | 2 | 5,000 | 10,000 | 0.0520 (at or under delta) | 0.0642 [0.0576, 0.0714] | 0 | 0 | 2 | 0 |
| `b1w` on a design-weighted sheet | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | Table 2 row 8 | holds | 18 | 4,000 | 72,000 | 0.0246 (at or under delta) | 0.0595 [0.0524, 0.0673] | 1 | 1 | 16 | 0 |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 2 row 9 | holds | 168 | 1,000, 4,000 | 624,000 | 0.0400 (at or under delta) | 0.0530 [0.0399, 0.0688] | 0 | 12 | 156 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | [020 P] | Table 2 row 10 | holds | 6 | 400 | 2,400 | 0.0400 (at or under delta) | 0.0600 [0.0388, 0.0880] | 0 | 2 | 4 | 0 |
| StratPPI estimator with a bootstrap-t limit | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Table 2 row 11; Table 3 | holds | 28 | 5,000 | 140,000 | 0.0345 (at or under delta) | 0.0452 [0.0396, 0.0513] | 0 | 0 | 28 | 0 |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | Table 2 row 11; section 6 | holds | 28 | 4,000 | 112,000 | 0.0397 (at or under delta) | 0.0560 [0.0491, 0.0636] | 0 | 5 | 23 | 0 |
| Student-t | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.1 | [R 6.2] | Table 2 row 12 | fails | 5 | 5,000 | 25,000 | ~0.1244 (over) | 0.1840 [0.1733, 0.1950] | 3 | 2 | 0 | 3 |
| `b1w` at rare rates | labels at 1-2%, n_s 100, 3 checkpoints | 0.05 | [013 H8] | Table 2 row 13 | fails | 6 | 5,000 | 30,000 | 0.2606 (over) | 0.4432 [0.4294, 0.4571] | 6 | 0 | 0 | 6 |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100, 3 checkpoints | 0.05 | [013 H8] | Table 2 row 13 | fails | 6 | 5,000 | 30,000 | 0.2612 (over) | 0.4476 [0.4338, 0.4615] | 6 | 0 | 0 | 6 |
| PPI++ with a normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 2 row 14 | fails | 168 | 1,000, 4,000 | 624,000 | 0.0791 (over) | 0.2412 [0.2281, 0.2548] | 119 | 32 | 17 | 82 |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Table 2 row 15; Table 3 | fails | 28 | 5,000 | 140,000 | 0.0498 (at or under delta) | 0.0862 [0.0786, 0.0943] | 9 | 5 | 14 | 7 |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | Table 2 row 15; section 6 | fails | 28 | 4,000 | 112,000 | 0.0982 (over) | 0.2387 [0.2256, 0.2523] | 26 | 2 | 0 | 25 |
| Clopper-Pearson over pairs | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | [020 P] | Table 2 row 16 | fails | 6 | 400 | 2,400 | 0.1500 (over) | 0.2750 [0.2318, 0.3216] | 5 | 0 | 1 | 5 |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | [020 P] | Table 2 row 17 | fails | 6 | 400 | 2,400 | 0.0525 (unresolved) | 0.1700 [0.1345, 0.2105] | 2 | 0 | 4 | 2 |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | Table 2 row 18 | fails | 18 | 4,000 | 72,000 | 0.2364 (over) | 0.9892 [0.9855, 0.9922] | 7 | 0 | 11 | 7 |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 2 row 19 | fails | 42 | 1,000, 4,000 | 156,000 | 0.6457 (over) | 1 [0.9991, 1] | 30 | 0 | 12 | 30 |
| `b1w` at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | 0.05 | [013 H8] | sections 3 and 5 (rare labels) | fails | 6 | 5,000 | 30,000 | 0.0728 (over) | 0.1744 [0.1640, 0.1852] | 3 | 0 | 3 | 3 |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | 0.05 | [013 H8] | sections 3 and 5 (rare labels) | fails | 6 | 5,000 | 30,000 | 0.0785 (over) | 0.1784 [0.1679, 0.1893] | 3 | 0 | 3 | 3 |
| StratPPI estimator with a bootstrap-t limit | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | section 3 (reading 1) | holds | 12 | 5,000 | 60,000 | 0.0023 (at or under delta) | 0.0100 [0.0074, 0.0132] | 0 | 0 | 12 | 0 |
| StratPPI estimator with a bootstrap-t limit | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 3 (reading 1) | holds | 12 | 5,000 | 60,000 | 0.0084 (at or under delta) | 0.0176 [0.0141, 0.0216] | 0 | 0 | 12 | 0 |
| stratified Wald-t `b1` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | section 3 (reading 1) | holds | 12 | 5,000 | 60,000 | 0.0033 (at or under delta) | 0.0272 [0.0229, 0.0321] | 0 | 0 | 12 | 0 |
| stratified Wald-t `b1` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 3 (reading 1) | holds | 12 | 5,000 | 60,000 | 0.0110 (at or under delta) | 0.0666 [0.0598, 0.0739] | 0 | 0 | 12 | 0 |
| Seldonian pipeline (safety test after selection) | synthetic bandit, tables (a) and (b): four bounds, n 200-5,000 | 0.1 | [R 6.3] | section 4 | holds | 7 | not stated | | | 0.0020 | | | | |
| Seldonian pipeline (safety test after selection) | synthetic bandit, table (c): pressures 0-4, n 1,000 | 0.1 | [R 6.3] | section 4 | holds | 5 | 500 | 2,500 | ~0.0004 (at or under delta) | 0.0020 [0.0001, 0.0111] | 0 | 0 | 5 | 0 |
| unconstrained training (no test) | synthetic bandit, table (c): pressures 0-4 | 0.1 | [R 6.3] | section 4 | fails | 5 | 500 | 2,500 | ~0.7068 (over) | 1 [0.9926, 1] | 5 | 0 | 0 | 5 |
| Seldonian pipeline, judge-level violation given a solution | synthetic bandit, table (d): four judges | 0.1 | [R 6.3d] | section 4 | holds | 4 | not stated | | | 0.0030 | | | | |
| Clopper-Pearson safety test after an adversarial split | classic setup, four settings, 11 split rules; miss = passed and truly violating | 0.05 | [012] | section 4 | holds | 42 | 2,000 | 84,000 | 0.0000 (at or under delta) | 0.0005 [0.0000, 0.0028] | 0 | 0 | 42 | 0 |
| pooled Wilson bound, random split | same labels and sizes | 0.05 | [013 H8] | section 4 (the comparator of the in-loop sentence); Table 3 baseline | - | 24 | 5,000 | 120,000 | 0.0307 (at or under delta) | 0.0566 [0.0504, 0.0634] | 1 | 0 | 23 | 0 |
| pooled Wilson bound, random split | same labels and sizes | 0.1 | [013 H8] | section 4 (the comparator of the in-loop sentence); Table 3 baseline | - | 24 | 5,000 | 120,000 | 0.0721 (at or under delta) | 0.1140 [0.1053, 0.1231] | 1 | 0 | 23 | 1 |
| `b1w` in the training loop, reference-rate strata | synthetic bandit, 4 heterogeneity levels, `SeldonianLLMPolicy` | 0.1 | [013 4] | section 4 | holds | 4 | 500 | 2,000 | 0.0960 (at or under delta) | 0.1160 [0.0893, 0.1474] | 0 | 1 | 3 | 0 |
| pooled `b1w` in the training loop, random split | same | 0.1 | [013 4] | section 4 (comparator) | holds | 4 | 500 | 2,000 | 0.0940 (at or under delta) | 0.1000 [0.0751, 0.1297] | 0 | 0 | 4 | 0 |
| trajectory certificate at delta / T | synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under its true rate | 0.1 | [SR 2.1] | section 4 | holds | 10 | 200 | 2,000 | 0.0495 (at or under delta) | 0.1000 [0.0622, 0.1502] | 0 | 0 | 10 | 0 |
| StratPPI estimator with a bootstrap-t limit | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 5 | holds | 28 | 5,000 | 140,000 | 0.0796 (at or under delta) | 0.0944 [0.0864, 0.1028] | 0 | 0 | 28 | 0 |
| StratPPI as published (normal limit) | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | section 5 | fails | 12 | 5,000 | 60,000 | 0.2054 (over) | 0.4454 [0.4316, 0.4593] | 12 | 0 | 0 | 12 |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 5 | fails | 28 | 5,000 | 140,000 | 0.0947 (at or under delta) | 0.1438 [0.1342, 0.1538] | 7 | 3 | 18 | 5 |
| StratPPI as published (normal limit) | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 5 | fails | 12 | 5,000 | 60,000 | 0.2401 (over) | 0.4542 [0.4403, 0.4681] | 12 | 0 | 0 | 12 |
| `b1w` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | Table 3 | holds | 28 | 5,000 | 140,000 | 0.0315 (at or under delta) | 0.0542 [0.0481, 0.0608] | 0 | 3 | 25 | 0 |
| `b1w` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | section 5 | fails | 12 | 5,000 | 60,000 | 0.1667 (over) | 0.4432 [0.4294, 0.4571] | 9 | 0 | 3 | 9 |
| `b1w` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | section 5 | holds | 28 | 5,000 | 140,000 | 0.0749 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 28 | 0 |
| `b1w` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | section 5 | fails | 12 | 5,000 | 60,000 | 0.1712 (over) | 0.4432 [0.4294, 0.4571] | 6 | 0 | 6 | 6 |
| stratified Wald-t `b1` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | section 5 | holds | 28 | 5,000 | 140,000 | 0.0179 (at or under delta) | 0.0364 [0.0314, 0.0420] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 5 | holds | 28 | 5,000 | 140,000 | 0.0501 (at or under delta) | 0.0804 [0.0730, 0.0883] | 0 | 0 | 28 | 0 |
| Clopper-Pearson, labels alone | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 6 (the baseline) | holds | 14 | 4,000 | 56,000 | 0.0318 (at or under delta) | 0.0508 [0.0442, 0.0580] | 0 | 1 | 13 | 0 |
| PPI++ with a normal limit | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 6 | fails | 14 | 4,000 | 56,000 | 0.1023 (over) | 0.2320 [0.2190, 0.2454] | 14 | 0 | 0 | 14 |
| PPI++ with a bootstrap-t limit | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 6 | holds | 14 | 4,000 | 56,000 | 0.0354 (at or under delta) | 0.0517 [0.0451, 0.0591] | 0 | 1 | 13 | 0 |
| `b1w` on judge-logit strata | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | section 6 | holds | 28 | 4,000 | 112,000 | 0.0391 (at or under delta) | 0.0532 [0.0465, 0.0607] | 0 | 1 | 27 | 0 |
| carried Youden-corrected bound | source: xstest -> orbench, six judge wordings | 0.05 | [017 5] | section 7.1 | over for 4 of 6 | 6 | 4,000 | 24,000 | 0.4020 (over) | 0.9435 [0.9359, 0.9505] | 4 | 0 | 2 | 4 |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | 0.05 | [017 5] | section 7.1 | over for 1 of 6 | 6 | 4,000 | 24,000 | 0.0222 (at or under delta) | 0.1328 [0.1224, 0.1437] | 1 | 0 | 5 | 1 |
| carried Youden-corrected bound | pool: over-refusal -> harmful, six judge wordings | 0.05 | [017 5] | section 7.1 | over for none | 6 | 4,000 | 24,000 | 0.0000 (at or under delta) | 0.0003 [0.0000, 0.0014] | 0 | 0 | 6 | 0 |
| carried Youden-corrected bound | pool: harmful -> over-refusal, six judge wordings | 0.05 | [017 5] | section 7.1 | over for 5 of 6 | 6 | 4,000 | 24,000 | 0.4275 (over) | 0.9960 [0.9935, 0.9977] | 5 | 0 | 1 | 5 |
| carried Youden-corrected bound | training: step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 7.2 | carried (holds) | 6 | 4,000 | 24,000 | 0.0103 (at or under delta) | 0.0415 [0.0355, 0.0481] | 0 | 0 | 6 | 0 |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | 0.05 | [017 E8] | section 7.2 | 1 of 6 failed | 6 | 4,000 | 24,000 | 0.1047 (over) | 0.5135 [0.4979, 0.5291] | 1 | 0 | 5 | 1 |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 7.2 | over for 3 of 6, marginal for a fourth | 6 | 4,000 | 24,000 | 0.1968 (over) | 0.8027 [0.7901, 0.8150] | 3 | 1 | 2 | 3 |
| block PPI, betting (finite-sample) | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | section 7.4 | holds | 84 | 500, 1,000 | 80,000 | 0.0174 (at or under delta) | 0.0440 [0.0321, 0.0586] | 0 | 0 | 84 | 0 |
| cluster bootstrap-t, by injection task | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | [020 P] | section 8.2 (half of the larger-of-two rule) | - | 6 | 400 | 2,400 | 0.0242 (at or under delta) | 0.1225 [0.0920, 0.1587] | 1 | 0 | 5 | 1 |
| the four limits of the human-terms certificate | design check on synthetic labels, prompts as 'pool' | 0.05 | [P9 design check] | section 8.4 | holds | 4 | 2,000 | 8,000 | 0.0319 (at or under delta) | 0.0455 [0.0368, 0.0556] | 0 | 0 | 4 | 0 |

Bounds in the same files that the paper does not quote.

| bound | setting | delta | source | where in the paper | the paper says | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tight Wald test after an adversarial split | same runs; miss = true gap above the safety-set bound | 0.05 | [012] | not quoted in the paper | - | 42 | 2,000 | 84,000 | 0.0288 (at or under delta) | 0.0550 [0.0454, 0.0659] | 0 | 3 | 39 | 0 |
| `ttest` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 16 | 500 | 8,000 | 0.1045 (unresolved) | 0.1360 [0.1072, 0.1692] | 2 | 6 | 8 | 0 |
| `b1w_pooled` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.0942 (at or under delta) | 0.1160 [0.0893, 0.1474] | 0 | 6 | 6 | 0 |
| `b1w_strat_pool` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.0733 (at or under delta) | 0.1060 [0.0804, 0.1364] | 0 | 1 | 11 | 0 |
| `b1_strat_pop` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.1095 (over) | 0.1320 [0.1036, 0.1649] | 2 | 4 | 6 | 0 |
| `b1w_strat_pop` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 8 | 500 | 4,000 | 0.1030 (unresolved) | 0.1140 [0.0875, 0.1452] | 0 | 5 | 3 | 0 |
| per-check delta read as a trajectory claim | synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under its true rate | 0.1 | [SR 2.1] | not quoted in the paper | - | 10 | 200 | 2,000 | 0.5025 (over) | 0.6500 [0.5795, 0.7159] | 10 | 0 | 0 | 10 |
| plain PPI, normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 168 | 1,000, 4,000 | 624,000 | 0.0603 (over) | 0.1420 [0.1313, 0.1532] | 69 | 60 | 39 | 40 |
| PPI++, score (Wilson-type) limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 168 | 1,000, 4,000 | 624,000 | 0.0661 (over) | 0.2208 [0.2080, 0.2339] | 71 | 31 | 66 | 51 |
| Youden correction from the same labels | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 1,000, 4,000 | 156,000 | 0.0003 (at or under delta) | 0.0080 [0.0035, 0.0157] | 0 | 0 | 42 | 0 |
| PPI, three exact limits | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 1,000, 4,000 | 156,000 | 0.0020 (at or under delta) | 0.0150 [0.0084, 0.0246] | 0 | 0 | 42 | 0 |
| post-stratified on the 0/1 judge, exact | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 1,000, 4,000 | 156,000 | 0.0027 (at or under delta) | 0.0160 [0.0092, 0.0259] | 0 | 0 | 42 | 0 |
| sheet labels read as i.i.d., Clopper-Pearson | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.0014 (at or under delta) | 0.0083 [0.0057, 0.0116] | 0 | 0 | 18 | 0 |
| design-weighted labels, normal limit | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.1323 (over) | 0.2160 [0.2033, 0.2291] | 18 | 0 | 0 | 18 |
| design-weighted PPI | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.0543 (over) | 0.0840 [0.0756, 0.0930] | 10 | 2 | 6 | 7 |
| design-weighted PPI++ | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.1324 (over) | 0.2160 [0.2033, 0.2291] | 18 | 0 | 0 | 18 |
| `b1w` on a design-weighted sheet, the two runs of each setting pooled | same sheets, 3 wordings x 3 planted rates | 0.05 | [017 4] | not quoted in the paper | - | 9 | 8,000 | 72,000 | 0.0246 (at or under delta) | 0.0540 [0.0491, 0.0592] | 0 | 1 | 8 | 0 |
| Wilson bound over pairs | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | [020 P] | not quoted in the paper | - | 6 | 400 | 2,400 | 0.1512 (over) | 0.2750 [0.2318, 0.3216] | 5 | 0 | 1 | 5 |
| pooled Wilson bound, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0305 (at or under delta) | 0.0566 [0.0504, 0.0634] | 1 | 0 | 27 | 0 |
| pooled Wilson bound, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.1699 (over) | 0.4476 [0.4338, 0.4615] | 9 | 0 | 3 | 9 |
| pooled Wilson bound, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0712 (at or under delta) | 0.1140 [0.1053, 0.1231] | 1 | 0 | 27 | 1 |
| pooled Wilson bound, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.1754 (over) | 0.4476 [0.4338, 0.4615] | 6 | 1 | 5 | 6 |
| PPI++ with a normal limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0296 (at or under delta) | 0.0922 [0.0843, 0.1006] | 4 | 2 | 22 | 3 |
| PPI++ with a normal limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.2189 (over) | 0.4606 [0.4467, 0.4745] | 12 | 0 | 0 | 12 |
| PPI++ with a normal limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0633 (at or under delta) | 0.1524 [0.1425, 0.1627] | 3 | 3 | 22 | 3 |
| PPI++ with a normal limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.2672 (over) | 0.4782 [0.4643, 0.4922] | 12 | 0 | 0 | 12 |
| PPI++ with a bootstrap-t limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0161 (at or under delta) | 0.0346 [0.0297, 0.0400] | 0 | 0 | 28 | 0 |
| PPI++ with a bootstrap-t limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0023 (at or under delta) | 0.0098 [0.0073, 0.0129] | 0 | 0 | 12 | 0 |
| PPI++ with a bootstrap-t limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0470 (at or under delta) | 0.0842 [0.0766, 0.0922] | 0 | 0 | 28 | 0 |
| PPI++ with a bootstrap-t limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0064 (at or under delta) | 0.0172 [0.0138, 0.0212] | 0 | 0 | 12 | 0 |
| the four limits of the human-terms certificate | design check on synthetic labels, prompts as 'new prompts' | 0.05 | [P9 design check] | not quoted in the paper | - | 4 | 2,000 | 8,000 | 0.0418 (at or under delta) | 0.0630 [0.0527, 0.0746] | 1 | 2 | 1 | 1 |

The same bound over all its settings at one delta (cells that repeat another row's draws left out).

| bound | delta | rows pooled | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson | 0.1 | 1 | 5 | 5,000 | 25,000 | ~0.0784 (at or under delta) | 0.0840 [0.0765, 0.0920] | 0 | 0 | 5 | 0 |
| Clopper-Pearson | 0.05 | 2 | 56 | 1,000, 4,000 | 212,000 | 0.0364 (at or under delta) | 0.0515 [0.0449, 0.0588] | 0 | 3 | 53 | 0 |
| b1w (Table 2 rows 6-8) | 0.05 | 3 | 44 | 4,000, 5,000 | 202,000 | 0.0296 (at or under delta) | 0.0595 [0.0524, 0.0673] | 1 | 4 | 39 | 0 |
| b1w (Table 2 rows 6-8) | 0.1 | 2 | 26 | 5,000 | 130,000 | 0.0756 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 26 | 0 |
| PPI++, bootstrap-t limit | 0.05 | 3 | 210 | 1,000, 4,000, 5,000 | 820,000 | 0.0356 (at or under delta) | 0.0530 [0.0399, 0.0688] | 0 | 13 | 197 | 0 |
| StratPPI estimator, bootstrap-t limit | 0.05 | 2 | 56 | 4,000, 5,000 | 252,000 | 0.0368 (at or under delta) | 0.0560 [0.0491, 0.0636] | 0 | 5 | 51 | 0 |
| PPI++, normal limit | 0.05 | 3 | 210 | 1,000, 4,000, 5,000 | 820,000 | 0.0723 (over) | 0.2412 [0.2281, 0.2548] | 137 | 34 | 39 | 98 |
| StratPPI, normal limit | 0.05 | 2 | 56 | 4,000, 5,000 | 252,000 | 0.0713 (over) | 0.2387 [0.2256, 0.2523] | 35 | 7 | 14 | 30 |
| StratPPI estimator, bootstrap-t limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0796 (at or under delta) | 0.0944 [0.0864, 0.1028] | 0 | 0 | 28 | 0 |
| StratPPI, normal limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0947 (at or under delta) | 0.1438 [0.1342, 0.1538] | 7 | 3 | 18 | 5 |
| PPI++, normal limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0633 (at or under delta) | 0.1524 [0.1425, 0.1627] | 3 | 3 | 22 | 3 |
| PPI++, bootstrap-t limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0470 (at or under delta) | 0.0842 [0.0766, 0.0922] | 0 | 0 | 28 | 0 |

Bonferroni by exact p-values (p < a / C) in place of the widened band gives the same count in every row except: plain PPI, normal limit, spike 017 plasmodes (38 against 40); Clopper-Pearson over pairs, AgentDojo (4 against 5); Wilson bound over pairs, AgentDojo (4 against 5); StratPPI as published (normal limit), 5 mid-rate labels (6 against 7); StratPPI as published (normal limit), judge-logit strata (5 and 10) on the refusal pools (24 against 25).

## (b) Table 2, recounted

*Table 2. Miss rate against delta for every bound used, one rule for every row. A cell is over its level when its miss exceeds delta by more than two Monte Carlo standard errors, unresolved when it is above delta inside that band, and at or under delta otherwise. `~`: printed rates, not regenerable. Bold and the `kind` column are the paper's and are not derived from the counts.*

| bound | kind | setting | delta | miss (paper) | miss (recount, per cell) | cells | draws per cell | pooled miss | over | unresolved, above delta | at or under delta | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson | exact (binary labels) | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | 0.10 | 0.067-0.084 | 0.0670-0.0840 | 5 | 5,000 | ~0.0784 | 0 | 0 | 5 | [R 6.2] |
| Clopper-Pearson | exact | spike 017 plasmodes, every cell | 0.05 | at most 0.051 | 0-0.0515 | 42 | 1,000, 4,000 | 0.0381 | 0 | 2 | 40 | [017 B6] |
| betting mixture | exact (bounded) | same pool as row 1 | 0.10 | 0.007-0.016 | 0.0070-0.0160 | 5 | 5,000 | ~0.0106 | 0 | 0 | 5 | [R 6.2] |
| Bentkus | exact (bounded) | same pool | 0.10 | 0.023-0.035 | 0.0230-0.0350 | 5 | 5,000 | ~0.0306 | 0 | 0 | 5 | [R 6.2] |
| Hoeffding, Anderson | exact (bounded) | same pool | 0.10 | 0.000 | 0 | 5 | 5,000 | ~0 | 0 | 0 | 5 | [R 6.2] |
| stratified Wilson-type `b1w` | approximate | 4 mid-rate labels (9-94%), real Granite-3.3-2B responses, n_s 100-200, 3 checkpoints [c] | 0.05 / 0.10 | 0.023-0.054 / 0.069-0.097 | 0.0166-0.0542 / 0.0576-0.0970 | 24 / 24 | 5,000 / 5,000 | 0.0336 / 0.0776 | 0 / 0 | 3 / 0 | 21 / 24 | [013 H8] |
| `b1w` | approximate | label pushed by the Lagrangian, n_s 200 | 0.05 / 0.10 | 0.011-0.023 / 0.040-0.064 | 0.0110-0.0228 / 0.0398-0.0642 | 2 / 2 | 5,000 / 5,000 | 0.0169 / 0.0520 | 0 / 0 | 0 / 0 | 2 / 2 | [014] |
| `b1w` on a design-weighted sheet | approximate | sheets re-drawn by their real sampling rule | 0.05 | 0.001-0.059 | 0.0010-0.0595 | 18 | 4,000 | 0.0246 | 1 | 1 | 16 | [017 4] |
| PPI++ with a bootstrap-t limit | approximate (second order) | spike 017 plasmodes, every cell and feature | 0.05 | at most 0.053 | 0-0.0530 | 168 | 1,000, 4,000 | 0.0400 | 0 | 12 | 156 | [017 B6] |
| cluster bootstrap-t, by user task | approximate | AgentDojo, 6 pipelines, user tasks resampled | 0.05 | 0.020-0.060 | 0.0200-0.0600 | 6 | 400 | 0.0400 | 0 | 2 | 4 | [020 P] |
| StratPPI estimator with a bootstrap-t limit | approximate | reference-rate strata, 5 mid-rate labels, n_s 100-200, 2-3 checkpoints [c]; judge-logit strata, 14 cells x 2 strata counts, n 100-1,000 | 0.05 | at most 0.045; at most 0.056 | 0.0236-0.0452; 0.0008-0.0560 | 28; 28 | 5,000; 4,000 | 0.0345; 0.0397 | 0; 0 | 0; 5 | 28; 23 | [P14] |
| **Student-t** | fails at low rates | same pool as row 1, n 200-2,400 (the paper's row: n 200-800) [a] | 0.10 | **0.115-0.184** | 0.1010-0.1840 | 5 | 5,000 | ~0.1244 | 3 | 2 | 0 | [R 6.2] |
| **`b1w` and the pooled Wilson bound at rare rates** | fail | harm labels at 1-2%, n_s 100, any design, 3 checkpoints [c] | 0.05 | **0.24-0.44** | 0.0758-0.4476 | 12 | 5,000 | 0.2609 | 12 | 0 | 0 | [013 H8] |
| **PPI++ with a normal limit** | fails | spike 017 plasmodes | 0.05 | **up to 0.241** | 0.0408-0.2412 | 168 | 1,000, 4,000 | 0.0791 | 119 | 32 | 17 | [017 B6] |
| **StratPPI as published (normal limit)** | fails at these sizes | reference-rate strata, 5 mid-rate labels, 2-3 checkpoints [c]; judge-logit strata | 0.05 | **up to 0.086; up to 0.239** | 0.0238-0.0862; 0.0535-0.2387 | 28; 28 | 5,000; 4,000 | 0.0498; 0.0982 | 9; 26 | 5; 2 | 14; 0 | [P14] |
| **Clopper-Pearson over pairs** | fails on a crossed design | AgentDojo, user tasks resampled | 0.05 | **0.048-0.275** | 0.0475-0.2750 | 6 | 400 | 0.1500 | 5 | 0 | 1 | [020 P] |
| **two-way bootstrap** | fails where positives sit in few clusters | AgentDojo | 0.05 | **up to 0.170** | 0-0.1700 | 6 | 400 | 0.0525 | 2 | 0 | 4 | [020 P] |
| **stratified sheet read as an i.i.d. sample** | fails | PPI on the sheet, three judge wordings (the paper's row: one) [b] | 0.05 | **up to 0.98** | 0-0.9892 | 18 | 4,000 | 0.2364 | 7 | 0 | 11 | [017 4] |
| **the judge's rate alone** | fails | spike 017 plasmodes | 0.05 | **1.000** | 0-1 | 42 | 1,000, 4,000 | 0.6457 | 30 | 0 | 12 | [017 B6] |

- [a] The paper's row covers n 200-800: 3 cells, all over (0.1150-0.1840). The source table has n 1,200 and 2,400 as well (0.1010 each, unresolved).
- [b] The paper's row is one wording: 6 cells, 4 over and 2 at or under (0-0.9892).
- [c] Cells are label x size x checkpoint. The paper's printed range is of the largest miss over the checkpoints: row 6 at 0.05, 8 such cells, 0 over, 1 above delta (0.0234-0.0542); row 6 at 0.10, 8 such cells, 0 over, 0 above delta (0.0694-0.0970); row 11, reference-rate strata, 10 such cells, 0 over, 0 above delta (0.0308-0.0452); row 13, `b1w`, 2 such cells, 2 over, 2 above delta (0.2382-0.4432); row 13, pooled Wilson, 2 such cells, 2 over, 2 above delta (0.2396-0.4476); row 15, reference-rate strata, 10 such cells, 4 over, 6 above delta (0.0260-0.0862).
- Rows 1, 3, 4, 5 and 12: counts rebuilt as round(m x 5,000) from rates printed to three decimals; no class changes within the rounding. Row 5 is one printed row for two bounds.

## (c) Sentences whose hold/fail wording changes or needs a qualifier

1. **section 3, above Table 2.**
   > One rule is applied to every bound, favoured or not.

   The rule is stated, and the table's `kind` column and bold type are not its output. Row 8 is plain type and has a cell over its level. Rows 12, 13 and 18 print only the sizes or the wording where the bound fails: the source has 2 more Student-t cells (2 not over), 12 more rare-label cells at n_s 200 (6 not over) and 12 more sheet cells (9 not over). Rows 16 to 19 are bold with 1 of 6, 4 of 6, 11 of 18 and 12 of 42 cells at or under delta. Rows 6, 13 and 15 take the largest miss of 2-3 checkpoints before the threshold, a stricter test than the rule and one the other rows do not get. The replacement in (b) counts every cell the source has and leaves the verdict to the three count columns.

2. **section 3, caption of Table 2.**
   > Monte Carlo standard errors are 0.003-0.004 for 5,000 resamples and 0.011 for 400.

   Right for those two counts, and five other counts are in the table or the text: 4,000 draws (se 0.0034 at delta 0.05; rows 2, 8, 9, 11, 14, 15, 18, 19 and section 7), 1,000 (0.0069; block PPI, and spike 017's cells with 20,000 judged responses, where row 9's largest miss of 0.053 sits), 500 (0.0097; block PPI at N 20,000, and 0.0134 at delta 0.1 in section 4), 2,000 (0.0049; sections 4 and 8.4) and 200 (0.0212 at delta 0.1; the trajectory certificate).

3. **section 3, Table 2 row 8; section 6, Table 4; section 9.**
   > | `b1w` on a design-weighted sheet | approximate | sheets re-drawn by their real sampling rule | 0.05 | 0.001-0.059 | [017 4] |
   > | gold labels from a stratified sheet | design-weighted labels, `b1w` | approximate |
   > `b1w`, the bootstrap-t limits and the cluster bootstrap are checked by resampling, not proved at finite n. Table 2 is the evidence, with its Monte Carlo error.

   One of 18 cells is over its level: judge wording 2, planted rate 0.2, 0.0595 (238 of 4,000; p = 0.0039; 95% interval 0.0524-0.0673), against a band that ends at 0.0569. One more is unresolved (0.0517), 16 are at or under delta. After a Bonferroni correction for 18 cells none is over (the band then ends at 0.0604). The route does not read the judge's feature, so the source's second row for the same wording and rate is a second run of the same setting: it gave 0.0485, and the two together 0.0540 (432 of 8,000; p = 0.054; 95% interval 0.0491-0.0592), unresolved, above delta. This is the only bound the paper lists as usable that has a cell over.

4. **section 3, Table 2 row 6; section 5.**
   > | stratified Wilson-type `b1w` | approximate | 4 mid-rate labels (9-94%), real Granite-3.3-2B responses, n_s 100-200 | 0.05 / 0.10 | 0.023-0.054 / 0.069-0.097 | [013 H8] |
   > with coverage as in Table 2

   No cell over. At delta 0.05, 3 of 24 checkpoint cells are unresolved above delta, all three the 94% label at n_s 100 (0.0524, 0.0526, 0.0542); together 796 of 15,000 = 0.0531, still inside the band for that many draws (0.0536; p = 0.045, picked after the fact as the worst label). At delta 0.1 all 24 are at or under. The printed ranges are of the largest miss over three checkpoints (0.0234-0.0542 at 0.05); per cell the range is 0.0166-0.0542.

5. **abstract; introduction; section 3, Table 2 rows 1-5.**
   > Exact bounds hold their level.
   > Exact bounds hold everywhere we test.

   No exact-bound cell is over. Clopper-Pearson on i.i.d. labels: 58 of 61 cells at or under delta, 3 unresolved above it (0.0512, 0.0515, 0.0508, each at 4,000 draws). The other exact rows and checks (betting, Bentkus, Hoeffding and Anderson, the split study, block PPI): 141 of 141 at or under. For an exact bound a miss above delta can only be Monte Carlo noise, so 'hold' is true by the proof; the resampling shows no cell over, which is what the sentence can cite. It also calibrates the rule: a bound known to be valid lands in the unresolved class in 3 of 61 cells.

6. **abstract; introduction; section 6 (Position); section 10.**
   > The normal-quantile intervals of PPI++ and StratPPI, in our implementation, miss in up to 24% of draws at a nominal 5%, and a bootstrap-t limit restores the level.
   > a bootstrap-t limit on the same estimators holds (sections 3 to 6)
   > a studentised bootstrap that does
   > their estimators with a bootstrap-t limit do

   'Holds' and 'restores the level' are stronger than the check. At delta 0.05 PPI++ with a bootstrap-t limit is over in 0 of 222 cells, unresolved above delta in 13 and at or under in 209; StratPPI's estimator with the same limit is over in 0 of 68, unresolved in 5, at or under in 63. By section 3's own definition an unresolved cell is weaker than holding. The supported wording is 'is over its level in no cell'.

7. **section 6.**
   > The StratPPI estimator with a bootstrap-t limit holds in all 28 (largest miss 0.056)

   Over in 0 of 28; 5 are unresolved above delta (0.0560, 0.0517, 0.0505, 0.0530, 0.0527) and 23 at or under. The largest is 0.0560 (224 of 4,000; p = 0.046; 95% interval 0.0491-0.0636); the band ends at 0.0569.

8. **section 6.**
   > Stratifying on the judge and ignoring it within strata (`b1w`) also holds (largest miss 0.053)

   Over in 0 of 28, unresolved in 1 (0.0532 (213 of 4,000; p = 0.18; 95% interval 0.0465-0.0607)), at or under in 27.

9. **section 3, Table 2 rows 14 and 15; section 6 (Position); section 10.**
   > evidence that the published normal-quantile intervals, stratified or not, do not hold their level at the sample sizes and rates of a safety test
   > (the normal-quantile intervals of PPI++ and StratPPI do not; their estimators with a bootstrap-t limit do)

   True of most cells, not all, and the sentence reads as all. PPI++ with a normal limit on spike 017's plasmodes: over in 119 of 168, unresolved in 32, at or under delta in 17 (13 of those on the brevity label, where the judge carries nothing; 82 over after Bonferroni). StratPPI as published on judge-logit strata: over in 26 of 28, unresolved in 2. On reference-rate strata at mid rates: over in 9 of 28 checkpoint cells, unresolved in 5, at or under in 14, pooled miss 0.0498; at delta 0.1, 7 of 28 over. By rate, PPI++ with a normal limit is over in 103 of 116 cells at rates of 20% and below and in 16 of 52 at 40%. 'Over its level in most cells at rates of 20% and below, in fewer at mid rates' is what the counts support.

10. **section 5, after Table 3.**
   > It exceeds delta in 4 of 10 cells at delta 0.05 (one of them marginally, at 0.056; 3 of 10 at delta 0.1), most at the 9% label, and in every rare-label cell; `b1w` fails in three of those four.

   The counts are of cells over the band, not over delta: taking the paper's cell (the largest miss of a label's checkpoints at one n_s), 4 of 10 are over at 0.05 and 6 are above delta; 3 of 10 and 4 at 0.1. 'Exceeds delta' should read 'is over its level'. The marginal cell is 0.0564 (282 of 5,000; p = 0.022; 95% interval 0.0502-0.0632), over by 0.0002. By checkpoint, which is the rule's cell, 9 of 28 are over (7 after Bonferroni), 5 unresolved, 14 at or under. Rare labels: StratPPI over in 4 of 4, `b1w` in 3 of 4, as printed.

11. **section 3, Table 2 row 13 and reading 1; section 5.**
   > At 1-2% and n_s 100 the Wilson-type bounds, pooled or stratified, missed in 24-44% of draws
   > Rare labels (the approximate bound is invalid there and exact stratified bounds did not beat pooling)

   All 12 cells at n_s 100 are over, and stay over after Bonferroni. By checkpoint they run 0.0758-0.4476; 24-44% is the stratified bound's largest checkpoint for each label (0.2382 and 0.4432; the pooled bound's are 0.2396 and 0.4476). At n_s 200, which the row leaves out, `b1w` is over for the 1% label (0.1744, 0.0920, 0.0926) and at or under delta for the 2% label (0.0380, 0.0272, 0.0124): 'invalid there' holds at n_s 100 and for one of two labels at 200.

12. **section 3, Table 2 row 12 and reading 2.**
   > | **Student-t** | fails at low rates | same pool as row 1, n 200-800 | 0.10 | **0.115-0.184** | [R 6.2] |
   > It is anti-conservative exactly where trained policies sit (harm rates of 3-6%)

   Over at n 200, 400 and 800 (0.1840, 0.1150, 0.1210). At n 1,200 and 2,400, in the same source table and left out of the row, it is 0.1010 and 0.1010: unresolved above delta (p = 0.41 each). 'Over its level at n up to 800; unresolved at 1,200 and 2,400' is the rule's reading. Printed rates, not regenerable.

13. **section 3, reading 3.**
   > In Table 2 the approximate bounds we use miss at most one percentage point over delta (the largest is 0.060 at delta 0.05).

   True of the printed values. Under the rule the two largest fall in different classes: 0.0600 at 400 draws is unresolved (band to 0.0718), 0.0595 at 4,000 draws is over (band to 0.0569). A distance from delta is not a class without the draw count.

14. **section 3, reading 3; section 4.**
   > Inside the full training loop the stratified test reached 0.116 at delta 0.1, with a Monte Carlo standard error of 0.013 (section 4).
   > the stratified `b1w` test missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error 0.013), the same as a random split with a pooled bound [013 4]

   Three of four cells at or under delta, one unresolved above it: 0.1160 (58 of 500; p = 0.13; 95% interval 0.0893-0.1474). At 500 runs a cell the check resolves only a miss above 0.1268. The random split with a pooled bound is 0.0900-0.1000, all four at or under: 'the same' is 'not distinguishable at 500 runs', and the stratified test has the one cell above delta.

15. **section 4.**
   > A certificate at level delta/T over every one of T checks held, with misses of 0.025-0.100 against a delta of 0.1 [SR 2.1].

   All 10 cells at or under delta, the largest exactly at it (20 of 200; 95% interval 0.0622-0.1502). At 200 runs a cell the check resolves only a miss above 0.1424, so 'held' is 'at or under delta in every cell, at low resolution'.

16. **abstract; introduction; section 3, Table 2 row 16; section 7.3.**
   > On AgentDojo's published runs the usual per-pair bound misses in 5-28% of resamples
   > The per-pair Clopper-Pearson bound missed in 5-28% of resamples at delta 0.05 (Table 2).
   > Independent-sample bounds fail on crossed designs

   Over in 5 of 6 pipelines (0.0800-0.2750; 5 after Bonferroni by the widened band, 4 by exact p-values). The low end of the printed range is the sixth: 0.0475 (19 of 400; p = 0.62; 95% interval 0.0288-0.0732), at or under delta. 'Over its level in 5 of 6 pipelines, 8-28%' is the rule's reading; '5-28%' counts a cell under delta as a miss of the level.

17. **section 3, Table 2 row 17.**
   > | **two-way bootstrap** | fails where positives sit in few clusters | AgentDojo | 0.05 | **up to 0.170** | [020 P] |

   Over in 2 of 6 pipelines (0.1700 and 0.1025), at or under delta in 4; pooled over the six it is 0.0525, unresolved. The `kind` column carries the qualifier; the count belongs beside it.

18. **section 3, Table 2 row 10; section 8.2; section 11.**
   > The check behind this is narrow: 6 of the 28 pipelines, 400 resamples each (standard error 0.011)
   > Where we could check against a known truth, exact bounds and design-respecting resampling held

   Cluster bootstrap-t by user task: over in 0 of 6, unresolved above delta in 2 (0.0525, 0.0600), at or under in 4. At 400 draws the check resolves only a miss above 0.0718, so 'held' is 'was not over its level in six pipelines'. The other half of section 8.2's certificate, the bootstrap-t by injection task, is over for one pipeline in the same file (Meta-SecAlign-70B: 0.1225 (49 of 400; p < 0.0001; 95% interval 0.0920-0.1587)). For that pipeline the user-task bound is the larger of the two in Table 6 (0.104 against 0.035), so the rule would use it; the rule itself was not resampled, as the paper says.

19. **section 3, Table 2 row 18; section 7.5.**
   > Read as i.i.d., it certified a negative harm rate under one judge wording, and missed in 98% of re-drawn sheets [017 4].

   Over in 7 of 18 cells and at or under delta in 11: 4 of 6 for the wording the paper names (0.5885-0.9892; its two cells at a 20% planted rate are at or under), 3 of 6 for a second wording (0.0927-0.6983), none of 6 for the third. The sentence says 'one judge wording'; the row's 'fails' needs the same words. The largest miss is 0.9892 (the logit feature) and 0.9838 (the 0/1 verdict); '98%' is the second.

20. **section 3, Table 2 row 19.**
   > | **the judge's rate alone** | fails | spike 017 plasmodes | 0.05 | **1.000** | [017 B6] |

   1.000 is the largest miss, reached in 26 of 42 cells: every wording of the compiled rubric, whose judged rate is far under the truth. The raw wording gives 0.1663-0.1765 in its 4 cells on the brevity label (over) and 0 in its 12 cells on the refusal label, where its judged rate (0.30 on the pool, against a truth of 0.20) is above the truth and the bound is never under it. At or under delta there says the judge over-reports, not that the route is valid. The row needs 'up to 1.000, where the judge under-reports'.

21. **section 8.4.**
   > puts the miss rates of the four limits at 0.001, 0.040, 0.045 and 0.042 for the pool rate against a level of 0.05

   All four at or under delta at 2,000 repetitions, as printed. The qualifier is the scheme: limit (b2) is built for new prompts, and in the same check's new-prompts scheme, which the paper does not quote, it is 0.0520 (104 of 2,000; p = 0.35; 95% interval 0.0427-0.0627) and the labels-alone bootstrap-t is 0.0515, both unresolved above delta; (b1), not built for that scheme, is 0.0630, over.

22. **section 11.**
   > three common shortcuts failed by wide margins: a normal quantile at small rates, a per-pair bound on a crossed design, and a judge calibration carried to a policy it was not measured on

   Wide in the worst cells (0.2412, 0.2750, 0.8027), and each shortcut also has cells that are not over: 49 of 168 for PPI++ with a normal limit, 1 of 6 for the per-pair bound, 3 of 6 wordings for the calibration carried across the constrained run. 'Failed in most cells, by wide margins in the worst' is what the counts carry.

Sentences checked that stand as written:

- **section 4.** "Clopper-Pearson missed in at most 1 of 2,000 runs in any cell [012]." 42 of 42 cells at or under delta 0.05; the largest is 1 of 2,000.
- **section 5.** "The same estimator with a bootstrap-t limit holds in all ten cells (largest miss 0.045)" Stronger by checkpoint: 28 of 28 cells at or under delta at 0.05 and 28 of 28 at 0.1, none unresolved.
- **section 5.** "A stratified Wald-t limit on the same strata also held in all ten cells (largest miss 0.036" 28 of 28 checkpoint cells at or under delta.
- **section 6.** "StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in those 26), as PPI++ with a normal limit is in all 14 (0.061-0.232)." 26 of 28 over (0.0590-0.2387), 25 after Bonferroni; 14 of 14 over (0.0610-0.2320), 14 after Bonferroni.
- **section 7.1.** "Between the two benign sources the bound was over its level for 4 of 6 wordings one way and 1 of 6 the other; between the pools, for 5 of 6 one way and none the other." 4, 1, 5 and 0 of 6 over; every other cell is at or under delta, and the counts are the same after Bonferroni for six wordings.
- **section 7.2.** "the carried bound was over its level for 3 of 6 (misses 0.08, 0.23 and 0.80) and marginal for a fourth (0.054)" 3 over, 1 unresolved above delta (0.0537 (215 of 4,000; p = 0.15; 95% interval 0.0470-0.0612)), 2 at or under. 'Marginal' is the rule's 'unresolved'. The spike's own summary counts 4 of 6, by misses above 0.05; the paper's count is the rule's.
- **section 7.2.** "At step 100, where the rate had moved by 3.4 points, 1 of 6 failed." 1 of 6 over, 5 at or under.
- **section 7.2.** "A calibration carried across 200 steps of one training run that moved the label only as a side effect" 6 of 6 at or under delta; the largest is 0.0415.
- **section 7.4.** "Finite-sample judge-assisted bounds (betting on blocks) were valid and bought nothing" 84 of 84 cells at or under delta; the largest is 0.0440.
- **section 3, Table 2 rows 1, 3, 4, 5 and 7.** "(rows as printed)" Every cell at or under delta: 5 each for Clopper-Pearson, the betting mixture, Bentkus, and Hoeffding and Anderson (printed rates, 5,000 draws), and 4 of 4 for `b1w` on the pushed label.

## (d) The two lists the rule produces

**Bounds the paper says hold, with a cell over.**

- `b1w` on a design-weighted sheet (Table 2 row 8, [017 4]): wording 2, planted rate 0.2, f01, 0.0595 (238 of 4,000; p = 0.0039; 95% interval 0.0524-0.0673); over after Bonferroni for 18 cells: no.

**Bounds the paper says fail, with their cells that are not over.**

- Student-t, real harm labels, trained-policy pool, rate 0.034, n 200-2,400, delta 0.1 (Table 2 row 12): 3 of 5 over, 2 unresolved, 0 at or under; pooled miss 0.1244.
- PPI++ with a normal limit, spike 017 plasmodes, every cell and feature (the cells of its table B6), delta 0.05 (Table 2 row 14): 119 of 168 over, 32 unresolved, 17 at or under; pooled miss 0.0791.
- StratPPI as published (normal limit), 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.05 (Table 2 row 15; Table 3): 9 of 28 over, 5 unresolved, 14 at or under; pooled miss 0.0498.
- StratPPI as published (normal limit), judge-logit strata (5 and 10) on the refusal pools, n 100-1,000, delta 0.05 (Table 2 row 15; section 6): 26 of 28 over, 2 unresolved, 0 at or under; pooled miss 0.0982.
- Clopper-Pearson over pairs, AgentDojo, 6 pipelines, user tasks resampled, delta 0.05 (Table 2 row 16): 5 of 6 over, 0 unresolved, 1 at or under; pooled miss 0.1500.
- two-way bootstrap, AgentDojo, 6 pipelines, user tasks resampled, delta 0.05 (Table 2 row 17): 2 of 6 over, 0 unresolved, 4 at or under; pooled miss 0.0525.
- stratified sheet read as an i.i.d. sample (PPI), sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features, delta 0.05 (Table 2 row 18): 7 of 18 over, 0 unresolved, 11 at or under; pooled miss 0.2364.
- the judge's rate alone, spike 017 plasmodes, every cell and feature (the cells of its table B6), delta 0.05 (Table 2 row 19): 30 of 42 over, 0 unresolved, 12 at or under; pooled miss 0.6457.
- `b1w` at rare rates, labels at 1-2%, n_s 200, 3 checkpoints, delta 0.05 (sections 3 and 5 (rare labels)): 3 of 6 over, 0 unresolved, 3 at or under; pooled miss 0.0728.
- pooled Wilson bound at rare rates, labels at 1-2%, n_s 200, 3 checkpoints, delta 0.05 (sections 3 and 5 (rare labels)): 3 of 6 over, 0 unresolved, 3 at or under; pooled miss 0.0785.
- StratPPI as published (normal limit), 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.1 (section 5): 7 of 28 over, 3 unresolved, 18 at or under; pooled miss 0.0947.
- carried Youden-corrected bound, source: xstest -> orbench, six judge wordings, delta 0.05 (section 7.1): 4 of 6 over, 0 unresolved, 2 at or under; pooled miss 0.4020.
- carried Youden-corrected bound, source: orbench -> xstest, six judge wordings, delta 0.05 (section 7.1): 1 of 6 over, 0 unresolved, 5 at or under; pooled miss 0.0222.
- carried Youden-corrected bound, pool: harmful -> over-refusal, six judge wordings, delta 0.05 (section 7.1): 5 of 6 over, 0 unresolved, 1 at or under; pooled miss 0.4275.
- carried Youden-corrected bound, training, constrained (014): step 0 -> step 100, six judge wordings, delta 0.05 (section 7.2): 1 of 6 over, 0 unresolved, 5 at or under; pooled miss 0.1047.
- carried Youden-corrected bound, training, constrained (014): step 0 -> step 200, six judge wordings, delta 0.05 (section 7.2): 3 of 6 over, 1 unresolved, 2 at or under; pooled miss 0.1968.

Of these, with no cell over (only unresolved or under): none.

## (e) Printed numbers against their sources

| where | the paper | the source, per cell | how the printed value arises |
|---|---|---|---|
| Table 2 row 6 | 0.023-0.054 / 0.069-0.097 | 0.0166-0.0542 / 0.0576-0.0970 | the largest of 3 checkpoints per label and size: 0.0234-0.0542 / 0.0694-0.0970 |
| Table 2 row 13; section 3 reading 1 | 0.24-0.44 | 0.0758-0.4476 | the stratified bound's largest checkpoint per label, 0.2382 and 0.4432; the pooled bound's are 0.2396 and 0.4476 |
| Table 2 row 12 | n 200-800, 0.115-0.184 | n 200-2,400, 0.1010-0.1840 | the three cells the source prints in bold |
| Table 2 row 18; section 7.5 | up to 0.98; 98% | largest 0.9892 | the 0/1-verdict cell (0.9838); the logit cell is higher |
| Table 2 row 19 | 1.000 | 1.000 in 26 cells, 0.1663-0.1765 in 4, 0 in 12 | the largest over cells (spike 017's table B6). `reports/figs/fig1_validity.csv` prints a lowest miss of 1.000 for this row |
| Table 2 row 16; abstract; section 7.3 | 0.048-0.275; 5-28% | 0.0475-0.2750 | reproduces; the low end is at or under delta |
| Table 2 rows 2 and 8 | at most 0.051; 0.001-0.059 | 0.0515; 0.0010-0.0595 | reproduces; 0.0515 and 0.0595 print as 0.051 and 0.059 in the sources' three decimals |
| Table 2 caption | se 0.003-0.004 (5,000), 0.011 (400) | also 4,000, 2,000, 1,000, 500 and 200 draws | see (c) item 2 |

Reproduced to the printed digits (asserted in the script): Table 2 rows 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16 and 17; the counts '4 of 10', '3 of 10', '26 of 28' and 'all 14' of sections 5 and 6 under the paper's own cells; section 4's 0.084-0.116 and 0.025-0.100; section 7's 4, 1, 5, 0, 3 and 1 of 6; section 8.4's four rates.

## (f) Not recounted, and why

- **[R 6.2], Table 2 rows 1, 3, 4, 5, 12.** Not regenerable: the cached labels were lost on 2026-09-19. Classified from the printed rates and the 5,000 resamples the training paper states; counts are round(m x 5,000).
- **[R 6.2], the Student-t bound at delta 0.05.** The training paper's sentence (0.084, 0.067, 0.083, 0.074, 0.067) repeats its Clopper-Pearson row in four of five values (the paper's Appendix A, open check 2). Not classified; Table 2 does not use it.
- **[R 6.3], section 4, tables (a), (b) and (d).** 11 printed cells with no class: the training paper gives '500-1,000 independent trials' a row and no count per row. Every printed rate (at most 0.0030) is under delta 0.1 whatever the count; the p-values and intervals need it. Table (c) is classified at the 500 trials of the reproduction command in that paper's Appendix B. Regenerable (`scripts/synthetic_calibration.py`), not re-run here.
- **Section 4, language-model policies.** 0 breaches in 3 seeds and in 10 seeds: the paper already declines to read these as a miss rate, and three or ten draws give no class worth the name.
- **Section 8.2, the larger-of-two clustered bound.** The certificate the paper uses was not resampled in the source; its two halves were, one at a time, with user tasks resampled.
- **Clustering of the sheet draws.** `harm.json` stores one rate per cell, not per planting, so a planting-level standard error cannot be formed without re-running `harm017.py`.
- **[SR 2.4].** No row of Table 2 in v0.5 carries this tag; the state report's section 2.4 digests spike 017, whose cells are counted from its own files above.
- **`plasmode_n500.json` (spike 017).** Not part of the spike's table B6 and not quoted in the paper; left out.

## Appendix. The cells behind (c) and (d)

Every cell that is not at or under delta in a row the paper says holds, and every cell that is not over in a row it says fails (rows with more than 12 such cells give the count; all cells are in the JSON).

| bound | setting | the paper says | cell | delta | draws | misses | miss | class | exact p | 95% CP |
|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson, labels alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | refusal rubric 0, rate 0.200, n 50 of 2000, - | 0.05 | 4,000 | 205 | 0.0512 | unresolved, above delta | 0.37 | [0.0446, 0.0585] |
| Clopper-Pearson, labels alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | refusal rubric 0, rate 0.200, n 100 of 2000, - | 0.05 | 4,000 | 206 | 0.0515 | unresolved, above delta | 0.34 | [0.0449, 0.0588] |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | holds | C2:refusal, n_s 100, step0 | 0.05 | 5,000 | 262 | 0.0524 | unresolved, above delta | 0.23 | [0.0464, 0.0589] |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | holds | C2:refusal, n_s 100, step100 | 0.05 | 5,000 | 263 | 0.0526 | unresolved, above delta | 0.21 | [0.0466, 0.0592] |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | holds | C2:refusal, n_s 100, step200 | 0.05 | 5,000 | 271 | 0.0542 | unresolved, above delta | 0.093 | [0.0481, 0.0608] |
| `b1w` on a design-weighted sheet | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | holds | wording 0, planted rate 0.2, f01 | 0.05 | 4,000 | 207 | 0.0517 | unresolved, above delta | 0.32 | [0.0451, 0.0591] |
| `b1w` on a design-weighted sheet | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | holds | wording 2, planted rate 0.2, f01 | 0.05 | 4,000 | 238 | 0.0595 | over | 0.0039 | [0.0524, 0.0673] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | brevity raw 0, rate 0.404, n 500 of 2000, platt | 0.05 | 4,000 | 203 | 0.0508 | unresolved, above delta | 0.42 | [0.0442, 0.0580] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | brevity rubric 1, rate 0.404, n 225 of 2000, logit | 0.05 | 4,000 | 201 | 0.0503 | unresolved, above delta | 0.48 | [0.0437, 0.0575] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | brevity rubric 3, rate 0.404, n 225 of 2000, f01 | 0.05 | 4,000 | 206 | 0.0515 | unresolved, above delta | 0.34 | [0.0449, 0.0588] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | brevity rubric 3, rate 0.404, n 225 of 2000, p | 0.05 | 4,000 | 204 | 0.0510 | unresolved, above delta | 0.4 | [0.0444, 0.0583] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | brevity rubric 3, rate 0.404, n 225 of 2000, logit | 0.05 | 4,000 | 206 | 0.0515 | unresolved, above delta | 0.34 | [0.0449, 0.0588] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | refusal raw 0, rate 0.200, n 225 of 2000, p | 0.05 | 4,000 | 203 | 0.0508 | unresolved, above delta | 0.42 | [0.0442, 0.0580] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | refusal rubric 0, rate 0.200, n 1000 of 20000, logit | 0.05 | 1,000 | 53 | 0.0530 | unresolved, above delta | 0.35 | [0.0399, 0.0688] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | refusal rubric 0, rate 0.200, n 1000 of 20000, platt | 0.05 | 1,000 | 51 | 0.0510 | unresolved, above delta | 0.46 | [0.0382, 0.0665] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | shifted refusal rubric 0, rate 0.200, n 1000 of 4000, f01 | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | shifted refusal rubric 0, rate 0.200, n 1000 of 4000, p | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | shifted refusal raw 0, rate 0.200, n 1000 of 4000, f01 | 0.05 | 4,000 | 208 | 0.0520 | unresolved, above delta | 0.29 | [0.0453, 0.0593] |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | shifted refusal raw 0, rate 0.200, n 1000 of 4000, p | 0.05 | 4,000 | 209 | 0.0522 | unresolved, above delta | 0.27 | [0.0456, 0.0596] |
| cluster bootstrap-t, by user task | AgentDojo, 6 pipelines, user tasks resampled | holds | gpt-4o-2024-05-13-tool_filter | 0.05 | 400 | 21 | 0.0525 | unresolved, above delta | 0.44 | [0.0328, 0.0791] |
| cluster bootstrap-t, by user task | AgentDojo, 6 pipelines, user tasks resampled | holds | gemini-2.0-flash-001 | 0.05 | 400 | 24 | 0.0600 | unresolved, above delta | 0.21 | [0.0388, 0.0880] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal rubric 0, rate pool, n 100 of 2000, K=5 | 0.05 | 4,000 | 224 | 0.0560 | unresolved, above delta | 0.046 | [0.0491, 0.0636] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal rubric 0, rate pool, n 100 of 2000, K=10 | 0.05 | 4,000 | 207 | 0.0517 | unresolved, above delta | 0.32 | [0.0451, 0.0591] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate pool, n 100 of 2000, K=5 | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate pool, n 500 of 2000, K=10 | 0.05 | 4,000 | 212 | 0.0530 | unresolved, above delta | 0.2 | [0.0463, 0.0604] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate 0.05, n 1000 of 4000, K=5 | 0.05 | 4,000 | 211 | 0.0527 | unresolved, above delta | 0.22 | [0.0460, 0.0601] |
| Student-t | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | fails | n 1200 | 0.1 | 5,000 | ~505 | 0.1010 | unresolved, above delta | 0.41 | [0.0928, 0.1097] |
| Student-t | real harm labels, trained-policy pool, rate 0.034, n 200-2,400 | fails | n 2400 | 0.1 | 5,000 | ~505 | 0.1010 | unresolved, above delta | 0.41 | [0.0928, 0.1097] |
| PPI++ with a normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | 49 cells, 0.0408-0.0620 | 0.05 | 1,000, 4,000 | | | 32 unresolved, 17 at or under | | |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 19 cells, 0.0238-0.0560 | 0.05 | 5,000 | | | 5 unresolved, 14 at or under | | |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | fails | refusal rubric 0, rate pool, n 500 of 2000, K=10 | 0.05 | 4,000 | 214 | 0.0535 | unresolved, above delta | 0.16 | [0.0467, 0.0609] |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000, K=5 | 0.05 | 4,000 | 225 | 0.0563 | unresolved, above delta | 0.04 | [0.0493, 0.0638] |
| Clopper-Pearson over pairs | AgentDojo, 6 pipelines, user tasks resampled | fails | claude-3-5-sonnet-20241022 | 0.05 | 400 | 19 | 0.0475 | at or under delta | 0.62 | [0.0288, 0.0732] |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled | fails | claude-3-7-sonnet-20250219 | 0.05 | 400 | 5 | 0.0125 | at or under delta | 1 | [0.0041, 0.0289] |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled | fails | gpt-4o-2024-05-13-tool_filter | 0.05 | 400 | 2 | 0.0050 | at or under delta | 1 | [0.0006, 0.0179] |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled | fails | gemini-2.0-flash-001 | 0.05 | 400 | 10 | 0.0250 | at or under delta | 1 | [0.0121, 0.0455] |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled | fails | gpt-4o-2024-05-13 | 0.05 | 400 | 0 | 0 | at or under delta | 1 | [0, 0.0092] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.013, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.013, logit | 0.05 | 4,000 | 22 | 0.0055 | at or under delta | 1 | [0.0034, 0.0083] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.05, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.05, logit | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.2, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 0, planted rate 0.2, logit | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 2, planted rate 0.2, f01 | 0.05 | 4,000 | 13 | 0.0032 | at or under delta | 1 | [0.0017, 0.0056] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 2, planted rate 0.2, logit | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 4, planted rate 0.05, f01 | 0.05 | 4,000 | 12 | 0.0030 | at or under delta | 1 | [0.0016, 0.0052] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 4, planted rate 0.2, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | fails | wording 4, planted rate 0.2, logit | 0.05 | 4,000 | 4 | 0.0010 | at or under delta | 1 | [0.0003, 0.0026] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 50 of 2000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 100 of 2000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 225 of 2000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 500 of 2000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 225 of 20000, f01 | 0.05 | 1,000 | 0 | 0 | at or under delta | 1 | [0, 0.0037] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 1000 of 20000, f01 | 0.05 | 1,000 | 0 | 0 | at or under delta | 1 | [0, 0.0037] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.200, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.200, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.050, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.050, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.013, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.013, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| `b1w` at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step0 | 0.05 | 5,000 | 190 | 0.0380 | at or under delta | 1 | [0.0329, 0.0437] |
| `b1w` at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step100 | 0.05 | 5,000 | 136 | 0.0272 | at or under delta | 1 | [0.0229, 0.0321] |
| `b1w` at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step200 | 0.05 | 5,000 | 62 | 0.0124 | at or under delta | 1 | [0.0095, 0.0159] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step0 | 0.05 | 5,000 | 239 | 0.0478 | at or under delta | 0.77 | [0.0421, 0.0541] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step100 | 0.05 | 5,000 | 151 | 0.0302 | at or under delta | 1 | [0.0256, 0.0353] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 200, 3 checkpoints | fails | C2:gated, n_s 200, step200 | 0.05 | 5,000 | 66 | 0.0132 | at or under delta | 1 | [0.0102, 0.0168] |
| `b1w` in the training loop, reference-rate strata | synthetic bandit, 4 heterogeneity levels, `SeldonianLLMPolicy` | holds | icc05, strat_ref, b1w_strat_pop | 0.1 | 500 | 58 | 0.1160 | unresolved, above delta | 0.13 | [0.0893, 0.1474] |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 21 cells, 0.0636-0.1082 | 0.1 | 5,000 | | | 3 unresolved, 18 at or under | | |
| Clopper-Pearson, labels alone | refusal pools, random labelled subset, n 100-1,000 | holds | refusal rubric 0, rate pool, n 100 of 2000 | 0.05 | 4,000 | 203 | 0.0508 | unresolved, above delta | 0.42 | [0.0442, 0.0580] |
| PPI++ with a bootstrap-t limit | refusal pools, random labelled subset, n 100-1,000 | holds | refusal raw 0, rate pool, n 225 of 2000 | 0.05 | 4,000 | 207 | 0.0517 | unresolved, above delta | 0.32 | [0.0451, 0.0591] |
| `b1w` on judge-logit strata | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal rubric 0, rate 0.013, n 225 of 4000, K=10 | 0.05 | 4,000 | 213 | 0.0532 | unresolved, above delta | 0.18 | [0.0465, 0.0607] |
| carried Youden-corrected bound | source: xstest -> orbench, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 15 | 0.0037 | at or under delta | 1 | [0.0021, 0.0062] |
| carried Youden-corrected bound | source: xstest -> orbench, six judge wordings | fails | wording 2 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | fails | wording 0 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 3 | 0.0008 | at or under delta | 1 | [0.0002, 0.0022] |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | pool: harmful -> over-refusal, six judge wordings | fails | wording 2 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 30 | 0.0075 | at or under delta | 1 | [0.0051, 0.0107] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | fails | wording 2 | 0.05 | 4,000 | 134 | 0.0335 | at or under delta | 1 | [0.0281, 0.0396] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 98 | 0.0245 | at or under delta | 1 | [0.0199, 0.0298] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 173 | 0.0432 | at or under delta | 0.98 | [0.0372, 0.0500] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 24 | 0.0060 | at or under delta | 1 | [0.0038, 0.0089] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 200, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 30 | 0.0075 | at or under delta | 1 | [0.0051, 0.0107] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 200, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 32 | 0.0080 | at or under delta | 1 | [0.0055, 0.0113] |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 200, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 215 | 0.0537 | unresolved, above delta | 0.15 | [0.0470, 0.0612] |

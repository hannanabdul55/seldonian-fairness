# Validity recount under one rule

`scripts/validity_recount.py`; source: `reports/paper_certification.md` (draft v0.9.11) and the result files it cites. Nothing was simulated: 7,427 cells and 53,173,000 draws are counted from existing files, or read from the training paper's printed tables where the data are lost (marked `~`). The script stops if a printed number of Table 4, or one of the sentences of section (c), does not agree with the files.

**The rule.** A cell is one resampling study: R draws at level delta, miss m. With se = sqrt(delta (1 - delta) / R): *over* if m > delta + 2 se; *unresolved, above delta* if delta < m <= delta + 2 se; *at or under delta* if m <= delta. The exact one-sided binomial p-value of H0 'true miss <= delta' and a 95% Clopper-Pearson interval are in the JSON for every cell, and below for the cells that matter. *Over after Bonferroni*: the 2 se replaced by z(1 - a / C) se for the C cells of the row, a = 1 - Phi(2) = 0.0228, so one cell gives the rule itself.

**What a cell is.** The unit a source file stores: one label, sample size, checkpoint, feature or pipeline, with its own draws. Table A1 and one count of section 8.1 take the largest miss of 2-3 checkpoints as one cell; both counts are given where they differ.

**Four cautions.** (1) Cells of one row are not independent: the two deltas of spikes 013 and 014 use the same draws, the four judge features of spike 017 share a cell's draws, cells of one checkpoint and size share their seeds across labels, and arms are paired. Bonferroni is then conservative, and the pooled miss is a description, not a test. (2) The 4,000 draws of a sheet cell are 20 plantings x 200 sheets; the per-planting counts were not stored, so the binomial treats them as 4,000. (3) *At or under delta* is a statement about the estimate; with 200-500 draws its interval still reaches well above delta (see the largest-miss column). (4) The reference-rate-strata cells of spikes 013 and 014 and of `stratppi.json` draw 20-40% of a pool of about 500 prompts without replacement and take the pool's rate as the truth, which makes every bound there look more conservative than it is for a large pool; the rows marked *redrawn with replacement* are the same cells without that help.

**The correction of 2026-10-06.** Until then `b1w` (and the pooled Wilson bound, which is `b1w` at one stratum) returned its estimate when the sample held no positive. The rare-label cells of spike 013 then read 0.076-0.448 at delta 0.05 and n_s 100, the chance of drawing no positive, and earlier versions of this file classed all twelve as over. Every caller was rerun with its original seeds (`reports/b1w_fix_and_audit_2026-10-06.md`); the counts below are from the corrected files.

## (a) Per-bound summary

Rows the paper quotes, in the order of Table 4 and then by section.

| bound | setting | delta | source | where in the paper | the paper says | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.1 | [binomial rows] | Table 4 row 1 | holds | 20 | exact enumeration | - | - | 0.0959 (exact) | 0 | 0 | 20 | 0 |
| Clopper-Pearson, labels alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 4 row 2 | holds | 42 | 4,000 | 168,000 | 0.0383 (at or under delta) | 0.0515 [0.0449, 0.0588] | 0 | 2 | 40 | 0 |
| betting mixture | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.1 | [binomial rows] | Table 4 row 3 | holds | 20 | exact enumeration | - | - | 0.0184 (exact) | 0 | 0 | 20 | 0 |
| Bentkus | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.1 | [binomial rows] | Table 4 row 4 | holds | 20 | exact enumeration | - | - | 0.0367 (exact) | 0 | 0 | 20 | 0 |
| Hoeffding, Anderson | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.1 | [binomial rows] | Table 4 row 5 | holds | 40 | exact enumeration | - | - | 0.0000 (exact) | 0 | 0 | 40 | 0 |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | 0.05 | [013 H8] | Table 4 row 6 | holds | 24 | 5,000 | 120,000 | 0.0336 (at or under delta) | 0.0542 [0.0481, 0.0608] | 0 | 3 | 21 | 0 |
| stratified Wilson-type `b1w` | 4 mid-rate labels (9-94%), Granite-3.3-2B, n_s 100-200, 3 checkpoints | 0.1 | [013 H8] | Table 4 row 6 | holds | 24 | 5,000 | 120,000 | 0.0776 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 24 | 0 |
| `b1w` | label pushed by the Lagrangian, n_s 200, steps 100 and 200 | 0.05 | [014] | Table 4 row 7 | holds | 2 | 5,000 | 10,000 | 0.0169 (at or under delta) | 0.0228 [0.0188, 0.0273] | 0 | 0 | 2 | 0 |
| `b1w` | label pushed by the Lagrangian, n_s 200, steps 100 and 200 | 0.1 | [014] | Table 4 row 7 | holds | 2 | 5,000 | 10,000 | 0.0520 (at or under delta) | 0.0642 [0.0576, 0.0714] | 0 | 0 | 2 | 0 |
| `b1w` on a design-weighted sheet | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | Table 4 row 8 | holds | 18 | 4,000 | 72,000 | 0.0232 (at or under delta) | 0.0595 [0.0524, 0.0673] | 1 | 1 | 16 | 0 |
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 4 row 9 | holds | 168 | 4,000 | 672,000 | 0.0405 (at or under delta) | 0.0535 [0.0467, 0.0609] | 0 | 13 | 155 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 10; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0415 (at or under delta) | 0.0505 [0.0439, 0.0577] | 0 | 2 | 26 | 0 |
| StratPPI estimator with a bootstrap-t limit | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Table 4 row 11; Table A1 | holds | 28 | 5,000 | 140,000 | 0.0345 (at or under delta) | 0.0452 [0.0396, 0.0513] | 0 | 0 | 28 | 0 |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | Table 4 row 11; section 8.2 | holds | 28 | 4,000 | 112,000 | 0.0397 (at or under delta) | 0.0560 [0.0491, 0.0636] | 0 | 5 | 23 | 0 |
| StratPPI estimator with a bootstrap-t limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | Table 4 row 12; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0424 (at or under delta) | 0.0496 [0.0475, 0.0518] | 0 | 0 | 28 | 0 |
| StratPPI estimator with a bootstrap-t limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | Table 4 row 12; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0911 (at or under delta) | 0.1003 [0.0973, 0.1033] | 0 | 1 | 27 | 0 |
| Clopper-Pearson, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | 0.05 | [replacement check] | Table 4 row 13; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0416 (at or under delta) | 0.0497 [0.0476, 0.0518] | 0 | 0 | 28 | 0 |
| Clopper-Pearson, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | 0.1 | [replacement check] | Table 4 row 13; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0779 (at or under delta) | 0.0970 [0.0942, 0.1000] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [two-way bounds] | Table 4 row 14; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0009 (at or under delta) | 0.0043 [0.0025, 0.0068] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [two-way bounds] | Table 4 row 14; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0123 (at or under delta) | 0.0370 [0.0314, 0.0433] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [two-way bounds] | Table 4 row 14; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0245 (at or under delta) | 0.0418 [0.0358, 0.0484] | 0 | 0 | 28 | 0 |
| Student-t | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.1 | [binomial rows] | Table 4 row 15 | fails | 20 | exact enumeration | - | - | 0.1951 (exact) | 20 | 0 | 0 | 20 |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | 0.05 | [013 H8] | Table 4 row 16 | not over at delta 0.05 | 12 | 5,000 | 60,000 | 0.0065 (at or under delta) | 0.0380 [0.0329, 0.0437] | 0 | 0 | 12 | 0 |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | 0.05 | [013 H8] | Table 4 row 16 | not over at delta 0.05 | 12 | 5,000 | 60,000 | 0.0076 (at or under delta) | 0.0478 [0.0421, 0.0541] | 0 | 0 | 12 | 0 |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | 0.1 | [013 H8] | Table 4 row 16 | over in 2 cells at delta 0.10 (both bounds together) | 12 | 5,000 | 60,000 | 0.0440 (at or under delta) | 0.1260 [0.1169, 0.1355] | 1 | 0 | 11 | 1 |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | 0.1 | [013 H8] | Table 4 row 16 | over in 2 cells at delta 0.10 (both bounds together) | 12 | 5,000 | 60,000 | 0.0467 (at or under delta) | 0.1258 [0.1167, 0.1353] | 1 | 1 | 10 | 1 |
| `b1w`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | Table 4 row 17; section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.0389 (at or under delta) | 0.0624 [0.0600, 0.0648] | 7 | 1 | 20 | 7 |
| `b1w`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | Table 4 row 17; section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.0863 (at or under delta) | 0.1128 [0.1097, 0.1159] | 6 | 2 | 20 | 4 |
| pooled Wilson bound, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | 0.05 | [replacement check] | Table 4 row 18; section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.0477 (at or under delta) | 0.0689 [0.0664, 0.0714] | 6 | 0 | 22 | 6 |
| pooled Wilson bound, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | 0.1 | [replacement check] | Table 4 row 18; section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.0967 (at or under delta) | 0.1275 [0.1243, 0.1309] | 10 | 3 | 15 | 10 |
| PPI++ with a normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 4 row 19 | fails | 168 | 4,000 | 672,000 | 0.0783 (over) | 0.2412 [0.2281, 0.2548] | 126 | 29 | 13 | 87 |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Table 4 row 20; Table A1 | fails | 28 | 5,000 | 140,000 | 0.0498 (at or under delta) | 0.0862 [0.0786, 0.0943] | 9 | 5 | 14 | 7 |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | Table 4 row 20; section 8.2 | fails | 28 | 4,000 | 112,000 | 0.0982 (over) | 0.2387 [0.2256, 0.2523] | 26 | 2 | 0 | 25 |
| StratPPI's normal limit, oracle allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | 0.05 | [StratPPI validation] | Table 4 row 21; Appendix A | fails | 26 | 5,000 | 130,000 | 0.0475 (at or under delta) | 0.1098 [0.1013, 0.1188] | 9 | 3 | 14 | 8 |
| StratPPI's normal limit, oracle allocation | judge-logit strata (5 and 10), n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 21; Appendix A | fails | 28 | 4,000 | 112,000 | 0.0945 (over) | 0.2188 [0.2060, 0.2319] | 28 | 0 | 0 | 25 |
| StratPPI's normal limit, heuristic allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | 0.05 | [StratPPI validation] | Table 4 row 21; Appendix A | fails | 26 | 5,000 | 130,000 | 0.0723 (over) | 0.2172 [0.2058, 0.2289] | 11 | 0 | 15 | 11 |
| StratPPI's normal limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 21; Appendix A | fails | 28 | 4,000 | 112,000 | 0.3317 (over) | 0.9025 [0.8929, 0.9115] | 24 | 1 | 3 | 24 |
| StratPPI estimator with a bootstrap-t limit, oracle allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | 0.05 | [StratPPI validation] | Table 4 row 22; Appendix A | fails | 26 | 5,000 | 130,000 | 0.0453 (at or under delta) | 0.0782 [0.0709, 0.0860] | 6 | 3 | 17 | 5 |
| StratPPI estimator with a bootstrap-t limit, oracle allocation | judge-logit strata (5 and 10), n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 22; Appendix A | fails | 28 | 4,000 | 112,000 | 0.0589 (over) | 0.0830 [0.0746, 0.0920] | 14 | 11 | 3 | 9 |
| StratPPI estimator with a bootstrap-t limit, heuristic allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | 0.05 | [StratPPI validation] | Table 4 row 22; Appendix A | fails | 26 | 5,000 | 130,000 | 0.0694 (over) | 0.1738 [0.1634, 0.1846] | 11 | 1 | 14 | 11 |
| StratPPI estimator with a bootstrap-t limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 22; Appendix A | fails | 28 | 4,000 | 112,000 | 0.2942 (over) | 0.9012 [0.8916, 0.9103] | 14 | 0 | 14 | 14 |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 23; section 8.2 | fails | 14 | 4,000 | 56,000 | 0.0653 (over) | 0.1328 [0.1224, 0.1437] | 7 | 6 | 1 | 5 |
| PPBoot, percentile limit, power-tuned | judge-logit cells, unstratified, n 100-1,000 | 0.05 | [StratPPI validation] | Table 4 row 23; section 8.2 | fails | 14 | 4,000 | 56,000 | 0.0755 (over) | 0.1482 [0.1374, 0.1596] | 12 | 2 | 0 | 11 |
| Clopper-Pearson over pairs | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 24; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.1534 (over) | 0.2682 [0.2546, 0.2823] | 27 | 0 | 1 | 26 |
| Clopper-Pearson over pairs | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 24; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.2319 (over) | 0.3397 [0.3251, 0.3547] | 28 | 0 | 0 | 27 |
| Clopper-Pearson over pairs | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | Table 4 row 24; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.3048 (over) | 0.3500 [0.3352, 0.3650] | 28 | 0 | 0 | 28 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 25; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.1064 (over) | 0.2577 [0.2443, 0.2716] | 21 | 0 | 7 | 20 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | Table 4 row 25; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.1522 (over) | 0.2745 [0.2607, 0.2886] | 24 | 0 | 4 | 24 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 26; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0065 (at or under delta) | 0.0285 [0.0236, 0.0341] | 0 | 0 | 28 | 0 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 26; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0335 (at or under delta) | 0.0658 [0.0583, 0.0739] | 4 | 0 | 24 | 3 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | Table 4 row 26; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0556 (over) | 0.0890 [0.0804, 0.0983] | 16 | 3 | 9 | 15 |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 27; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0318 (at or under delta) | 0.1745 [0.1629, 0.1866] | 5 | 0 | 23 | 5 |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | Table 4 row 27; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0655 (over) | 0.1777 [0.1660, 0.1900] | 14 | 1 | 13 | 10 |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | Table 4 row 27; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.1236 (over) | 0.2777 [0.2639, 0.2919] | 27 | 0 | 1 | 26 |
| stratified sheet read as an i.i.d. sample (PPI) | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | Table 4 row 28 | fails | 18 | 4,000 | 72,000 | 0.2364 (over) | 0.9892 [0.9855, 0.9922] | 7 | 0 | 11 | 7 |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | Table 4 row 29 | fails | 42 | 4,000 | 168,000 | 0.6353 (over) | 1 [0.9991, 1] | 30 | 0 | 12 | 30 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [two-way bounds] | Table 4 row 30; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0259 (at or under delta) | 0.1360 [0.1255, 0.1470] | 5 | 0 | 23 | 5 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [two-way bounds] | Table 4 row 30; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0557 (over) | 0.1747 [0.1631, 0.1869] | 8 | 3 | 17 | 7 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [two-way bounds] | Table 4 row 30; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.1031 (over) | 0.1787 [0.1670, 0.1910] | 26 | 1 | 1 | 25 |
| two clustered margins added in quadrature, fresh draws | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [two-way bounds] | Table 4 row 31; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0021 (at or under delta) | 0.0152 [0.0117, 0.0195] | 0 | 0 | 28 | 0 |
| two clustered margins added in quadrature, fresh draws | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [two-way bounds] | Table 4 row 31; section 10.2 | holds | 28 | 4,000 | 112,000 | 0.0175 (at or under delta) | 0.0457 [0.0395, 0.0527] | 0 | 0 | 28 | 0 |
| two clustered margins added in quadrature, fresh draws | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [two-way bounds] | Table 4 row 31; section 10.2 | fails | 28 | 4,000 | 112,000 | 0.0344 (at or under delta) | 0.0580 [0.0510, 0.0657] | 1 | 1 | 26 | 0 |
| stratified Wald-t `b1` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Table 4 row 32; section 8.1 | holds | 28 | 5,000 | 140,000 | 0.0179 (at or under delta) | 0.0364 [0.0314, 0.0420] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | Table 4 row 32; section 8.1 | holds | 28 | 5,000 | 140,000 | 0.0501 (at or under delta) | 0.0804 [0.0730, 0.0883] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | Table 4 row 33; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0235 (at or under delta) | 0.0376 [0.0358, 0.0395] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | Table 4 row 33; section 7.1 | holds | 28 | 40,000 | 1,120,000 | 0.0599 (at or under delta) | 0.0857 [0.0830, 0.0885] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | Table 4 row 34; section 8.1 (the prompt source) | holds | 28 | 10,000 | 280,000 | 0.0297 (at or under delta) | 0.0504 [0.0462, 0.0549] | 0 | 1 | 27 | 0 |
| stratified Wald-t `b1` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | Table 4 row 34; section 8.1 (the prompt source) | holds | 28 | 10,000 | 280,000 | 0.0683 (at or under delta) | 0.0948 [0.0891, 0.1007] | 0 | 0 | 28 | 0 |
| stratified Wald-t `b1` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.05 | [013] | Table 4 row 35; section 7.1 (reading 3) | fails | 18 | 4,000 | 72,000 | 0.0559 (over) | 0.1110 [0.1014, 0.1211] | 5 | 5 | 8 | 4 |
| stratified Wald-t `b1` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.1 | [013] | Table 4 row 35; section 7.1 (reading 3) | fails | 18 | 4,000 | 72,000 | 0.1050 (over) | 0.1470 [0.1362, 0.1584] | 5 | 5 | 8 | 4 |
| `b1w` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.05 | [013] | Table 4 row 36; section 7.1 (reading 3) | fails | 18 | 4,000 | 72,000 | 0.0405 (at or under delta) | 0.0573 [0.0503, 0.0649] | 1 | 2 | 15 | 0 |
| `b1w` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.1 | [013] | Table 4 row 36; section 7.1 (reading 3) | fails | 18 | 4,000 | 72,000 | 0.1024 (over) | 0.1350 [0.1246, 0.1460] | 4 | 5 | 9 | 2 |
| stratified Wald-t limit inside the training loop, 8 strata | synthetic bandit, Lagrangian GRPO, 4 environments, the pool's own rate | 0.05 | [O2] | Table 4 row 37; section 7.2 | - | 4 | 8,000 | 32,000 | 0.0397 (at or under delta) | 0.0521 [0.0474, 0.0572] | 0 | 1 | 3 | 0 |
| stratified Wald-t limit inside the training loop, 8 strata | synthetic bandit, Lagrangian GRPO, 4 environments, the pool's own rate | 0.1 | [O2] | Table 4 row 37; section 7.2 | - | 4 | 8,000 | 32,000 | 0.0842 (at or under delta) | 0.1003 [0.0938, 0.1070] | 0 | 1 | 3 | 0 |
| stratified Wald-t limit inside the training loop, 8 strata | synthetic bandit, Lagrangian GRPO, 4 environments, the population's rate, with the sampled-pool term | 0.05 | [O2] | Table 4 row 38; section 7.2 | - | 4 | 8,000 | 32,000 | 0.0512 (unresolved) | 0.0566 [0.0517, 0.0619] | 1 | 1 | 2 | 1 |
| stratified Wald-t limit inside the training loop, 8 strata | synthetic bandit, Lagrangian GRPO, 4 environments, the population's rate, with the sampled-pool term | 0.1 | [O2] | Table 4 row 38; section 7.2 | - | 4 | 8,000 | 32,000 | 0.0988 (at or under delta) | 0.1071 [0.1004, 0.1141] | 1 | 0 | 3 | 0 |
| stratified Wald-t limit, a costly label stratified on a cheap reference rate | XSTest human labels of five released models, 8 mid-rate labels, two stratifiers, n_s 50-100, redrawn with replacement | 0.05 | [O3] | Table 4 row 39; section 8.1.4 | - | 32 | 40,000 | 1,280,000 | 0.0323 (at or under delta) | 0.0525 [0.0503, 0.0547] | 1 | 0 | 31 | 0 |
| stratified Wald-t limit, a costly label stratified on a cheap reference rate | XSTest human labels of five released models, 8 mid-rate labels, two stratifiers, n_s 50-100, redrawn with replacement | 0.1 | [O3] | Table 4 row 39; section 8.1.4 | - | 32 | 40,000 | 1,280,000 | 0.0732 (at or under delta) | 0.1079 [0.1049, 0.1110] | 2 | 0 | 30 | 2 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | Table 4 row 40 | holds | 28 | 4,000 | 112,000 | 0.0031 (at or under delta) | 0.0070 [0.0047, 0.0101] | 0 | 0 | 28 | 0 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | Table 4 row 40 | holds | 28 | 4,000 | 112,000 | 0.0007 (at or under delta) | 0.0050 [0.0031, 0.0077] | 0 | 0 | 28 | 0 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | Table 4 row 40 | holds | 28 | 4,000 | 112,000 | 0.0019 (at or under delta) | 0.0065 [0.0043, 0.0095] | 0 | 0 | 28 | 0 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | Table 4 row 40 | holds | 28 | 4,000 | 112,000 | 0.0004 (at or under delta) | 0.0030 [0.0016, 0.0052] | 0 | 0 | 28 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | Table 4 row 41 | holds | 28 | 4,000 | 112,000 | 0.0213 (at or under delta) | 0.0423 [0.0362, 0.0490] | 0 | 0 | 28 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | Table 4 row 42 | fails | 28 | 4,000 | 112,000 | 0.0597 (over) | 0.2310 [0.2180, 0.2444] | 11 | 1 | 16 | 11 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | Table 4 row 42 | fails | 28 | 4,000 | 112,000 | 0.0977 (over) | 0.2487 [0.2354, 0.2625] | 19 | 2 | 7 | 18 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | Table 4 row 43 | - | 28 | 4,000 | 112,000 | 0.0012 (at or under delta) | 0.0160 [0.0123, 0.0204] | 0 | 0 | 28 | 0 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | Table 4 row 43 | - | 28 | 4,000 | 112,000 | 0.0050 (at or under delta) | 0.0375 [0.0318, 0.0439] | 0 | 0 | 28 | 0 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | Table 4 row 43 | - | 28 | 4,000 | 112,000 | 0.0120 (at or under delta) | 0.0395 [0.0337, 0.0460] | 0 | 0 | 28 | 0 |
| stratified Wald-t limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | Table 4 row 44 | holds | 4 | 40,000 | 160,000 | 0.0393 (at or under delta) | 0.0459 [0.0439, 0.0480] | 0 | 0 | 4 | 0 |
| stratified Wald-t limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | Table 4 row 44 | holds | 4 | 40,000 | 160,000 | 0.0847 (at or under delta) | 0.0919 [0.0891, 0.0948] | 0 | 0 | 4 | 0 |
| `b1w`, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | Table 4 row 45 | holds | 4 | 40,000 | 160,000 | 0.0369 (at or under delta) | 0.0408 [0.0389, 0.0428] | 0 | 0 | 4 | 0 |
| `b1w`, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | Table 4 row 45 | holds | 4 | 40,000 | 160,000 | 0.0888 (at or under delta) | 0.0919 [0.0891, 0.0948] | 0 | 0 | 4 | 0 |
| Wald-t b1, sampled-pool term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [O6] | Table 4 row 46 | - | 4 | 10,000 | 40,000 | 0.0436 (at or under delta) | 0.0513 [0.0471, 0.0558] | 0 | 1 | 3 | 0 |
| Wald-t b1, sampled-pool term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [O6] | Table 4 row 46 | - | 4 | 10,000 | 40,000 | 0.0938 (at or under delta) | 0.1114 [0.1053, 0.1177] | 1 | 0 | 3 | 1 |
| stratified Clopper-Pearson (pooled count plus the allocation's slack) | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.05 | [B25] | Table 4 row 47 | holds | 116 | exact enumeration | - | - | 0.0479 (exact) | 0 | 0 | 116 | 0 |
| stratified Clopper-Pearson (pooled count plus the allocation's slack) | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.1 | [B25] | Table 4 row 47 | holds | 116 | exact enumeration | - | - | 0.0934 (exact) | 0 | 0 | 116 | 0 |
| stratified betting bound (union-intersection) | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.05 | [B25] | Table 4 row 48 | holds | 116 | 4,000 | 464,000 | 0.0000 (at or under delta) | 0.0008 [0.0002, 0.0022] | 0 | 0 | 116 | 0 |
| stratified betting bound (union-intersection) | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.1 | [B25] | Table 4 row 48 | holds | 116 | 4,000 | 464,000 | 0.0002 (at or under delta) | 0.0073 [0.0049, 0.0104] | 0 | 0 | 116 | 0 |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | Table 4 row 49 | holds | 28 | 4,000 | 112,000 | 0.0233 (at or under delta) | 0.0442 [0.0381, 0.0511] | 0 | 0 | 28 | 0 |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | Table 4 row 50 | fails | 28 | 4,000 | 112,000 | 0.1027 (over) | 0.2467 [0.2335, 0.2604] | 18 | 3 | 7 | 17 |
| bet_a1@uavg, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | Table 4 row 51 | holds | 28 | 4,000 | 112,000 | 0.0007 (at or under delta) | 0.0045 [0.0027, 0.0071] | 0 | 0 | 28 | 0 |
| bet_a2, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | Table 4 row 51 | holds | 28 | 4,000 | 112,000 | 0.0003 (at or under delta) | 0.0032 [0.0017, 0.0056] | 0 | 0 | 28 | 0 |
| quad, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | Table 4 row 52 | holds | 28 | 4,000 | 112,000 | 0.0030 (at or under delta) | 0.0163 [0.0126, 0.0207] | 0 | 0 | 28 | 0 |
| quad, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | Table 4 row 52 | holds | 28 | 4,000 | 112,000 | 0.0166 (at or under delta) | 0.0473 [0.0409, 0.0543] | 0 | 0 | 28 | 0 |
| pig_t, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | Table 4 row 52 | holds | 28 | 4,000 | 112,000 | 0.0009 (at or under delta) | 0.0107 [0.0078, 0.0145] | 0 | 0 | 28 | 0 |
| pig_t, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | Table 4 row 52 | holds | 28 | 4,000 | 112,000 | 0.0111 (at or under delta) | 0.0367 [0.0311, 0.0431] | 0 | 0 | 28 | 0 |
| stratified Wald-t limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | Table 4 row 53 | holds | 4 | 40,000 | 160,000 | 0.0457 (at or under delta) | 0.0487 [0.0466, 0.0509] | 0 | 0 | 4 | 0 |
| stratified Wald-t limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | Table 4 row 53 | holds | 4 | 40,000 | 160,000 | 0.0979 (at or under delta) | 0.0999 [0.0970, 0.1029] | 0 | 0 | 4 | 0 |
| `b1w`, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | Table 4 row 54 | - | 4 | 40,000 | 160,000 | 0.0518 (over) | 0.0592 [0.0569, 0.0616] | 2 | 0 | 2 | 2 |
| `b1w`, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | Table 4 row 54 | - | 4 | 40,000 | 160,000 | 0.1041 (over) | 0.1121 [0.1090, 0.1152] | 2 | 1 | 1 | 2 |
| Wald-t b1, sampled-pool term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [B3] | Table 4 row 55 | holds | 4 | 10,000 | 40,000 | 0.0444 (at or under delta) | 0.0480 [0.0439, 0.0524] | 0 | 0 | 4 | 0 |
| Wald-t b1, sampled-pool term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [B3] | Table 4 row 55 | holds | 4 | 10,000 | 40,000 | 0.0891 (at or under delta) | 0.0943 [0.0886, 0.1002] | 0 | 0 | 4 | 0 |
| `b1w` at the other settings of the strata | 4 mid-rate labels, 1-8 reference samples, 2-8 strata (11 settings), n_s 100-200, 3 checkpoints | 0.05 | [013] | section 7.1 (reading 3) | - | 264 | 5,000 | 1,320,000 | 0.0309 (at or under delta) | 0.0598 [0.0534, 0.0667] | 2 | 14 | 248 | 0 |
| `b1w` at the other settings of the strata | 4 mid-rate labels, 1-8 reference samples, 2-8 strata (11 settings), n_s 100-200, 3 checkpoints | 0.1 | [013] | section 7.1 (reading 3) | - | 264 | 5,000 | 1,320,000 | 0.0748 (at or under delta) | 0.1166 [0.1078, 0.1258] | 6 | 13 | 245 | 1 |
| pooled Wilson bound on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.05 | [013] | section 7.1 (reading 3) | - | 18 | 4,000 | 72,000 | 0.0301 (at or under delta) | 0.0573 [0.0503, 0.0649] | 1 | 0 | 17 | 0 |
| pooled Wilson bound on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | 0.1 | [013] | section 7.1 (reading 3) | - | 18 | 4,000 | 72,000 | 0.0835 (at or under delta) | 0.1350 [0.1246, 0.1460] | 3 | 2 | 13 | 2 |
| `b1w`, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | - | 12 | 40,000 | 480,000 | 0.0095 (at or under delta) | 0.0555 [0.0533, 0.0578] | 1 | 0 | 11 | 1 |
| `b1w`, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | - | 12 | 40,000 | 480,000 | 0.0538 (at or under delta) | 0.1380 [0.1346, 0.1414] | 3 | 0 | 9 | 3 |
| pooled Wilson bound, redrawn with replacement | 2 rare labels (1-2%), random draws, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | - | 12 | 40,000 | 480,000 | 0.0109 (at or under delta) | 0.0636 [0.0612, 0.0660] | 1 | 0 | 11 | 1 |
| pooled Wilson bound, redrawn with replacement | 2 rare labels (1-2%), random draws, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | - | 12 | 40,000 | 480,000 | 0.0578 (at or under delta) | 0.1422 [0.1388, 0.1457] | 3 | 0 | 9 | 3 |
| Clopper-Pearson, redrawn with replacement | 2 rare labels (1-2%), random draws, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0056 (at or under delta) | 0.0471 [0.0450, 0.0492] | 0 | 0 | 12 | 0 |
| Clopper-Pearson, redrawn with replacement | 2 rare labels (1-2%), random draws, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0188 (at or under delta) | 0.0945 [0.0917, 0.0974] | 0 | 0 | 12 | 0 |
| stratified Wald-t `b1`, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0049 (at or under delta) | 0.0427 [0.0407, 0.0447] | 0 | 0 | 12 | 0 |
| stratified Wald-t `b1`, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0155 (at or under delta) | 0.0878 [0.0851, 0.0907] | 0 | 0 | 12 | 0 |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.0599 (over) | 0.0955 [0.0927, 0.0985] | 20 | 1 | 7 | 20 |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | fails | 28 | 40,000 | 1,120,000 | 0.1084 (over) | 0.1507 [0.1472, 0.1542] | 18 | 1 | 9 | 17 |
| StratPPI's normal limit, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | fails | 12 | 40,000 | 480,000 | 0.2374 (over) | 0.4747 [0.4698, 0.4796] | 12 | 0 | 0 | 12 |
| StratPPI's normal limit, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | fails | 12 | 40,000 | 480,000 | 0.2829 (over) | 0.4920 [0.4871, 0.4969] | 12 | 0 | 0 | 12 |
| StratPPI estimator with a bootstrap-t limit, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0093 (at or under delta) | 0.0155 [0.0144, 0.0168] | 0 | 0 | 12 | 0 |
| StratPPI estimator with a bootstrap-t limit, redrawn with replacement | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [replacement check] | section 7.1 | holds | 12 | 40,000 | 480,000 | 0.0238 (at or under delta) | 0.0432 [0.0413, 0.0453] | 0 | 0 | 12 | 0 |
| Seldonian pipeline (safety test after selection) | synthetic bandit, tables (a) and (b): four bounds, n 200-5,000 | 0.1 | [R 6.3] | section 7.2 | holds | 7 | not stated | | | 0.0020 | | | | |
| Seldonian pipeline (safety test after selection) | synthetic bandit, table (c): pressures 0-4, n 1,000 | 0.1 | [R 6.3] | section 7.2 | holds | 5 | 500 | 2,500 | ~0.0004 (at or under delta) | 0.0020 [0.0001, 0.0111] | 0 | 0 | 5 | 0 |
| unconstrained training (no test) | synthetic bandit, table (c): pressures 0-4 | 0.1 | [R 6.3] | section 7.2 | fails | 5 | 500 | 2,500 | ~0.7068 (over) | 1 [0.9926, 1] | 5 | 0 | 0 | 5 |
| Seldonian pipeline, judge-level violation given a solution | synthetic bandit, table (d): four judges | 0.1 | [R 6.3d] | section 7.2 | holds | 4 | not stated | | | 0.0030 | | | | |
| Clopper-Pearson safety test after an adversarial split | classic setup, four settings, 11 split rules; miss = passed and truly violating | 0.05 | [012] | section 7.2 | holds | 42 | 2,000 | 84,000 | 0.0000 (at or under delta) | 0.0005 [0.0000, 0.0028] | 0 | 0 | 42 | 0 |
| tight Wald test after an adversarial split | same runs; miss = true gap above the safety-set bound | 0.05 | [012] | section 7.2 | - | 42 | 2,000 | 84,000 | 0.0288 (at or under delta) | 0.0550 [0.0454, 0.0659] | 0 | 3 | 39 | 0 |
| `b1w` in the training loop, reference-rate strata | synthetic bandit, 4 heterogeneity levels, `SeldonianLLMPolicy` | 0.1 | [013 4] | section 7.2 | holds | 4 | 500 | 2,000 | 0.0960 (at or under delta) | 0.1160 [0.0893, 0.1474] | 0 | 1 | 3 | 0 |
| pooled `b1w` in the training loop, random split | same | 0.1 | [013 4] | section 7.2 (comparator) | holds | 4 | 500 | 2,000 | 0.0940 (at or under delta) | 0.1000 [0.0751, 0.1297] | 0 | 0 | 4 | 0 |
| trajectory certificate at delta / T | synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under its true rate | 0.1 | [SR 2.1] | section 7.2 | holds | 10 | 200 | 2,000 | 0.0495 (at or under delta) | 0.1000 [0.0622, 0.1502] | 0 | 0 | 10 | 0 |
| StratPPI estimator with a bootstrap-t limit | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 8.1 | holds | 28 | 5,000 | 140,000 | 0.0796 (at or under delta) | 0.0944 [0.0864, 0.1028] | 0 | 0 | 28 | 0 |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | section 8.1 | fails | 28 | 5,000 | 140,000 | 0.0947 (at or under delta) | 0.1438 [0.1342, 0.1538] | 7 | 3 | 18 | 5 |
| `b1w` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | section 8.1 | holds | 28 | 5,000 | 140,000 | 0.0749 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 28 | 0 |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | holds | 28 | 10,000 | 280,000 | 0.0381 (at or under delta) | 0.0525 [0.0482, 0.0571] | 0 | 4 | 24 | 0 |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | fails | 28 | 10,000 | 280,000 | 0.0725 (over) | 0.1701 [0.1628, 0.1776] | 17 | 1 | 10 | 16 |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | fails | 28 | 10,000 | 280,000 | 0.0777 (over) | 0.1663 [0.1591, 0.1737] | 21 | 2 | 5 | 20 |
| pooled Wilson bound, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 28 | 10,000 | 280,000 | 0.0475 (at or under delta) | 0.0703 [0.0654, 0.0755] | 7 | 0 | 21 | 4 |
| Clopper-Pearson, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | holds | 28 | 10,000 | 280,000 | 0.0414 (at or under delta) | 0.0500 [0.0458, 0.0545] | 0 | 0 | 28 | 0 |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 28 | 10,000 | 280,000 | 0.0855 (at or under delta) | 0.1116 [0.1055, 0.1179] | 1 | 0 | 27 | 1 |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | fails | 28 | 10,000 | 280,000 | 0.1291 (over) | 0.2271 [0.2189, 0.2354] | 18 | 5 | 5 | 17 |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | fails | 28 | 10,000 | 280,000 | 0.1339 (over) | 0.2303 [0.2221, 0.2387] | 23 | 1 | 4 | 20 |
| pooled Wilson bound, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 28 | 10,000 | 280,000 | 0.0967 (at or under delta) | 0.1287 [0.1222, 0.1354] | 6 | 8 | 14 | 5 |
| Clopper-Pearson, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | holds | 28 | 10,000 | 280,000 | 0.0776 (at or under delta) | 0.0967 [0.0910, 0.1027] | 0 | 0 | 28 | 0 |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0099 (at or under delta) | 0.0559 [0.0515, 0.0606] | 1 | 0 | 11 | 0 |
| stratified Wald-t `b1` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0053 (at or under delta) | 0.0454 [0.0414, 0.0497] | 0 | 0 | 12 | 0 |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0099 (at or under delta) | 0.0559 [0.0515, 0.0606] | 1 | 0 | 11 | 0 |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0028 (at or under delta) | 0.0073 [0.0057, 0.0092] | 0 | 0 | 12 | 0 |
| pooled Wilson bound, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0111 (at or under delta) | 0.0631 [0.0584, 0.0680] | 1 | 1 | 10 | 1 |
| Clopper-Pearson, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.05 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0058 (at or under delta) | 0.0514 [0.0472, 0.0559] | 0 | 1 | 11 | 0 |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0546 (at or under delta) | 0.1393 [0.1326, 0.1462] | 3 | 0 | 9 | 3 |
| stratified Wald-t `b1` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0152 (at or under delta) | 0.0807 [0.0754, 0.0862] | 0 | 0 | 12 | 0 |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0546 (at or under delta) | 0.1393 [0.1326, 0.1462] | 3 | 0 | 9 | 3 |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0086 (at or under delta) | 0.0193 [0.0167, 0.0222] | 0 | 0 | 12 | 0 |
| pooled Wilson bound, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0583 (at or under delta) | 0.1462 [0.1393, 0.1533] | 3 | 1 | 8 | 3 |
| Clopper-Pearson, random sample of the pool | a pool redrawn from its source, strata rebuilt, 5 rare-rate labels, n_s 100-200 | 0.1 | [two-phase check] | section 8.1 (the prompt source) | - | 12 | 10,000 | 120,000 | 0.0183 (at or under delta) | 0.0863 [0.0809, 0.0920] | 0 | 0 | 12 | 0 |
| StratPPI's normal limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | fails | 4 | 40,000 | 160,000 | 0.0743 (over) | 0.0819 [0.0793, 0.0847] | 4 | 0 | 0 | 4 |
| StratPPI's normal limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | fails | 4 | 40,000 | 160,000 | 0.1277 (over) | 0.1374 [0.1341, 0.1408] | 4 | 0 | 0 | 4 |
| pooled Wilson bound, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0450 (at or under delta) | 0.0539 [0.0517, 0.0562] | 1 | 0 | 3 | 1 |
| pooled Wilson bound, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.1101 (over) | 0.1164 [0.1133, 0.1196] | 3 | 0 | 1 | 3 |
| Clopper-Pearson, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0404 (at or under delta) | 0.0481 [0.0460, 0.0502] | 0 | 0 | 4 | 0 |
| Clopper-Pearson, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0719 (at or under delta) | 0.0783 [0.0756, 0.0809] | 0 | 0 | 4 | 0 |
| bootstrap-t StratPPI limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0466 (at or under delta) | 0.0490 [0.0469, 0.0511] | 0 | 0 | 4 | 0 |
| bootstrap-t StratPPI limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0970 (at or under delta) | 0.0984 [0.0955, 0.1014] | 0 | 0 | 4 | 0 |
| b1w, sampled-pool term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0414 (at or under delta) | 0.0479 [0.0438, 0.0523] | 0 | 0 | 4 | 0 |
| b1w, sampled-pool term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0957 (at or under delta) | 0.1041 [0.0982, 0.1103] | 0 | 1 | 3 | 0 |
| b1w, no term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0596 (over) | 0.0710 [0.0660, 0.0762] | 2 | 1 | 1 | 2 |
| b1w, no term | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1197 (over) | 0.1338 [0.1272, 0.1406] | 4 | 0 | 0 | 4 |
| StratPPI estimator, bootstrap-t | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0705 (over) | 0.0798 [0.0746, 0.0853] | 4 | 0 | 0 | 4 |
| StratPPI estimator, bootstrap-t | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1288 (over) | 0.1400 [0.1333, 0.1470] | 4 | 0 | 0 | 4 |
| pooled Wilson | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0431 (at or under delta) | 0.0542 [0.0498, 0.0588] | 0 | 1 | 3 | 0 |
| pooled Wilson | Qwen2.5-1.5B-Instruct, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [O6] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1071 (over) | 0.1152 [0.1090, 0.1216] | 2 | 1 | 1 | 2 |
| Clopper-Pearson on a random draw | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.05 | [B25] | section 8.1.4 | - | 116 | exact enumeration | - | - | 0.0500 (exact) | 0 | 0 | 116 | 0 |
| Clopper-Pearson on a random draw | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.1 | [B25] | section 8.1.4 | - | 116 | exact enumeration | - | - | 0.0990 (exact) | 0 | 0 | 116 | 0 |
| pooled Wilson bound on a random draw | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.05 | [B25] | section 8.1.4 | - | 116 | exact enumeration | - | - | 0.0793 (exact) | 27 | 0 | 89 | 27 |
| pooled Wilson bound on a random draw | 116 cells (real responses, a costly label, a synthetic grid), drawn with replacement within strata, n_s 50-400 | 0.1 | [B25] | section 8.1.4 | - | 116 | exact enumeration | - | - | 0.1432 (exact) | 41 | 0 | 75 | 41 |
| StratPPI's normal limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0593 (over) | 0.0707 [0.0682, 0.0732] | 3 | 1 | 0 | 3 |
| StratPPI's normal limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.1099 (over) | 0.1200 [0.1169, 0.1232] | 3 | 1 | 0 | 3 |
| pooled Wilson bound, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0486 (at or under delta) | 0.0563 [0.0540, 0.0586] | 2 | 0 | 2 | 1 |
| pooled Wilson bound, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.1001 (unresolved) | 0.1176 [0.1144, 0.1208] | 2 | 0 | 2 | 2 |
| Clopper-Pearson, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0406 (at or under delta) | 0.0447 [0.0427, 0.0467] | 0 | 0 | 4 | 0 |
| Clopper-Pearson, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0818 (at or under delta) | 0.0917 [0.0889, 0.0945] | 0 | 0 | 4 | 0 |
| bootstrap-t StratPPI limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0444 (at or under delta) | 0.0467 [0.0447, 0.0488] | 0 | 0 | 4 | 0 |
| bootstrap-t StratPPI limit, a policy moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label (26% and 58%), steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 40,000 | 160,000 | 0.0937 (at or under delta) | 0.0963 [0.0934, 0.0992] | 0 | 0 | 4 | 0 |
| b1w, sampled-pool term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0478 (at or under delta) | 0.0579 [0.0534, 0.0627] | 1 | 0 | 3 | 1 |
| b1w, sampled-pool term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1031 (over) | 0.1164 [0.1102, 0.1228] | 1 | 1 | 2 | 1 |
| b1w, no term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0599 (over) | 0.0674 [0.0626, 0.0725] | 3 | 1 | 0 | 3 |
| b1w, no term | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1145 (over) | 0.1209 [0.1146, 0.1275] | 4 | 0 | 0 | 4 |
| StratPPI estimator, bootstrap-t | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0541 (over) | 0.0624 [0.0577, 0.0673] | 1 | 2 | 1 | 1 |
| StratPPI estimator, bootstrap-t | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.1049 (over) | 0.1164 [0.1102, 0.1228] | 1 | 2 | 1 | 1 |
| pooled Wilson | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.05 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0490 (at or under delta) | 0.0576 [0.0531, 0.0623] | 1 | 0 | 3 | 1 |
| pooled Wilson | Qwen2.5-1.5B-Instruct trained on the guard's flag alone, refusal label, a claim about the prompt source, steps 100 and 200, n_s 100-200 | 0.1 | [B3] | section 8.1.3 | - | 4 | 10,000 | 40,000 | 0.0989 (at or under delta) | 0.1174 [0.1112, 0.1239] | 1 | 1 | 2 | 1 |
| Clopper-Pearson, labels alone | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 8.2 (the baseline) | holds | 14 | 4,000 | 56,000 | 0.0318 (at or under delta) | 0.0508 [0.0442, 0.0580] | 0 | 1 | 13 | 0 |
| PPI++ with a normal limit | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 8.2 | fails | 14 | 4,000 | 56,000 | 0.1023 (over) | 0.2320 [0.2190, 0.2454] | 14 | 0 | 0 | 14 |
| PPI++ with a bootstrap-t limit | refusal pools, random labelled subset, n 100-1,000 | 0.05 | [P14] | section 8.2 | holds | 14 | 4,000 | 56,000 | 0.0354 (at or under delta) | 0.0517 [0.0451, 0.0591] | 0 | 1 | 13 | 0 |
| `b1w` on judge-logit strata | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | 0.05 | [P14] | section 8.2 | holds | 28 | 4,000 | 112,000 | 0.0391 (at or under delta) | 0.0532 [0.0465, 0.0607] | 0 | 1 | 27 | 0 |
| carried Youden-corrected bound | source: xstest -> orbench, six judge wordings | 0.05 | [017 5] | section 9.1 | over for 4 of 6 | 6 | 4,000 | 24,000 | 0.4020 (over) | 0.9435 [0.9359, 0.9505] | 4 | 0 | 2 | 4 |
| carried Youden-corrected bound | source: orbench -> xstest, six judge wordings | 0.05 | [017 5] | section 9.1 | over for 1 of 6 | 6 | 4,000 | 24,000 | 0.0222 (at or under delta) | 0.1328 [0.1224, 0.1437] | 1 | 0 | 5 | 1 |
| carried Youden-corrected bound | pool: over-refusal -> harmful, six judge wordings | 0.05 | [017 5] | section 9.1 | over for none | 6 | 4,000 | 24,000 | 0.0000 (at or under delta) | 0.0003 [0.0000, 0.0014] | 0 | 0 | 6 | 0 |
| carried Youden-corrected bound | pool: harmful -> over-refusal, six judge wordings | 0.05 | [017 5] | section 9.1 | over for 5 of 6 | 6 | 4,000 | 24,000 | 0.4275 (over) | 0.9960 [0.9935, 0.9977] | 5 | 0 | 1 | 5 |
| carried Youden-corrected bound | training: step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 9.2 | carried (holds) | 6 | 4,000 | 24,000 | 0.0103 (at or under delta) | 0.0415 [0.0355, 0.0481] | 0 | 0 | 6 | 0 |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 100, six judge wordings | 0.05 | [017 E8] | section 9.2 | 1, 1 and 4 of 6 | 6 | 4,000 | 24,000 | 0.1047 (over) | 0.5135 [0.4979, 0.5291] | 1 | 0 | 5 | 1 |
| carried Youden-corrected bound | training, constrained (014): step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 9.2 | over for 3, 3 and 2 of 6 | 6 | 4,000 | 24,000 | 0.1968 (over) | 0.8027 [0.7901, 0.8150] | 3 | 1 | 2 | 3 |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | 0.05 | [017 E8] | section 9.2 | 1, 1 and 4 of 6 | 6 | 4,000 | 24,000 | 0.0551 (over) | 0.2707 [0.2570, 0.2848] | 1 | 0 | 5 | 1 |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 9.2 | over for 3, 3 and 2 of 6 | 6 | 4,000 | 24,000 | 0.1261 (over) | 0.3990 [0.3838, 0.4144] | 3 | 0 | 3 | 2 |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 100, six judge wordings | 0.05 | [017 E8] | section 9.2 | 1, 1 and 4 of 6 | 6 | 4,000 | 24,000 | 0.2134 (over) | 0.8285 [0.8165, 0.8401] | 4 | 0 | 2 | 4 |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 200, six judge wordings | 0.05 | [017 E8] | section 9.2 | over for 3, 3 and 2 of 6 | 6 | 4,000 | 24,000 | 0.1514 (over) | 0.6248 [0.6095, 0.6398] | 2 | 0 | 4 | 2 |
| block PPI, betting (finite-sample) | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | section 9.4 | holds | 84 | 500, 1,000 | 80,000 | 0.0174 (at or under delta) | 0.0440 [0.0321, 0.0586] | 0 | 0 | 84 | 0 |
| cluster bootstrap-t, by injection task | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0224 (at or under delta) | 0.1650 [0.1536, 0.1769] | 3 | 0 | 25 | 3 |
| cluster bootstrap-t, by injection task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0419 (at or under delta) | 0.0912 [0.0825, 0.1006] | 4 | 2 | 22 | 4 |
| cluster bootstrap-t, by injection task | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0827 (over) | 0.2060 [0.1936, 0.2189] | 23 | 2 | 3 | 21 |
| clustered bound of the larger intraclass correlation | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0267 (at or under delta) | 0.0503 [0.0437, 0.0575] | 0 | 1 | 27 | 0 |
| clustered bound of the larger intraclass correlation | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0815 (over) | 0.1837 [0.1719, 0.1961] | 20 | 0 | 8 | 18 |
| clustered bound of the larger intraclass correlation | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.1151 (over) | 0.1958 [0.1836, 0.2084] | 22 | 1 | 5 | 22 |
| two clustered margins added in quadrature | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0022 (at or under delta) | 0.0180 [0.0141, 0.0226] | 0 | 0 | 28 | 0 |
| two clustered margins added in quadrature | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0169 (at or under delta) | 0.0403 [0.0344, 0.0468] | 0 | 0 | 28 | 0 |
| two clustered margins added in quadrature | AgentDojo, 28 pipelines, scheme (c): both resampled | 0.05 | [AgentDojo recheck] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0327 (at or under delta) | 0.0522 [0.0456, 0.0596] | 0 | 1 | 27 | 0 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0190 (at or under delta) | 0.1013 [0.0921, 0.1110] | 4 | 0 | 24 | 4 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0393 (at or under delta) | 0.1310 [0.1207, 0.1419] | 7 | 1 | 20 | 6 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0132 (at or under delta) | 0.0818 [0.0734, 0.0907] | 3 | 0 | 25 | 3 |
| betting bound on user-task rates, against their average | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0253 (at or under delta) | 0.1025 [0.0933, 0.1123] | 4 | 1 | 23 | 3 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0015 (at or under delta) | 0.0177 [0.0139, 0.0223] | 0 | 0 | 28 | 0 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0104 (at or under delta) | 0.0770 [0.0689, 0.0857] | 2 | 0 | 26 | 1 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0253 (at or under delta) | 0.1037 [0.0945, 0.1136] | 3 | 3 | 22 | 2 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0006 (at or under delta) | 0.0092 [0.0065, 0.0127] | 0 | 0 | 28 | 0 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0069 (at or under delta) | 0.0638 [0.0564, 0.0718] | 1 | 0 | 27 | 1 |
| betting bound on user-task rates, against the pair rate | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0152 (at or under delta) | 0.0690 [0.0613, 0.0773] | 2 | 0 | 26 | 2 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0253 (at or under delta) | 0.1247 [0.1147, 0.1354] | 4 | 2 | 22 | 4 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0423 (at or under delta) | 0.1300 [0.1197, 0.1408] | 8 | 3 | 17 | 8 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0109 (at or under delta) | 0.0983 [0.0892, 0.1079] | 2 | 0 | 26 | 2 |
| betting bound on the pair rate | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0184 (at or under delta) | 0.1050 [0.0957, 0.1149] | 2 | 0 | 26 | 2 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0417 (at or under delta) | 0.0520 [0.0453, 0.0593] | 0 | 2 | 26 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.1079 (over) | 0.2567 [0.2433, 0.2706] | 21 | 0 | 7 | 21 |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.1521 (over) | 0.2730 [0.2592, 0.2871] | 24 | 0 | 4 | 23 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0026 (at or under delta) | 0.0170 [0.0132, 0.0215] | 0 | 0 | 28 | 0 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0174 (at or under delta) | 0.0460 [0.0397, 0.0530] | 0 | 0 | 28 | 0 |
| two clustered margins in quadrature | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0338 (at or under delta) | 0.0505 [0.0439, 0.0577] | 0 | 1 | 27 | 0 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0067 (at or under delta) | 0.0288 [0.0238, 0.0344] | 0 | 0 | 28 | 0 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0341 (at or under delta) | 0.0665 [0.0590, 0.0747] | 4 | 1 | 23 | 4 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0562 (over) | 0.0875 [0.0789, 0.0967] | 17 | 1 | 10 | 14 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0038 (at or under delta) | 0.0288 [0.0238, 0.0344] | 0 | 0 | 28 | 0 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0106 (at or under delta) | 0.0665 [0.0590, 0.0747] | 1 | 0 | 27 | 1 |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0243 (at or under delta) | 0.0620 [0.0547, 0.0699] | 1 | 1 | 26 | 1 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0010 (at or under delta) | 0.0065 [0.0043, 0.0095] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0120 (at or under delta) | 0.0300 [0.0249, 0.0358] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0245 (at or under delta) | 0.0408 [0.0348, 0.0473] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0003 (at or under delta) | 0.0025 [0.0012, 0.0046] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0020 (at or under delta) | 0.0123 [0.0091, 0.0162] | 0 | 0 | 28 | 0 |
| pigeonhole bootstrap-t | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0061 (at or under delta) | 0.0200 [0.0159, 0.0248] | 0 | 0 | 28 | 0 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (a) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0247 (at or under delta) | 0.1232 [0.1132, 0.1338] | 5 | 0 | 23 | 5 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (b) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0564 (over) | 0.1750 [0.1633, 0.1871] | 8 | 2 | 18 | 6 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (c) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.1035 (over) | 0.2023 [0.1899, 0.2150] | 25 | 2 | 1 | 24 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0200 (at or under delta) | 0.1217 [0.1118, 0.1323] | 5 | 0 | 23 | 4 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (b_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0253 (at or under delta) | 0.1545 [0.1434, 0.1661] | 4 | 0 | 24 | 4 |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (c_s) | 0.05 | [O4] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0593 (over) | 0.1730 [0.1614, 0.1851] | 12 | 2 | 14 | 12 |
| bet_a1, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0003 (at or under delta) | 0.0045 [0.0027, 0.0071] | 0 | 0 | 28 | 0 |
| bet_a1, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0168 (at or under delta) | 0.0850 [0.0765, 0.0941] | 2 | 0 | 26 | 1 |
| bet_a1@uavg, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0268 (at or under delta) | 0.1172 [0.1074, 0.1276] | 5 | 1 | 22 | 4 |
| bet_a2, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0185 (at or under delta) | 0.1245 [0.1144, 0.1351] | 3 | 0 | 25 | 2 |
| max, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0066 (at or under delta) | 0.0362 [0.0307, 0.0425] | 0 | 0 | 28 | 0 |
| max, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0286 (at or under delta) | 0.0655 [0.0580, 0.0736] | 3 | 2 | 23 | 2 |
| cgm_t, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_a) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0222 (at or under delta) | 0.1320 [0.1217, 0.1429] | 4 | 1 | 23 | 4 |
| cgm_t, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | 0.05 | [B6] | section 10.2 | - | 28 | 4,000 | 112,000 | 0.0575 (over) | 0.1427 [0.1320, 0.1540] | 13 | 1 | 14 | 13 |
| the four limits of the human-terms certificate | design check on synthetic labels, prompts as 'pool' | 0.05 | [P9 design check] | section 10.4 | holds | 4 | 2,000 | 8,000 | 0.0319 (at or under delta) | 0.0455 [0.0368, 0.0556] | 0 | 0 | 4 | 0 |
| pooled Wilson bound, random split | same labels and sizes | 0.05 | [013 H8] | Table A1 (the baseline) | - | 24 | 5,000 | 120,000 | 0.0307 (at or under delta) | 0.0566 [0.0504, 0.0634] | 1 | 0 | 23 | 0 |
| pooled Wilson bound, random split | same labels and sizes | 0.1 | [013 H8] | Table A1 (the baseline) | - | 24 | 5,000 | 120,000 | 0.0721 (at or under delta) | 0.1140 [0.1053, 0.1231] | 1 | 0 | 23 | 1 |
| StratPPI estimator with a bootstrap-t limit | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Appendix A (rare labels) | - | 12 | 5,000 | 60,000 | 0.0023 (at or under delta) | 0.0100 [0.0074, 0.0132] | 0 | 0 | 12 | 0 |
| StratPPI estimator with a bootstrap-t limit | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | Appendix A (rare labels) | - | 12 | 5,000 | 60,000 | 0.0084 (at or under delta) | 0.0176 [0.0141, 0.0216] | 0 | 0 | 12 | 0 |
| StratPPI as published (normal limit) | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Appendix A (rare labels) | fails | 12 | 5,000 | 60,000 | 0.2054 (over) | 0.4454 [0.4316, 0.4593] | 12 | 0 | 0 | 12 |
| StratPPI as published (normal limit) | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | Appendix A (rare labels) | fails | 12 | 5,000 | 60,000 | 0.2401 (over) | 0.4542 [0.4403, 0.4681] | 12 | 0 | 0 | 12 |
| `b1w` | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | Table A1 | holds | 28 | 5,000 | 140,000 | 0.0315 (at or under delta) | 0.0542 [0.0481, 0.0608] | 0 | 3 | 25 | 0 |
| `b1w` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | Appendix A (rare labels) | holds | 12 | 5,000 | 60,000 | 0.0065 (at or under delta) | 0.0380 [0.0329, 0.0437] | 0 | 0 | 12 | 0 |
| `b1w` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | Appendix A (rare labels) | - | 12 | 5,000 | 60,000 | 0.0440 (at or under delta) | 0.1260 [0.1169, 0.1355] | 1 | 0 | 11 | 1 |
| stratified Wald-t `b1` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | Appendix A (rare labels) | - | 12 | 5,000 | 60,000 | 0.0033 (at or under delta) | 0.0272 [0.0229, 0.0321] | 0 | 0 | 12 | 0 |
| stratified Wald-t `b1` | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | Appendix A (rare labels) | - | 12 | 5,000 | 60,000 | 0.0110 (at or under delta) | 0.0666 [0.0598, 0.0739] | 0 | 0 | 12 | 0 |
| the four limits of the human-terms certificate | design check on synthetic labels, prompts as 'new prompts' | 0.05 | [P9 design check] | Appendix C | - | 4 | 2,000 | 8,000 | 0.0418 (at or under delta) | 0.0630 [0.0527, 0.0746] | 1 | 2 | 1 | 1 |
| StratPPI estimator with a bootstrap-t limit, proportional allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | 0.05 | [StratPPI validation] | Appendix A | - | 26 | 5,000 | 130,000 | 0.0352 (at or under delta) | 0.0452 [0.0396, 0.0513] | 0 | 0 | 26 | 0 |

Bounds in the same files that the paper does not quote.

| bound | setting | delta | source | where in the paper | the paper says | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ttest` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 16 | 500 | 8,000 | 0.1045 (unresolved) | 0.1360 [0.1072, 0.1692] | 2 | 6 | 8 | 0 |
| `b1w_pooled` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.0942 (at or under delta) | 0.1160 [0.0893, 0.1474] | 0 | 6 | 6 | 0 |
| `b1w_strat_pool` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.0733 (at or under delta) | 0.1060 [0.0804, 0.1364] | 0 | 1 | 11 | 0 |
| `b1_strat_pop` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 12 | 500 | 6,000 | 0.1095 (over) | 0.1320 [0.1036, 0.1649] | 2 | 4 | 6 | 0 |
| `b1w_strat_pop` in the training loop, the other rule-by-bound cells | same runs | 0.1 | [013 4] | not quoted in the paper | - | 8 | 500 | 4,000 | 0.1030 (unresolved) | 0.1140 [0.0875, 0.1452] | 0 | 5 | 3 | 0 |
| per-check delta read as a trajectory claim | synthetic bandit, 5 arms, checks every 25 and 10 steps; miss = some check's bound under its true rate | 0.1 | [SR 2.1] | not quoted in the paper | - | 10 | 200 | 2,000 | 0.5025 (over) | 0.6500 [0.5795, 0.7159] | 10 | 0 | 0 | 10 |
| plain PPI, normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 168 | 4,000 | 672,000 | 0.0599 (over) | 0.1420 [0.1313, 0.1532] | 72 | 60 | 36 | 41 |
| PPI++, score (Wilson-type) limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 168 | 4,000 | 672,000 | 0.0657 (over) | 0.2208 [0.2080, 0.2339] | 74 | 29 | 65 | 53 |
| Youden correction from the same labels | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 4,000 | 168,000 | 0.0004 (at or under delta) | 0.0047 [0.0029, 0.0074] | 0 | 0 | 42 | 0 |
| PPI, three exact limits | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 4,000 | 168,000 | 0.0022 (at or under delta) | 0.0105 [0.0076, 0.0142] | 0 | 0 | 42 | 0 |
| post-stratified on the 0/1 judge, exact | spike 017 plasmodes, every cell and feature (the cells of its table B6) | 0.05 | [017 B6] | not quoted in the paper | - | 42 | 4,000 | 168,000 | 0.0028 (at or under delta) | 0.0110 [0.0080, 0.0147] | 0 | 0 | 42 | 0 |
| sheet labels read as i.i.d., Clopper-Pearson | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.0014 (at or under delta) | 0.0083 [0.0057, 0.0116] | 0 | 0 | 18 | 0 |
| design-weighted labels, normal limit | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.1323 (over) | 0.2160 [0.2033, 0.2291] | 18 | 0 | 0 | 18 |
| design-weighted PPI | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.0543 (over) | 0.0840 [0.0756, 0.0930] | 10 | 2 | 6 | 7 |
| design-weighted PPI++ | sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features | 0.05 | [017 4] | not quoted in the paper | - | 18 | 4,000 | 72,000 | 0.1324 (over) | 0.2160 [0.2033, 0.2291] | 18 | 0 | 0 | 18 |
| `b1w` on a design-weighted sheet, the two runs of each setting pooled | same sheets, 3 wordings x 3 planted rates | 0.05 | [017 4] | not quoted in the paper | - | 9 | 8,000 | 72,000 | 0.0232 (at or under delta) | 0.0540 [0.0491, 0.0592] | 0 | 1 | 8 | 0 |
| cluster bootstrap-t, by user task | AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check) | 0.05 | [020 P] | not quoted in the paper (replaced by the recheck on 28 pipelines) | - | 6 | 400 | 2,400 | 0.0400 (at or under delta) | 0.0600 [0.0388, 0.0880] | 0 | 2 | 4 | 0 |
| Clopper-Pearson over pairs | AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check) | 0.05 | [020 P] | not quoted in the paper (replaced by the recheck) | - | 6 | 400 | 2,400 | 0.1500 (over) | 0.2750 [0.2318, 0.3216] | 5 | 0 | 1 | 5 |
| two-way bootstrap | AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check) | 0.05 | [020 P] | not quoted in the paper (replaced by the recheck) | - | 6 | 400 | 2,400 | 0.0525 (unresolved) | 0.1700 [0.1345, 0.2105] | 2 | 0 | 4 | 2 |
| cluster bootstrap-t, by injection task | AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check) | 0.05 | [020 P] | not quoted in the paper (replaced by the recheck) | - | 6 | 400 | 2,400 | 0.0242 (at or under delta) | 0.1225 [0.0920, 0.1587] | 1 | 0 | 5 | 1 |
| Wilson bound over pairs | AgentDojo, 6 pipelines, user tasks resampled (spike 020's own check) | 0.05 | [020 P] | not quoted in the paper | - | 6 | 400 | 2,400 | 0.1512 (over) | 0.2750 [0.2318, 0.3216] | 5 | 0 | 1 | 5 |
| pooled Wilson bound, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0305 (at or under delta) | 0.0566 [0.0504, 0.0634] | 1 | 0 | 27 | 0 |
| pooled Wilson bound, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0076 (at or under delta) | 0.0478 [0.0421, 0.0541] | 0 | 0 | 12 | 0 |
| pooled Wilson bound, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0712 (at or under delta) | 0.1140 [0.1053, 0.1231] | 1 | 0 | 27 | 1 |
| pooled Wilson bound, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint (the draws of spikes 013 and 014) | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0467 (at or under delta) | 0.1258 [0.1167, 0.1353] | 1 | 1 | 10 | 1 |
| PPI++ with a normal limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0296 (at or under delta) | 0.0922 [0.0843, 0.1006] | 4 | 2 | 22 | 3 |
| PPI++ with a normal limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.2189 (over) | 0.4606 [0.4467, 0.4745] | 12 | 0 | 0 | 12 |
| PPI++ with a normal limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0633 (at or under delta) | 0.1524 [0.1425, 0.1627] | 3 | 3 | 22 | 3 |
| PPI++ with a normal limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.2672 (over) | 0.4782 [0.4643, 0.4922] | 12 | 0 | 0 | 12 |
| PPI++ with a bootstrap-t limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0161 (at or under delta) | 0.0346 [0.0297, 0.0400] | 0 | 0 | 28 | 0 |
| PPI++ with a bootstrap-t limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.05 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0023 (at or under delta) | 0.0098 [0.0073, 0.0129] | 0 | 0 | 12 | 0 |
| PPI++ with a bootstrap-t limit, random split | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 28 | 5,000 | 140,000 | 0.0470 (at or under delta) | 0.0842 [0.0766, 0.0922] | 0 | 0 | 28 | 0 |
| PPI++ with a bootstrap-t limit, random split | 2 rare labels (1-2%), reference-rate strata, n_s 100-200, by checkpoint | 0.1 | [P14] | results/paper/stratppi.md only | - | 12 | 5,000 | 60,000 | 0.0064 (at or under delta) | 0.0172 [0.0138, 0.0212] | 0 | 0 | 12 | 0 |

The same bound over all its settings at one delta (cells that repeat another row's draws left out).

| bound | delta | rows pooled | cells | draws per cell | total draws | pooled miss (its class) | largest miss [95% CP] | over | unresolved, above delta | at or under delta | over after Bonferroni |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clopper-Pearson | 0.1 | 1 | 20 | exact enumeration | - | - | 0.0959 (exact) | 0 | 0 | 20 | 0 |
| Clopper-Pearson | 0.05 | 2 | 56 | 4,000 | 224,000 | 0.0367 (at or under delta) | 0.0515 [0.0449, 0.0588] | 0 | 3 | 53 | 0 |
| b1w (Table 4 rows 6-8) | 0.05 | 3 | 44 | 4,000, 5,000 | 202,000 | 0.0291 (at or under delta) | 0.0595 [0.0524, 0.0673] | 1 | 4 | 39 | 0 |
| b1w (Table 4 rows 6-8) | 0.1 | 2 | 26 | 5,000 | 130,000 | 0.0756 (at or under delta) | 0.0970 [0.0889, 0.1055] | 0 | 0 | 26 | 0 |
| PPI++, bootstrap-t limit | 0.05 | 3 | 210 | 4,000, 5,000 | 868,000 | 0.0362 (at or under delta) | 0.0535 [0.0467, 0.0609] | 0 | 14 | 196 | 0 |
| StratPPI estimator, bootstrap-t limit | 0.05 | 2 | 56 | 4,000, 5,000 | 252,000 | 0.0368 (at or under delta) | 0.0560 [0.0491, 0.0636] | 0 | 5 | 51 | 0 |
| PPI++, normal limit | 0.05 | 3 | 210 | 4,000, 5,000 | 868,000 | 0.0720 (over) | 0.2412 [0.2281, 0.2548] | 144 | 31 | 35 | 103 |
| StratPPI, normal limit | 0.05 | 2 | 56 | 4,000, 5,000 | 252,000 | 0.0713 (over) | 0.2387 [0.2256, 0.2523] | 35 | 7 | 14 | 30 |
| StratPPI estimator, bootstrap-t limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0796 (at or under delta) | 0.0944 [0.0864, 0.1028] | 0 | 0 | 28 | 0 |
| StratPPI, normal limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0947 (at or under delta) | 0.1438 [0.1342, 0.1538] | 7 | 3 | 18 | 5 |
| PPI++, normal limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0633 (at or under delta) | 0.1524 [0.1425, 0.1627] | 3 | 3 | 22 | 3 |
| PPI++, bootstrap-t limit | 0.1 | 1 | 28 | 5,000 | 140,000 | 0.0470 (at or under delta) | 0.0842 [0.0766, 0.0922] | 0 | 0 | 28 | 0 |

Bonferroni by exact p-values (p < a / C) in place of the widened band gives the same count in every row except: PPI++ with a normal limit, spike 017 plasmodes (86 against 87); plain PPI, normal limit, spike 017 plasmodes (38 against 41); Clopper-Pearson over pairs, AgentDojo (4 against 5); Wilson bound over pairs, AgentDojo (4 against 5); StratPPI as published (normal limit), 5 mid-rate labels (6 against 7); StratPPI as published (normal limit), judge-logit strata (5 and 10) on the refusal pools (24 against 25); two-way bootstrap, AgentDojo (9 against 10); multiway cluster variance with a t quantile, AgentDojo (6 against 7); betting bound on user-task rates, against their average, AgentDojo (3 against 4); betting bound on the pair rate, AgentDojo (7 against 8); larger of the two clustered bounds, AgentDojo (3 against 4); cgm_t, a fitted crossed model, AgentDojo (12 against 13).

## (b) Table 4 of the draft against the files

Every printed miss rate agrees with the recount to its printed digits and every count of cells is equal (asserted). Bold and the `kind` column are the paper's and are not derived from the counts.

| row | bound | setting | delta | miss, as printed | miss, recounted | cells | draws per cell | over / unresolved / at or under | largest 95% upper limit of a cell's miss (over after Bonferroni) | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Clopper-Pearson | real harm labels, trained-policy pool, 121-124 positives of 3,600, n 200-2,400 | 0.10 | 0.066-0.096 | 0.0658-0.0959 | 20 | exact enumeration | 0 / 0 / 20 | 0.096 (0) | [binomial rows] |
| 2 | Clopper-Pearson | spike 017 plasmodes, every cell | 0.05 | 0-0.052 | 0-0.0515 | 42 | 4,000 | 0 / 2 / 40 | 0.059 (0) | [017 B6] |
| 3 | betting mixture | same pool as row 1 | 0.10 | 0.007-0.018 | 0.0073-0.0184 | 20 | exact enumeration | 0 / 0 / 20 | 0.018 (0) | [binomial rows] |
| 4 | Bentkus | same pool | 0.10 | 0.017-0.037 | 0.0172-0.0367 | 20 | exact enumeration | 0 / 0 / 20 | 0.037 (0) | [binomial rows] |
| 5 | Hoeffding, Anderson | same pool | 0.10 | 0.000 | 0-0.0000 | 40 | exact enumeration | 0 / 0 / 40 | 0.000 (0) | [binomial rows] |
| 6 | stratified Wilson-type `b1w` | 4 mid-rate labels (9-95%), real Granite-3.3-2B responses, n_s 100-200, 3 checkpoints | 0.05 / 0.10 | 0.017-0.054 / 0.058-0.097 | 0.0166-0.0542; 0.0576-0.0970 | 24; 24 | 5,000 | 0 / 3 / 21; 0 / 0 / 24 | 0.061 (0); 0.106 (0) | [013 H8] |
| 7 | `b1w` | label pushed by the Lagrangian, n_s 200 | 0.05 / 0.10 | 0.011-0.023 / 0.040-0.064 | 0.0110-0.0228; 0.0398-0.0642 | 2; 2 | 5,000 | 0 / 0 / 2; 0 / 0 / 2 | 0.027 (0); 0.071 (0) | [014] |
| 8 | `b1w` on a design-weighted sheet | sheets re-drawn by their real sampling rule | 0.05 | 0-0.060 | 0-0.0595 | 18 | 4,000 | 1 / 1 / 16 | 0.067 (0) | [017 4] |
| 9 | PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature | 0.05 | 0-0.054 | 0-0.0535 | 168 | 4,000 | 0 / 13 / 155 | 0.061 (0) | [017 B6] |
| 10 | cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a) | 0.05 | 0.013-0.051 | 0.0130-0.0505 | 28 | 4,000 | 0 / 2 / 26 | 0.058 (0) | [AgentDojo recheck] |
| 11 | StratPPI estimator with a bootstrap-t limit, proportional allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200; judge-logit strata, n 100-1,000 | 0.05 | 0.024-0.045; 0.001-0.056 | 0.0236-0.0452; 0.0008-0.0560 | 28; 28 | 5,000; 4,000 | 0 / 0 / 28; 0 / 5 / 23 | 0.051 (0); 0.064 (0) | [P14] |
| 12 | the same estimator and limit, redrawn with replacement | reference-rate strata, 5 mid-rate labels, n_s 100-200 | 0.05 / 0.10 | 0.030-0.050 / 0.069-0.100 | 0.0295-0.0496; 0.0694-0.1003 | 28; 28 | 40,000 | 0 / 0 / 28; 0 / 1 / 27 | 0.052 (0); 0.103 (0) | [replacement check] |
| 13 | Clopper-Pearson, random draws with replacement (the control) | the same 5 labels and sizes | 0.05 / 0.10 | 0.029-0.050 / 0.050-0.097 | 0.0293-0.0497; 0.0497-0.0970 | 28; 28 | 40,000 | 0 / 0 / 28; 0 / 0 / 28 | 0.052 (0); 0.100 (0) | [replacement check] |
| 14 | pigeonhole bootstrap-t | AgentDojo, 28 pipelines, schemes (a); (b); (c) | 0.05 | at most 0.004; 0.037; 0.042 | 0-0.0043; 0-0.0370; 0-0.0418 | 28; 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28 | 0.007 (0); 0.043 (0); 0.048 (0) | [two-way bounds] |
| 15 | **Student-t** | same pool as row 1, n 200-2,400 | 0.10 | **0.104-0.195** | 0.1037-0.1951 | 20 | exact enumeration | 20 / 0 / 0 | 0.195 (20) | [binomial rows] |
| 16 | **`b1w` and the pooled Wilson bound at rare rates** | harm labels at 1-2%, n_s 100-200, 3 checkpoints | 0.05 / 0.10 | 0-0.048 / **0-0.126** | 0-0.0478; 0-0.1260 | 24; 24 | 5,000 | 0 / 0 / 24; 2 / 1 / 21 | 0.054 (0); 0.136 (2) | [013 H8] |
| 17 | **`b1w`, redrawn with replacement** | reference-rate strata, 5 mid-rate labels, n_s 100-200 | 0.05 / 0.10 | **0.017-0.062 / 0.056-0.113** | 0.0171-0.0624; 0.0563-0.1128 | 28; 28 | 40,000 | 7 / 1 / 20; 6 / 2 / 20 | 0.065 (7); 0.116 (4) | [replacement check] |
| 18 | **pooled Wilson bound, random draws with replacement** | the same 5 labels and sizes | 0.05 / 0.10 | **0.031-0.069 / 0.069-0.128** | 0.0312-0.0689; 0.0686-0.1275 | 28; 28 | 40,000 | 6 / 0 / 22; 10 / 3 / 15 | 0.071 (6); 0.131 (10) | [replacement check] |
| 19 | **PPI++ with a normal limit** | spike 017 plasmodes | 0.05 | **0.041-0.241** | 0.0408-0.2412 | 168 | 4,000 | 126 / 29 / 13 | 0.255 (87) | [017 B6] |
| 20 | **StratPPI's normal limit, proportional allocation** | reference-rate strata; judge-logit strata | 0.05 | **0.024-0.086; 0.054-0.239** | 0.0238-0.0862; 0.0535-0.2387 | 28; 28 | 5,000; 4,000 | 9 / 5 / 14; 26 / 2 / 0 | 0.094 (7); 0.252 (25) | [P14] |
| 21 | **StratPPI's normal limit, the paper's allocations (oracle; heuristic)** | reference-rate strata; judge-logit strata | 0.05 | **up to 0.110; 0.217; up to 0.219; 0.90** | 0.0094-0.1098; 0.0022-0.2172; 0.0580-0.2188; 0.0350-0.9025 | 26; 26; 28; 28 | 5,000; 4,000 | 9 / 3 / 14; 11 / 0 / 15; 28 / 0 / 0; 24 / 1 / 3 | 0.119 (8); 0.229 (11); 0.232 (25); 0.912 (24) | [StratPPI validation] |
| 22 | **StratPPI estimator with a bootstrap-t limit, oracle or heuristic allocation** | the same cells | 0.05 | **up to 0.078; 0.174; up to 0.083; 0.90** | 0.0074-0.0782; 0.0064-0.1738; 0.0348-0.0830; 0.0067-0.9012 | 26; 26; 28; 28 | 5,000; 4,000 | 6 / 3 / 17; 11 / 1 / 14; 14 / 11 / 3; 14 / 0 / 14 | 0.086 (5); 0.185 (11); 0.092 (9); 0.910 (14) | [StratPPI validation] |
| 23 | **PPBoot, percentile limit (basic; power-tuned)** | judge-logit cells, unstratified | 0.05 | **up to 0.133; 0.148** | 0.0498-0.1328; 0.0542-0.1482 | 14; 14 | 4,000 | 7 / 6 / 1; 12 / 2 / 0 | 0.144 (5); 0.160 (11) | [StratPPI validation] |
| 24 | **Clopper-Pearson over pairs** | AgentDojo, 28 pipelines, schemes (a); (b); (c) | 0.05 | **0.042-0.268; 0.060-0.340; 0.195-0.350** | 0.0418-0.2682; 0.0600-0.3397; 0.1948-0.3500 | 28; 28; 28 | 4,000 | 27 / 0 / 1; 28 / 0 / 0; 28 / 0 / 0 | 0.282 (26); 0.355 (27); 0.365 (28) | [AgentDojo recheck] |
| 25 | **cluster bootstrap-t by user task, when injection tasks are sampled** | AgentDojo, schemes (b); (c) | 0.05 | **up to 0.258; 0.275** | 0.0008-0.2577; 0.0140-0.2745 | 28; 28 | 4,000 | 21 / 0 / 7; 24 / 0 / 4 | 0.272 (20); 0.289 (24) | [AgentDojo recheck] |
| 26 | **larger of the two clustered bounds** | AgentDojo, schemes (a); (b); (c) | 0.05 | at most 0.029; **up to 0.066; 0.089** | 0-0.0285; 0-0.0658; 0.0005-0.0890 | 28; 28; 28 | 4,000 | 0 / 0 / 28; 4 / 0 / 24; 16 / 3 / 9 | 0.034 (0); 0.074 (3); 0.098 (15) | [AgentDojo recheck] |
| 27 | **pigeonhole bootstrap, basic limit** | AgentDojo, schemes (a); (b); (c) | 0.05 | **up to 0.174; 0.178; 0.278** | 0-0.1745; 0.0220-0.1777; 0.0485-0.2777 | 28; 28; 28 | 4,000 | 5 / 0 / 23; 14 / 1 / 13; 27 / 0 / 1 | 0.187 (5); 0.190 (10); 0.292 (26) | [AgentDojo recheck] |
| 28 | **PPI on a stratified sheet read as an i.i.d. sample** | three judge wordings, labels planted at three rates | 0.05 | **0-0.989** | 0-0.9892 | 18 | 4,000 | 7 / 0 / 11 | 0.992 (7) | [017 4] |
| 29 | **the judge's rate alone** | spike 017 plasmodes | 0.05 | **0-1.000** | 0-1 | 42 | 4,000 | 30 / 0 / 12 | 1.000 (30) | [017 B6] |
| 30 | **multiway cluster variance with a t quantile** | AgentDojo, schemes (a); (b); (c) | 0.05 | **up to 0.136; 0.175; 0.179** | 0-0.1360; 0.0155-0.1747; 0.0403-0.1787 | 28; 28; 28 | 4,000 | 5 / 0 / 23; 8 / 3 / 17; 26 / 1 / 1 | 0.147 (5); 0.187 (7); 0.191 (25) | [two-way bounds] |
| 31 | **two clustered margins added in quadrature** | AgentDojo, schemes (a); (b); (c) | 0.05 | at most 0.015; 0.046; **0.058** | 0-0.0152; 0-0.0457; 0-0.0580 | 28; 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28; 1 / 1 / 26 | 0.020 (0); 0.053 (0); 0.066 (0) | [two-way bounds] |
| 32 | stratified Wald-t limit, 8 reference-rate strata | 5 mid-rate labels, real Granite-3.3-2B responses, n_s 100-200, the draws of the StratPPI row | 0.05 / 0.10 | 0.007-0.036 / 0.027-0.080 | 0.0074-0.0364; 0.0274-0.0804 | 28; 28 | 5,000 | 0 / 0 / 28; 0 / 0 / 28 | 0.042 (0); 0.088 (0) | [P14] |
| 33 | the same limit, redrawn with replacement | the same 5 labels and sizes | 0.05 / 0.10 | 0.007-0.038 / 0.026-0.086 | 0.0066-0.0376; 0.0257-0.0857 | 28; 28 | 40,000 | 0 / 0 / 28; 0 / 0 / 28 | 0.040 (0); 0.088 (0) | [replacement check] |
| 34 | the same limit with the term for a sampled pool | a claim about the prompt source: the pool redrawn and the strata rebuilt, the same 5 labels and sizes | 0.05 / 0.10 | 0.007-0.050 / 0.031-0.095 | 0.0071-0.0504; 0.0310-0.0948 | 28; 28 | 10,000 | 0 / 1 / 27; 0 / 0 / 28 | 0.055 (0); 0.101 (0) | [two-phase check] |
| 35 | **stratified Wald-t limit on a synthetic i.i.d. grid** | six strata profiles, n_s 100-400, binomial strata | 0.05 / 0.10 | **0-0.111 / 0.086-0.147** | 0-0.1110; 0.0855-0.1470 | 18; 18 | 4,000 | 5 / 5 / 8; 5 / 5 / 8 | 0.121 (4); 0.158 (4) | [013] |
| 36 | **`b1w` on the same grid** | the same profiles and sizes | 0.05 / 0.10 | **0-0.057 / 0.086-0.135** | 0-0.0573; 0.0855-0.1350 | 18; 18 | 4,000 | 1 / 2 / 15; 4 / 5 / 9 | 0.065 (0); 0.146 (2) | [013] |
| 37 | stratified Wald-t limit inside the training loop, 8 strata | synthetic bandit, Lagrangian GRPO, 4 environments, the pool's own rate, 8,000 runs a cell | 0.05 / 0.10 | 0.029-0.052 / 0.072-0.100 | 0.0289-0.0521; 0.0715-0.1003 | 4; 4 | 8,000 | 0 / 1 / 3; 0 / 1 / 3 | 0.057 (0); 0.107 (0) | [O2] |
| 38 | the same limit with the sampled-pool term | the same runs, the population's rate | 0.05 / 0.10 | 0.047-0.057 / 0.093-0.107 | 0.0465-0.0566; 0.0930-0.1071 | 4; 4 | 8,000 | 1 / 1 / 2; 1 / 0 / 3 | 0.062 (1); 0.114 (0) | [O2] |
| 39 | stratified Wald-t limit, a costly label stratified on a cheap reference rate | XSTest human labels of five released models, 8 mid-rate labels, two stratifiers, n_s 50-100, redrawn with replacement | 0.05 / 0.10 | 0-0.053 / 0.021-0.108 | 0-0.0525; 0.0210-0.1079 | 32; 32 | 40,000 | 1 / 0 / 31; 2 / 0 / 30 | 0.055 (0); 0.111 (2) | [O3] |
| 40 | betting mixture on user-task rates; on the pair rate | AgentDojo, 28 pipelines, user tasks redrawn across the table (a) and within suites (a_s); truth: the average over user tasks; the pair rate | 0.05 | at most 0.007; 0.005; 0.006; 0.003 | 0-0.0070; 0-0.0050; 0-0.0065; 0-0.0030 | 28; 28; 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28 | 0.010 (0); 0.008 (0); 0.010 (0); 0.005 (0) | [O4] |
| 41 | cluster bootstrap-t, by user task, redrawn within suites | AgentDojo, 28 pipelines, scheme (a_s) | 0.05 | 0.003-0.042 | 0.0027-0.0423 | 28 | 4,000 | 0 / 0 / 28 | 0.049 (0) | [O4] |
| 42 | **cluster bootstrap-t by user task, injection tasks redrawn within suites** | AgentDojo, schemes (b_s); (c_s) | 0.05 | **up to 0.231; 0.249** | 0-0.2310; 0.0077-0.2487 | 28; 28 | 4,000 | 11 / 1 / 16; 19 / 2 / 7 | 0.244 (11); 0.262 (18) | [O4] |
| 43 | two clustered margins in quadrature, redrawn within suites | AgentDojo, schemes (a_s); (b_s); (c_s) | 0.05 | at most 0.016; 0.037; 0.040 | 0-0.0160; 0-0.0375; 0-0.0395 | 28; 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28 | 0.020 (0); 0.044 (0); 0.046 (0) | [O4] |
| 44 | stratified Wald-t limit, a second model family, redrawn with replacement | Qwen2.5-1.5B-Instruct, refusal label (18-19%), 8 reference-rate strata, steps 100 and 200, n_s 100-200 | 0.05 / 0.10 | 0.033-0.046 / 0.077-0.092 | 0.0329-0.0459; 0.0772-0.0919 | 4; 4 | 40,000 | 0 / 0 / 4; 0 / 0 / 4 | 0.048 (0); 0.095 (0) | [O6] |
| 45 | `b1w`, the same | the same cells | 0.05 / 0.10 | 0.033-0.041 / 0.086-0.092 | 0.0329-0.0408; 0.0861-0.0919 | 4; 4 | 40,000 | 0 / 0 / 4; 0 / 0 / 4 | 0.043 (0); 0.095 (0) | [O6] |
| 46 | the stratified Wald-t limit with the term for a sampled pool, a second model family | a claim about the prompt source, the same label, steps and sizes | 0.05 / 0.10 | 0.036-0.051 / 0.085-0.111 | 0.0357-0.0513; 0.0850-0.1114 | 4; 4 | 10,000 | 0 / 1 / 3; 1 / 0 / 3 | 0.056 (0); 0.118 (1) | [O6] |
| 47 | stratified Clopper-Pearson: a stratified draw's pooled count plus the allocation's slack | 116 cells: real responses (58), a costly label (40), a synthetic grid (18), drawn with replacement within strata, n_s 50-400 | 0.05 / 0.10 | 0-0.048 / 0-0.093 | 0-0.0479; 0-0.0934 | 116; 116 | exact enumeration | 0 / 0 / 116; 0 / 0 / 116 | 0.048 (0); 0.093 (0) | [B25] |
| 48 | stratified betting bound (union-intersection) | the same cells | 0.05 / 0.10 | 0-0.001 / 0-0.007 | 0-0.0008; 0-0.0073 | 116; 116 | 4,000 | 0 / 0 / 116; 0 / 0 / 116 | 0.002 (0); 0.010 (0) | [B25] |
| 49 | cluster bootstrap-t, by user task, against a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model with injection tasks fixed (P_a); truth: the model's rate | 0.05 | 0.001-0.044 | 0.0008-0.0442 | 28 | 4,000 | 0 / 0 / 28 | 0.051 (0) | [B6] |
| 50 | **the same, both kinds of task drawn from the model** | the same, scheme (P_c) | 0.05 | **up to 0.247** | 0.0175-0.2467 | 28 | 4,000 | 18 / 3 / 7 | 0.260 (17) | [B6] |
| 51 | betting mixture on user-task rates; on the pair rate, against a fitted crossed model | the same, scheme (P_a); truth: the model's user-task average; its pair rate | 0.05 | at most 0.005; 0.003 | 0-0.0045; 0-0.0032 | 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28 | 0.007 (0); 0.006 (0) | [B6] |
| 52 | pigeonhole bootstrap-t; two clustered margins in quadrature, against a fitted crossed model | the same, schemes (P_a); (P_c) for each | 0.05 | at most 0.011; 0.037; 0.016; 0.047 | 0-0.0107; 0-0.0367; 0-0.0163; 0.0008-0.0473 | 28; 28; 28; 28 | 4,000 | 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28; 0 / 0 / 28 | 0.014 (0); 0.043 (0); 0.021 (0); 0.054 (0) | [B6] |
| 53 | stratified Wald-t limit, a policy that training moved far, redrawn with replacement | Qwen2.5-1.5B-Instruct trained on the guard's refusal flag alone, refusal label (26% at step 100, 58% at step 200), the second family's pool and strata, n_s 100-200 | 0.05 / 0.10 | 0.043-0.049 / 0.095-0.100 | 0.0430-0.0487; 0.0952-0.0999 | 4; 4 | 40,000 | 0 / 0 / 4; 0 / 0 / 4 | 0.051 (0); 0.103 (0) | [B3] |
| 54 | `b1w`, the same | the same cells | 0.05 / 0.10 | 0.045-0.059 / 0.098-0.112 | 0.0453-0.0592; 0.0982-0.1121 | 4; 4 | 40,000 | 2 / 0 / 2; 2 / 1 / 1 | 0.062 (2); 0.115 (2) | [B3] |
| 55 | the stratified Wald-t limit with the term for a sampled pool, a policy that training moved far | a claim about the prompt source, the same label, steps and sizes | 0.05 / 0.10 | 0.040-0.048 / 0.085-0.094 | 0.0398-0.0480; 0.0851-0.0943 | 4; 4 | 10,000 | 0 / 0 / 4; 0 / 0 / 4 | 0.052 (0); 0.100 (0) | [B3] |

- Rows 1, 3, 4, 5 and 15: exact enumeration at every pool count the training paper's printed rate allows (121-124 of 3,600) and the five sizes, 20 cells a bound; row 5 holds two bounds. Row 15, Student-t: over in every cell, 0.1037 to 0.1951.
- Row 6 by the largest miss over three checkpoints, the cell of earlier drafts: 8 such cells, 0 over, 0.0234-0.0542 at delta 0.05 and 0.0694-0.0970 at 0.10.
- Rows 12, 13, 17 and 18 are the cells of rows 6, 7 and 11 redrawn with replacement (caution 4). Row 13 is the control: Clopper-Pearson is exact on those draws, so its unresolved cells are Monte Carlo noise.

## (c) Sentences of the draft that state a count, against the files

Each quotation is in the draft word for word and each count is asserted in the script.

1. **section 7.1, below Table 4.**
   > (rows 38, 39, 46 and 54, with 2, 3, 1 and 4 cells over, of which 1, 2, 1 and 4 stay over after the
correction; interval ends up to 0.062 at delta 0.05 and 0.118 at 0.10)

   Registered rows with cells over (cells over, after Bonferroni): {38: (2, 1), 39: (3, 2), 46: (1, 1), 54: (4, 4)}; interval ends {0.05: 0.061915859142313484, 0.1: 0.11773008335265213}.

2. **section 7.1, below Table 4; section 11.**
   > In the remaining rows, which we treat as valid, no cell stays over its level after the correction, and the largest miss rate a cell's interval leaves open is 0.067 at delta 0.05 and 0.106 at 0.10.
   > One cell there is over under the uncorrected rule: `b1w` on a design-weighted sheet, 0.0595 at 4,000 draws.
   > the other run of the same setting gave 0.0485, and the two together give 0.054, which is unresolved
   > for the bounds we use its largest value is 0.067 at a nominal 0.05

   0.0595 (238 of 4,000; p = 0.0039; 95% interval 0.0524-0.0673), against a band that ends at 0.0569; after Bonferroni for 18 cells the band ends at 0.0604. The second run gave 0.0485; the two together 0.0540 (432 of 8,000; p = 0.054; 95% interval 0.0491-0.0592). No other cell of a Table 4 row the draft calls valid is over (1768 cells in 56 groups).

3. **section 7.1, below Table 4.**
   > Every row in bold has at least one cell over, and every one but the quadrature bound keeps one after the correction.
   > Most also have cells that are not over

   43 groups in the failing rows; each has a cell over (the rare-label row's two bounds taken together). 39 of them also have cells that are not over.

4. **section 7.1, reading 1.**
   > neither `b1w` nor the pooled Wilson bound is over in any of the 24 rare-label cells at delta 0.05; at delta 0.10 two are, both at 0.126, for one harm label at the checkpoint where its rate is 1.9%, at n_s 100

   Delta 0.05: 0 over, 0 unresolved, 24 at or under of 24 (`b1w` 0-0.0380, pooled 0-0.0478). Delta 0.10: 2 over, 1 unresolved, 21 at or under; the cells over are C2:gated, n_s 100, step200 0.1260; C2:gated, n_s 100, step200 0.1258. Before the correction of 2026-10-06 the same cells at delta 0.05 and n_s 100 read 0.076-0.448, the chance of drawing no positive.

5. **section 7.1, redrawn with replacement; section 11; abstract.**
   > `b1w` is over its level in 7 of 28 mid-rate cells at delta 0.05 (largest miss 0.062), all on the two labels at rates of 65% and above, and in none of the 16 cells at rates of 9-18% (largest 0.034)
   > The pooled Wilson bound is over in 6 of 28 (largest 0.069), two of them at rates of 15-18%
   > Each of the two is over in 1 of the 12 rare-label cells at delta 0.05, the same harm label at a checkpoint where its rate is 1.4%, at n_s 200 (0.055 and 0.064), and in 3 of 12 at delta 0.10 (up to 0.138 and 0.142)
   > Clopper-Pearson, exact on those draws, is over or unresolved in none
   > redrawn with replacement, `b1w` is over its level in 7 of 28 cells

   The 40,000-draw pass. `b1w`, delta 0.05: 7 / 1 / 20 of 28, 0.0171-0.0624; the cells over have true rates 0.654-0.954; the 16 cells under one half (0.093-0.185) are all at or under delta, largest 0.0337. Pooled Wilson: 6 / 0 / 22 of 28; its cells over have rates 0.149, 0.185, 0.660, 0.660, 0.930, 0.933. Rare labels: `b1w` 1 / 0 / 11, pooled 1 / 0 / 11. Clopper-Pearson over its 80 cells at both deltas, rare labels included: 0 / 0 / 80.

6. **section 7.1, reading 3.**
   > The registered rows 37, 41 and 49, the same limits in
   new designs, add 64 cells, none over and 2 unresolved.
   > the approximate bounds we use are over their level in 1 of 378 cells and unresolved in 24; the largest miss at delta 0.05 is 0.060
   > over in 8 of 528 mid-rate cells, all at n_s 100 on the two labels at 65% and above
   > `b1w` is over in 7 of 28 cells, by at most 1.2 points at delta 0.05, where the StratPPI estimator with a bootstrap-t limit and a stratified Wald-t limit are over in none
   > (36 cells, 4,000 draws) `b1w` is over in 5 cells and the Wald-t limit in 10

   1 over, 24 unresolved, 353 at or under of 378; the largest at delta 0.05 is 0.0595. The 378 are not 378 separate studies: the 168 PPI++ cells are 42 sets of draws read through four judge features, the delta 0.10 cells of the `b1w` rows reuse their delta 0.05 draws, and the reference-rate cells of the StratPPI row share seeds with the `b1w` rows. Other settings of the strata: 8 / 27 / 493 of 528. With replacement the bootstrap-t StratPPI limit is 0 / 0 / 28; the Wald-t limit is over in 0 of 160 cells, both designs and both deltas, rare labels included. Synthetic i.i.d. grid: `b1w` 5 / 7 / 24, Wald-t 10 / 10 / 16 of 36.

7. **abstract.**
   > promised to be wrong in at most 5% of repeated trials, the standard one (Clopper-Pearson) was wrong in at most 5.3% of ours, a difference within the noise of the check

   Largest Clopper-Pearson miss at delta 0.05 outside the AgentDojo per-pair rows: 0.0528 (the replacement check's paired random draws, 5,000 a cell; not over by the cell rule). Exact bounds at delta 0.05 (Clopper-Pearson on spike 017's plasmodes and on the random draws with replacement): largest upper limit of a cell's interval 0.0588. Bootstrap-t limits at delta 0.05: PPI++ with a bootstrap-t limit (017_boot) 0.0609, StratPPI estimator with a bootstrap-t limit (p14_a_sboot_mid_0.05) 0.0513, StratPPI estimator with a bootstrap-t limit (p14_b_sboot) 0.0636, StratPPI estimator with a bootstrap-t limit, redrawn with replacement (rep_sboot_mid_0.05) 0.0518, cluster bootstrap-t, by user task (ad_t_user_a) 0.0577. None of these groups has a cell over.

8. **section 7.2.**
   > with Clopper-Pearson a candidate that passed and truly violated was returned in at most 1 of 2,000 runs in any cell
   > whose bound fell below the truth in at most 0.055 of runs at delta 0.05; it was unresolved above delta in 3 of 42 cells

   Clopper-Pearson: 42 of 42 cells at or under delta, the largest 1 of 2,000. Wald: 0 / 3 / 39 of 42.

9. **section 7.2.**
   > missed 0.084-0.116 at delta 0.1 (Monte Carlo standard error 0.013): three cells at or under delta and one unresolved above it
   > A random split with a pooled bound missed 0.090-0.100

   Stratified: 0.0840-0.1160; the unresolved cell is 0.1160 (58 of 500; p = 0.13; 95% interval 0.0893-0.1474). At 500 runs a cell the check resolves only a miss above 0.1268. Random split, pooled: 0.0900-0.1000, all four at or under.

10. **section 7.2.**
   > A certificate at level delta/T over every one of T checks was at or under delta in all 10 cells, with misses of 0.025-0.100 against a delta of 0.1, at 200 runs a cell, which resolve only a miss above 0.14

   All 10 cells at or under delta, the largest exactly at it (20 of 200); the band ends at 0.1424.

11. **section 8.1; Appendix A.**
   > It is over its level in 4 of 10 cells at delta 0.05 when a cell is the largest miss over checkpoints (one of them marginally, at 0.056), and in 9 of 28 cells counted by checkpoint
   > the same estimator with a bootstrap-t limit is over its level in none of the ten cells (largest miss 0.045)
   > delta 0.1 StratPPI as published is over its level in 3 of 10 cells

   StratPPI's normal limit: 4 of 10 over at delta 0.05 (3 at 0.1); by checkpoint 9 of 26. The marginal cell is 0.0564 (282 of 5,000; p = 0.022; 95% interval 0.0502-0.0632). Bootstrap-t: 0 of 10, largest 0.0452.

12. **section 8.1.**
   > `b1w` is over its level in none of the 28 mid-rate cells at either delta (3 unresolved at delta 0.05, on the 93% label) and in 1 of its 12 rare-label cells at delta 0.10 (0.126)
   > it is over in 7 of 28 at delta 0.05 and 6 at 0.10, all at rates of 65% and above, and the bootstrap-t StratPPI limit and the stratified Wald-t limit are over in none

   Without replacement `b1w` is 0 / 3 / 25 of 28 at delta 0.05 and over in none at 0.1; with replacement 7 / 1 / 20, the bootstrap-t StratPPI limit 0 / 0 / 28.

13. **Appendix A.**
   > At delta 0.05 it is also over in every one of the four rare-label cells, where `b1w` is over in none.
   > 9 of 26 cells are over with the oracle rule and 11 with the heuristic
   > estimator is over its level in 6 of 26 cells under the oracle allocation and in 11 under the
heuristic, against none of 26 under proportional allocation

   Rare labels, largest miss over checkpoints: StratPPI's normal limit over in 4 of 4, `b1w` in 0. Normal limit with the paper's allocations: 9 and 11 of 26. Bootstrap-t: 6 and 11 of 26.

14. **Appendix A; Table 4's caption.**
   > with 10 labels a stratum the same cell still misses in 59% of draws, and four other cells on this judge miss in 68% to 85%

   StratPPI's normal limit under the heuristic allocation, rubric judge: 0.9025, 0.8535, 0.7967, 0.7935, 0.6820 in its five worst cells; the worst cell with a floor of 10 labels: 0.5895. Enumeration against the training paper's printed table: 25 of 25 within 2 se at a pool count of 124 (largest 1.27), 14 at 121.

15. **section 7.2; section 7, item 3; section 11.**
   > For the pool's own rate it was over its level in none of the 8 cells (2 unresolved; largest miss 0.052 at delta 0.05 and 0.100 at 0.10).
   > it was over in the least heterogeneous environment at both deltas (0.057 and 0.107; we had predicted the cell at delta 0.10, and at most two of the
eight), and in no other cell; `b1w` with the term and the pooled bound on a random split were over in none.
   > from 1.1 at 0.43 to 1.7 for the population's rate and 3.1 for the pool's own at 0.98
   > with 8 strata and 8,000 runs a cell the Wald-t limit was over in 2 of 16 cells
   > in one cell of four in the 4-strata in-loop check and in 2 of 16 in the 8-strata one

   O2: o2_b1_pool_0.05: 0 over, largest 0.0521; o2_b1_pool_0.1: 0 over, largest 0.1003; o2_b1N_pop_0.05: 1 over, largest 0.0566; o2_b1N_pop_0.1: 1 over, largest 0.1071

16. **section 8.1.4; section 7, item 3.**
   > the Wald-t limit's `ESS_truth` at delta 0.05 against the pooled Wilson bound was 0.97 to 1.34 (median 1.18), and `b1w`'s 1.10 to 1.36 (median 1.24)
   > did about as well (medians 1.19
and 1.25)
   > The Wald-t limit was over its level in 3 of the 64 cells drawn with replacement
(both stratifiers, safety sets of 50 and 100, both deltas; all three at 100), by at most
0.008, and in none of the 64 drawn without replacement
   > but `b1w` was over its
level in 8 of the 32 mid-rate cells drawn with replacement at delta 0.05 and 11 at 0.10
   > baseline is itself over its level in 6 and 8 of these 32 cells
   > it was over in 3 of 64 cells, by at most 0.008

   O3: 8 mid-rate labels, rates 0.0640-0.5960; Wald-t gain 0.9669-1.3446 (median 1.1817); b1w 1.1013-1.3645; over in 3 of 64 large cells.

17. **section 8.2.**
   > StratPPI as published, with 5 or 10 strata, is over its level in 26 of 28 cells (misses 0.059-0.239 in those 26), as PPI++ with a normal limit is in all 14 (0.061-0.232).
   > level in none of the 28 (5 unresolved, largest miss 0.056)
   > Stratifying on the judge and ignoring it within strata (`b1w`) is also over in none (1 unresolved, largest miss 0.053)
   > is over its level in 7 with the basic estimator and 12 with the power-tuned one

   StratPPI's normal limit: 26 of 28 over (0.0590-0.2387); PPI++ normal: 14 of 14 (0.0610-0.2320). Bootstrap-t StratPPI: 0 / 5 / 23. `b1w` on judge strata: 0 / 1 / 27. PPBoot: 7 and 12 of 14 over.

18. **section 8.1, a claim about the prompt source; abstract.**
   > `b1w` is over its level in none of the 28 mid-rate cells at delta 0.05 (4 unresolved, largest miss 0.052) and in one at 0.10 (0.112, on the 93% label); the Wald-t limit is over in none at either.
   > is 1.83 and 1.54 for over-refusal (1.68 and 1.44 when the Lagrangian pushes the label), 2.36 and 1.75 for refusal of harmful requests, and 1.35 and 1.25 for non-refusal of encoded requests.
   > gives 1.84 and 1.54, 1.71 and 1.46, 2.68 and 1.92, and 1.35 and 1.24
   > `b1w` without the term is over in 17 of the 28 cells (misses up to 0.170). So is the bootstrap-t StratPPI limit, in 21 (up to 0.166)
   > On the two rare labels `b1w` is over in 1 of 12 cells at delta 0.05 and 3 at 0.10, with or without the term
   > gives 2.39 and 2.47, 2.11 and 2.12, 4.66 and 5.14, and 1.50 and 1.52.
   > the strata lose (0.88 and 0.93), and at rare rates they gain nothing (1.01 to 1.05).
   > is itself over its level in 7 of these 28 cells

   10,000 two-phase replications a cell. `b1w` with the term: 0 / 4 / 24 at delta 0.05, 1 / 0 / 27 at 0.10 (the cell over: C2:refusal, n_s 100, step 100, 0.1116 (1,116 of 10,000; p < 0.0001; 95% interval 0.1055-0.1179)). Wald-t with the term: 0 / 1 / 27 and 0 / 0 / 28. Without the term: 17 / 1 / 10; bootstrap-t StratPPI: 21 / 2 / 5. Clopper-Pearson on a random sample of the pool, the control: over in none at either delta. ESS and caps are medians over a label's checkpoints, from the file's `ess` table.

19. **section 8.1, confirmation on new prompts; abstract; section 11.**
   > Its refusal rates came out at 12%, 82% and 98%, so no label fell between 18% and 65%
   > Six predictions were kept and two refuted
   > in all 4 at 82% (misses 0.062 at delta 0.05 and 0.111-0.113 at 0.10)
   > `b1w` 1.53 and 1.62, the Wald-t limit 1.47 and 1.73; the bootstrap-t StratPPI limit, which also held there, 1.55 and 1.88) and 0.86 and 1.13 at 82% (the Wald-t limit; `b1w`'s 1.49 and 1.62 there
   > is over in 1 of 8 mid-rate cells (0.105 at delta 0.10 on the 82% label)
   > its gain at 82% is 0.86 and 1.13 with a large pool and 0.75 and 0.89 for the source, against 1.47 and 1.73 at 12%
   > kept six of eight predictions; it covers refusal rates of 12%, 82% and 98% and two rare labels on one model
   > and in one more at the 98% rate (0.052 at delta 0.05)
   > (0.058 at delta 0.05 and 0.117 at 0.10; unresolved at 200 and delta 0.05, 0.052)
   > it gives 2.67 at the 98% rate, where with a large pool the strata lose under every limit (0.55 and 0.63 for `b1w`, against a baseline itself over its level at a safety set of 100), and 2.47 on a rare label
   > gained 1.5 to 1.9 at 12% (`b1w` 1.53 and 1.62, the Wald-t limit 1.47 and 1.73; the bootstrap-t StratPPI limit, which also held there, 1.55 and 1.88) and 0.86 and 1.13 at 82%
   > the stratified Wald-t limit is over in none of the 8 (misses at most 0.039 at delta 0.05 and 0.082 at 0.10)
   > term is over in none of 8 (at most 0.047 and 0.088), with a gain of 1.31 and 1.32 at 12% and 0.75 and 0.89 at 82%
   > the Wald-t limit with the term is over in one rare-label cell for the source (0.110 at delta 0.10, with a safety set of 200)
   > baseline that is itself over its level there at a safety set of 100 (against Clopper-Pearson, which held, the large-pool figures are 1.06 and 1.32)
   > at 98% it is over in none of its 4 cells
   > measure gives 1.5 to 5.0 for `b1w` at rates of 9% to 66% on the pools that chose the rule

   The registered run (`results/paper/confirm/confirm.md`): P1 refuted (Wald-t b1: 0 of 8 over; StratPPI estimator, bootstrap-t: 1 of 8 over); P2a kept (0 of 4 cells over on 1 such labels); P2b kept (K2:refusal: 4 of 4); P3 kept (K1:refusal n_s 100: 1.53 against 1.82; K1:refusal n_s 200: 1.62 against 1.82; K2:refusal n_s 100: 1.49 against 1.65; K2:refusal n_s 200: 1.62 against 1.65); P4a refuted (`b1w`: 1 of 4; Wald-t: 0 of 8); P4b kept (no term: 3 of 4; bootstrap-t StratPPI: 4 of 4); P4c kept (K1:refusal n_s 100: 1.32 against 1.35; K1:refusal n_s 200: 1.22 against 1.24; K2:refusal n_s 100: 1.22 against 1.33; K2:refusal n_s 200: 1.20 against 1.24); P5 kept (K2:unsafe n_s 100: 1.06; K2:unsafe n_s 200: 1.11; K3:unsafe n_s 100: 1.01; K3:unsafe n_s 200: 1.07; Clopper-Pearson 0 of 8 over).

20. **section 8.1, a pool aimed at the gap; abstract; section 11.**
   > By prompt category the three pools' refusal rates run from 0% to 22%, from 46% to 89% and from 87% to 100%; three categories fall inside the range, with 47, 20 and 17 prompts
   > From the reference samples alone, with `omega = 1`, it predicts 2.58, 4.12 and 2.22, and the realised gains are 27% to 41% lower
   > (one cell unresolved, 0.102 at delta 0.10)
   > is itself over its level at a safety set of 200 and delta 0.05 (0.053, and 0.057 for the source)
   > are both over at a safety set of 100 and delta 0.10 (0.140 and 0.142), and Clopper-Pearson is not
   > With 4 strata it was over in 1 cell of 26 (0.062 at n_s 100 and delta 0.05, on the 9% label)
   > the 193 hard-1K prompts still unused and 207 unused benign prompts of the 80K set
   > is 1.93 times that of a stratified one with two strata, the two sources, and 2.90 times with the 8 strata
   > The refusal rate came out at 46% (81% on the hard prompts, 12% on the others)
   > Nine predictions were kept, none was refuted, and one had no label to test it
   > (misses 0.026 and 0.036 at delta 0.05)
   > (three cells unresolved, the largest miss 0.001 above its level)
   > (0.044 and 0.045 at delta 0.05, 0.096 at 0.10)
   > On the re-mix, at a rate of 48%, `b1w` sat at its level (0.051 and 0.102, unresolved)
   > The gain of `b1w` is 2.78 and 2.97 against a pre-flight of 2.79, and that of the Wald-t limit 2.09 and 2.56
   > with a gain of 1.49 and 1.34, 7% to 9% under its cap
   > fail in both cells at delta 0.05 (misses 0.079 to 0.120)
   > The safety-field label was rare (1.9%) and the strata gained nothing on it (1.00 and 1.08)
   > At all three mid rates, 12%, 46% and 82%, the pre-flight was within 20%
   > against 1.47 and 1.73 at 12% and 2.09 and 2.56 on the mixed pool at 46%
   > At 46% the Wald-t limit kept 75% to 86% of `b1w`'s gain
   > was wrong no more often than promised at breach rates of 12%, 46% and 82%
   > 1.5 to 1.7 times its size at 12% and 2.1 to 2.6 times at 46%
   > The gain is smaller, 1.3 to 1.5 at the first two rates, when the certificate must cover all prompts of that kind
   > At 82% it gained little or lost, at 98% it lost, and for rare behaviours (under 4%) it gained nothing
   > Of 72 predictions
registered in advance, 53 held, 18 failed and one could not be tested
   > Between 18% and 65% Granite has one rate, 46%, from a fourth pool mixed from two sources

   The registered run (`results/paper/confirm/mid/confirm.md`): M1 kept (0 of 4 over); M2 kept (0 of 4 over; largest miss minus level +0.001); M3a no label (rate 0.459; 0 of 4 over); M3b kept (rate 0.459; miss minus level from -0.006 to -0.004); M4 kept (n_s 100: 2.78 against 2.79; n_s 200: 2.97 against 2.79); M5 kept (n_s 100: 2.09; n_s 200: 2.56); M6 kept (0 of 4 over); M7 kept (no term: 2 of 2; bootstrap-t StratPPI: 2 of 2); M8 kept (n_s 100: 1.49 against 1.64; n_s 200: 1.34 against 1.44); M9 kept (unsafe n_s 100: 1.00; unsafe n_s 200: 1.08; Clopper-Pearson 0 of 4 over).

21. **abstract; Appendix D.**
   > Of 72 predictions
registered in advance, 53 held, 18 failed and one could not be tested
   > | T1 | the multiway variance with a t quantile is over for some pipelines | kept: 26 of 28 |
   > | T3 | the quadrature bound is over for none | refuted: one (0.058) |
   > *Table D2. The 28 predictions registered on 2026-10-14

   72 predictions scored from the result files: 53 kept, 18 refuted, 1 without a label.

22. **sections 9.1 and 9.2; sections 5, 11 and 13; Appendix B.**
   > Between the two benign sources the bound was over its level for 4 of 6 wordings one way and 1 of 6 the other; between the pools, for 5 of 6 one way and none the other.
   > 0.03, 0.07 and 0.05 for the canonical wording in seeds 0, 1 and 2, recall differed in 2, 2 and
1 of 6 wordings
   > the carried bound was over its level for 3, 3 and 2 of 6 (seed 0: misses
0.08, 0.23 and 0.80, and marginal for a fourth at 0.054; the largest in seeds 1 and 2 are 0.399
and 0.625), although the pool's rate was within 1.4 points of the reference's in each seed
   > At step 100 the bound was over for 1, 1 and 4 of 6.
   > returned NSF (their candidates, steps 200 and 175, had upper bounds 0.2317 and 0.2064 against a
threshold of 0.2058; seed 0's step 175 had 0.1979 and was returned)
   > for 2 of the 3 seeds of the constrained run of section 9.2
   > rose from 37% in the reference to 82% in the trained policy, and to 76% and 82% in the two
further seeds

   4, 1, 5 and 0 of 6 over between populations. Constrained run, seeds 0-2: 3, 3, 2 over at step 200 (seed 0: 1 unresolved, 0.0537 (215 of 4,000; p = 0.15; 95% interval 0.0470-0.0612); largest in seeds 1, 2: 0.399, 0.625); 1, 1, 4 of 6 over at step 100. Recall (wording 0) at step 200: 0.032, 0.074, 0.052; safety-test upper bounds 0.1979, 0.2317, 0.2064 against 0.20575. The side-effect run: 6 of 6 at or under delta.

23. **section 10.2; abstract.**
   > that bound is over its level for none (2 unresolved, neither of them the pipeline that certifies below)
   > same bound is over its level for 24 of 28 pipelines. So are the others we had: the bootstrap by injection task (23), the pigeonhole bootstrap of Owen (2007) with a basic limit, which resamples both kinds of task (27), the bound clustered in the direction of the larger intraclass correlation (22), and the larger of the two clustered bounds
   > the usual certificate is wrong too often for 27 of 28 agent configurations

   Scheme (a): cluster bootstrap-t by user task 0 / 2 / 26; per-pair Clopper-Pearson over for 27 of 28. Scheme (c): 24, 23, 27 and 16 of 28 over; the quadrature bound 0. The two-way bootstrap returns 0 when a redrawn table has no success; such tables are at most 0.7%, 1.2% and 5.8% of a cell's draws under the three schemes, and taking them out of the misses leaves its counts of cells over unchanged (5, 14, 27).

24. **section 10.2; section 12; abstract.**
   > multiway limit is over its level for 26 of 28 pipelines when both kinds of task are resampled (misses up to 0.18)
   > The pigeonhole bootstrap-t is over for none under any of the three schemes (largest miss 0.042)
   > (median ratio 2.05), and no pipeline certifies 5% under it
   > (misses up to 0.18), at rates near 30% as well as at rates of a few percent
   > for 7 of the 28 pipelines it returns no limit in
more than 5% of the resampled tables when both kinds of task are resampled (in 77% for the
pipeline with the lowest rate)
   > is over for one pipeline on the fresh draws (0.058)
   > rates of 1% to 56%
   > at most one of the 28 certifies an attack success rate under 5%

   Scheme (c), 4,000 resampled tables a pipeline: multiway-t 26 / 1 / 1 (largest 0.1787); pigeonhole bootstrap-t 0 / 0 / 28, and (a) 0 / 0 / 28; (b) 0 / 0 / 28; quadrature 1 / 1 / 26 (claude-3-haiku-20240307, 0.0580 (232 of 4,000; p = 0.012; 95% interval 0.0510-0.0657)), against 0 over on the earlier draws. On the published tables the pigeonhole bootstrap-t's margin is 2.05 times the user-task bound's at the median (1.17 to 87.49), 0 pipelines certify 5% and 1 gets no limit. Table 7's last column is asserted against the same file.

25. **section 10.2; section 11.**
   > both
are over their level for no pipeline (misses at most 0.007), and so is the bound clustered by
user task within suites (largest miss 0.042)
   > bound is over for 11 of 28 pipelines, against 21 when they are redrawn across the table
   > unblocked redraws lost a whole suite in under 1% of tables
   > are then over for 3 (against the
average over user tasks) and 2 (the pair-rate version), and the quadrature bound for none
   > certifies 5% for the same pipeline (0.048, against 0.022 clustered) and for no other; the
pair-rate version certifies none (0.060 for that pipeline)
   > user task is still over for 11 of 28 pipelines when injection tasks are sampled

   O4: betting bounds under (a), (a_s) over in none (largest 0.0070); cluster bootstrap-t (a_s) 0/0/28, (b_s) 11/1/16, (b) 21/0/7; betting (b_s) 3 and 2. Real tables: (a1) certifies ['claude-3-5-sonnet-20241022']; (a2) none. Table 7's exact column is asserted against the same file.

26. **section 8.1.3; sections 7.1, 8.1.5 and 11.**
   > The pool's refusal rate went from 19.7% to 19.1%; it
fell on the categories the reward pushed up (16.9% to 14.8%) and rose on those it pushed down
(22.5% to 23.5%)
   > was 0.84 at step 200, where we predicted under 0.8
   > with a large pool the Wald-t limit is over its level in none of its 8 cells (misses at most
  0.046 at delta 0.05 and 0.092 at 0.10), and `b1w` in none (at most 0.041 and 0.092), where
  StratPPI's normal limit is over in all 8 and the pooled Wilson baseline in 4
   > the Wald-t limit's gain is 1.52 and 1.77 at safety sets of 100 and 200
   > `b1w`'s 1.60 and 1.70
  against a pre-flight of 1.73 with `ICC_cand` and `omega` measured; from the reference samples alone the pre-flight predicted 1.55, or 1.30
  with the pushed-label rho
   > rho fell from 0.88 to 0.84 and the Wald-t limit's gain at a safety
  set of 200 from 1.84 to 1.77
   > the Wald-t limit with the sampled-pool term is over in 1 of 8 cells
  (0.111 at delta 0.10, a safety set of 200, step 200) and `b1w` with it in none, while `b1w`
  without the term is over in 6 of 8
   > Three of the six predictions were kept

   O6 at step 200: rate 0.197 -> 0.191; UP 0.169 -> 0.148; DOWN 0.225 -> 0.235; rho 0.843 (step 100 0.878); reward-model term +4.97 from the first 25 calls to the last 50.

27. **section 8.1.3 (B3); sections 5, 8.1.5, 11 and 13; abstract.**
   > At step 200 the pool's refusal rate was 58.0%, against the
reference's 19.7%, and rho fell to 0.46 (0.87 at step 100), where we predicted under 0.7
   > from 16.9% to 67.4%
on those pushed up and from 22.5% to 48.7% on those pushed down
   > the Wald-t limit is over its level in none of its 8 cells over both steps (misses at most
  0.049 at delta 0.05 and 0.100 at 0.10)
   > `b1w` is over in all 4 cells at step 200 (up to 0.059 and 0.112) and in none at step 100
  (26%; one unresolved)
   > Wilson baseline is over in 4 of 8 cells
   > the Wald-t limit's is 0.97 and 1.08 at safety sets of 100 and 200
   > against 1.36 and 1.46 at step 100
   > gives 1.14, and `b1w`'s 1.22
   > pre-flight predicted 1.55, or 1.30 with the pushed-label rho
   > over in none of 8
  cells (at most 0.048 and 0.094), `b1w` with it in 2, and `b1w` without it in 7
   > Four of the five predictions were kept

   B3 at step 200: rate 0.197 -> 0.580; UP 0.169 -> 0.674; DOWN 0.225 -> 0.487; rho 0.462 (step 100 0.870).

28. **section 10.4; Appendix C.**
   > puts the miss rates of the four limits at or under 0.045 against a level of 0.05 when the 490 prompts are fixed
   > the labels-alone bootstrap-t and (b2) are at 0.051 and 0.052
   > puts the miss rates of the four limits at 0.001, 0.040, 0.045 and 0.042 for the pool rate against a level of 0.05
   > the four miss rates are 0.001, 0.051, 0.063 and 0.052

   Pool scheme: all four at or under delta at 2,000 repetitions. New-prompts scheme: (a) at or under, (a') and (b2) unresolved above delta, (b1), not built for that scheme, over.

29. **section 8.1.4 (exact stratified bounds; B25).**
   > computed exactly, its miss is at most 0.048 at delta 0.05 over all 116 cells
   > the stratified Clopper-Pearson bound has a width gain of 0.82 against Clopper-Pearson (median; 0.50 to 1.00) and a power ESS of 1.11 (median; 0.27 to 1.83); it certifies more often than Clopper-Pearson in 19 of the 36 cells. At the lower threshold, the one Clopper-Pearson passes with probability 0.5, its power ESS is 0.67
   > on the costly label it certifies more often in none of the 32 cells
   > the betting bound has a width gain of 0.20 and a power ESS of 0.05
   > the Wald-t limit has a width gain of 2.02 against Clopper-Pearson (0.57 to 3.95; 1.86 against the pooled Wilson bound) and a power ESS of 1.88 (median; 0.47 at the lowest)
   > On the 32 costly-label cells in that range the stratified Clopper-Pearson bound's power ESS is 0.88 (median) and the Wald-t limit's 1.22, and on rare labels the stratified bound loses (0.69 on the real cells)
   > Four of the eight predictions held
   > the median was 0.92, the lowest 0.27, and the Wald-t limit was above it in 35 of 36
   > the stratified bound's median is 1.00 over the 68 real and costly cells (lowest 0.42), the Wald-t limit's 2.10 on the real ones, and the pooled Wilson bound's 1.07
   > The pooled Wilson baseline, computed exactly, is over its level in 10 of the 36 real-response cells
   > ESS is above 1 in 26 of the 36 real cells and in 8 of the 32 costly ones
   > stratified Clopper-Pearson bound did so in 19 of the 36 real-response cells and in none of
the 32 costly ones

   results/paper/exact_strat.json (registered) and exact_strat_tau.json (the step-spread sensitivity), delta 0.05.

30. **section 10.2 (a fitted crossed model; B6); section 11.**
   > the bound clustered by user task is over its level for no pipeline (largest miss 0.044), and neither are the betting bounds (0.005), the quadrature bound (0.016), the pigeonhole bootstrap-t (0.011) or the larger of the two clustered bounds (0.036); the multiway limit is over for 4
   > user task is over for 18 of 28 pipelines (misses up to 0.25), the multiway limit for 13, the betting bounds for 5 (against the user-task average) and 3 (the pair rate), and the larger of the clustered bounds for 3
   > The quadrature bound and the pigeonhole bootstrap-t are over for none (largest misses 0.047 and 0.037). All five predictions held
   > whose rate under the model is 1.6% given the benchmark's injection tasks (1.4% over new ones), certifies in 99% of tables with the injection tasks fixed under the bound clustered by user task, and in 35% under the betting bound on user-task rates
   > for two pipelines the user-task one sits at the grid's edge of 6

   P_a over: {'t_user': 0, 'bet_a1@uavg': 0, 'bet_a2': 0, 'quad': 0, 'pig_t': 0, 'max': 0, 'cgm_t': 4}; P_c over: {'t_user': 18, 'bet_a1@uavg': 5, 'bet_a2': 3, 'quad': 0, 'pig_t': 0, 'max': 3, 'cgm_t': 13}; grid edge: ['Meta-SecAlign-70B', 'gpt-3.5-turbo-0125'].

31. **section 9.2 (a second judge family; B4); Appendix B; section 11.**
   > its recall on the guard's refusals rose with training, from 0.88 at step 0 to 0.94, 0.94 and 0.91 at step 200 (no rise significant; Fisher p 0.22 to 0.65); its false alarms were 1.0% at step 0 and 1.0% to 1.7% at step 200, and the carried bound was over its level in no seed (misses 0.000)
   > canonical wording's recall on WildGuard's refusals fell from 0.23 to 0.03, 0.08 and 0.05, and its carried bound was over its level in all three seeds (misses 0.84, 0.46 and 0.66)
   > WildGuard flagged 95% of the 607 responses with the surface pattern of Appendix B and 92% of the 479 without it (Fisher p 0.064
   > (design-weighted accuracy 0.930 and 0.929; recall 16 of 18 each), and the compiled judge flags none of the 18
   > its carried bound missed in 85% of draws
   > it flags 95% of the guard-flagged responses with the pattern and 92% of those without

   results/paper/b4/b4.json: shifts A (WildGuard as judge) and B (WildGuard as gold), sheet C, flags by shape D; Q1-Q5 Q1 refuted, Q2 refuted, Q3 refuted, Q4 kept, Q5 refuted.

## (d) The two lists the rule produces

**Bounds the paper says hold, with a cell over.**

- `b1w` on a design-weighted sheet (Table 4 row 8, [017 4]): wording 2, planted rate 0.2, f01, 0.0595 (238 of 4,000; p = 0.0039; 95% interval 0.0524-0.0673); over after Bonferroni for 18 cells: no.

**Bounds the paper says fail, with their cells that are not over.**

- `b1w` at rare rates, labels at 1-2%, n_s 100-200, 3 checkpoints, delta 0.1 (Table 4 row 16): 1 of 12 over, 0 unresolved, 11 at or under; pooled miss 0.0440.
- pooled Wilson bound at rare rates, labels at 1-2%, n_s 100-200, 3 checkpoints, delta 0.1 (Table 4 row 16): 1 of 12 over, 1 unresolved, 10 at or under; pooled miss 0.0467.
- `b1w`, redrawn with replacement, 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.05 (Table 4 row 17; section 7.1): 7 of 28 over, 1 unresolved, 20 at or under; pooled miss 0.0389.
- `b1w`, redrawn with replacement, 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.1 (Table 4 row 17; section 7.1): 6 of 28 over, 2 unresolved, 20 at or under; pooled miss 0.0863.
- pooled Wilson bound, redrawn with replacement, 5 mid-rate labels, random draws, n_s 100-200, by checkpoint, delta 0.05 (Table 4 row 18; section 7.1): 6 of 28 over, 0 unresolved, 22 at or under; pooled miss 0.0477.
- pooled Wilson bound, redrawn with replacement, 5 mid-rate labels, random draws, n_s 100-200, by checkpoint, delta 0.1 (Table 4 row 18; section 7.1): 10 of 28 over, 3 unresolved, 15 at or under; pooled miss 0.0967.
- PPI++ with a normal limit, spike 017 plasmodes, every cell and feature (the cells of its table B6), delta 0.05 (Table 4 row 19): 126 of 168 over, 29 unresolved, 13 at or under; pooled miss 0.0783.
- StratPPI as published (normal limit), 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.05 (Table 4 row 20; Table A1): 9 of 28 over, 5 unresolved, 14 at or under; pooled miss 0.0498.
- StratPPI as published (normal limit), judge-logit strata (5 and 10) on the refusal pools, n 100-1,000, delta 0.05 (Table 4 row 20; section 8.2): 26 of 28 over, 2 unresolved, 0 at or under; pooled miss 0.0982.
- StratPPI's normal limit, oracle allocation, reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out), delta 0.05 (Table 4 row 21; Appendix A): 9 of 26 over, 3 unresolved, 14 at or under; pooled miss 0.0475.
- StratPPI's normal limit, heuristic allocation, reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out), delta 0.05 (Table 4 row 21; Appendix A): 11 of 26 over, 0 unresolved, 15 at or under; pooled miss 0.0723.
- StratPPI's normal limit, heuristic allocation, judge-logit strata (5 and 10), n 100-1,000, delta 0.05 (Table 4 row 21; Appendix A): 24 of 28 over, 1 unresolved, 3 at or under; pooled miss 0.3317.
- StratPPI estimator with a bootstrap-t limit, oracle allocation, reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out), delta 0.05 (Table 4 row 22; Appendix A): 6 of 26 over, 3 unresolved, 17 at or under; pooled miss 0.0453.
- StratPPI estimator with a bootstrap-t limit, oracle allocation, judge-logit strata (5 and 10), n 100-1,000, delta 0.05 (Table 4 row 22; Appendix A): 14 of 28 over, 11 unresolved, 3 at or under; pooled miss 0.0589.
- StratPPI estimator with a bootstrap-t limit, heuristic allocation, reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out), delta 0.05 (Table 4 row 22; Appendix A): 11 of 26 over, 1 unresolved, 14 at or under; pooled miss 0.0694.
- StratPPI estimator with a bootstrap-t limit, heuristic allocation, judge-logit strata (5 and 10), n 100-1,000, delta 0.05 (Table 4 row 22; Appendix A): 14 of 28 over, 0 unresolved, 14 at or under; pooled miss 0.2942.
- PPBoot, percentile limit, basic, judge-logit cells, unstratified, n 100-1,000, delta 0.05 (Table 4 row 23; section 8.2): 7 of 14 over, 6 unresolved, 1 at or under; pooled miss 0.0653.
- PPBoot, percentile limit, power-tuned, judge-logit cells, unstratified, n 100-1,000, delta 0.05 (Table 4 row 23; section 8.2): 12 of 14 over, 2 unresolved, 0 at or under; pooled miss 0.0755.
- Clopper-Pearson over pairs, AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed, delta 0.05 (Table 4 row 24; section 10.2): 27 of 28 over, 0 unresolved, 1 at or under; pooled miss 0.1534.
- cluster bootstrap-t, by user task, AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed, delta 0.05 (Table 4 row 25; section 10.2): 21 of 28 over, 0 unresolved, 7 at or under; pooled miss 0.1064.
- cluster bootstrap-t, by user task, AgentDojo, 28 pipelines, scheme (c): both resampled, delta 0.05 (Table 4 row 25; section 10.2): 24 of 28 over, 0 unresolved, 4 at or under; pooled miss 0.1522.
- larger of the two clustered bounds, AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed, delta 0.05 (Table 4 row 26; section 10.2): 4 of 28 over, 0 unresolved, 24 at or under; pooled miss 0.0335.
- larger of the two clustered bounds, AgentDojo, 28 pipelines, scheme (c): both resampled, delta 0.05 (Table 4 row 26; section 10.2): 16 of 28 over, 3 unresolved, 9 at or under; pooled miss 0.0556.
- two-way bootstrap, AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed, delta 0.05 (Table 4 row 27; section 10.2): 5 of 28 over, 0 unresolved, 23 at or under; pooled miss 0.0318.
- two-way bootstrap, AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed, delta 0.05 (Table 4 row 27; section 10.2): 14 of 28 over, 1 unresolved, 13 at or under; pooled miss 0.0655.
- two-way bootstrap, AgentDojo, 28 pipelines, scheme (c): both resampled, delta 0.05 (Table 4 row 27; section 10.2): 27 of 28 over, 0 unresolved, 1 at or under; pooled miss 0.1236.
- stratified sheet read as an i.i.d. sample (PPI), sheets re-drawn by their real sampling rule, 3 wordings x 3 planted rates x 2 features, delta 0.05 (Table 4 row 28): 7 of 18 over, 0 unresolved, 11 at or under; pooled miss 0.2364.
- the judge's rate alone, spike 017 plasmodes, every cell and feature (the cells of its table B6), delta 0.05 (Table 4 row 29): 30 of 42 over, 0 unresolved, 12 at or under; pooled miss 0.6353.
- multiway cluster variance with a t quantile, AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed, delta 0.05 (Table 4 row 30; section 10.2): 5 of 28 over, 0 unresolved, 23 at or under; pooled miss 0.0259.
- multiway cluster variance with a t quantile, AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed, delta 0.05 (Table 4 row 30; section 10.2): 8 of 28 over, 3 unresolved, 17 at or under; pooled miss 0.0557.
- multiway cluster variance with a t quantile, AgentDojo, 28 pipelines, scheme (c): both resampled, delta 0.05 (Table 4 row 30; section 10.2): 26 of 28 over, 1 unresolved, 1 at or under; pooled miss 0.1031.
- two clustered margins added in quadrature, fresh draws, AgentDojo, 28 pipelines, scheme (c): both resampled, delta 0.05 (Table 4 row 31; section 10.2): 1 of 28 over, 1 unresolved, 26 at or under; pooled miss 0.0344.
- stratified Wald-t `b1` on a synthetic i.i.d. grid, six strata profiles, n_s 100-400, binomial strata, delta 0.05 (Table 4 row 35; section 7.1 (reading 3)): 5 of 18 over, 5 unresolved, 8 at or under; pooled miss 0.0559.
- stratified Wald-t `b1` on a synthetic i.i.d. grid, six strata profiles, n_s 100-400, binomial strata, delta 0.1 (Table 4 row 35; section 7.1 (reading 3)): 5 of 18 over, 5 unresolved, 8 at or under; pooled miss 0.1050.
- `b1w` on a synthetic i.i.d. grid, six strata profiles, n_s 100-400, binomial strata, delta 0.05 (Table 4 row 36; section 7.1 (reading 3)): 1 of 18 over, 2 unresolved, 15 at or under; pooled miss 0.0405.
- `b1w` on a synthetic i.i.d. grid, six strata profiles, n_s 100-400, binomial strata, delta 0.1 (Table 4 row 36; section 7.1 (reading 3)): 4 of 18 over, 5 unresolved, 9 at or under; pooled miss 0.1024.
- cluster bootstrap-t, by user task, AgentDojo, 28 pipelines, scheme (b_s), delta 0.05 (Table 4 row 42): 11 of 28 over, 1 unresolved, 16 at or under; pooled miss 0.0597.
- cluster bootstrap-t, by user task, AgentDojo, 28 pipelines, scheme (c_s), delta 0.05 (Table 4 row 42): 19 of 28 over, 2 unresolved, 7 at or under; pooled miss 0.0977.
- t_user, a fitted crossed model, AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c), delta 0.05 (Table 4 row 50): 18 of 28 over, 3 unresolved, 7 at or under; pooled miss 0.1027.
- StratPPI's normal limit, redrawn with replacement, 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.05 (section 7.1): 20 of 28 over, 1 unresolved, 7 at or under; pooled miss 0.0599.
- StratPPI's normal limit, redrawn with replacement, 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.1 (section 7.1): 18 of 28 over, 1 unresolved, 9 at or under; pooled miss 0.1084.
- StratPPI as published (normal limit), 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint, delta 0.1 (section 8.1): 7 of 28 over, 3 unresolved, 18 at or under; pooled miss 0.0947.
- `b1w` without the term, a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200, delta 0.05 (section 8.1 (the prompt source)): 17 of 28 over, 1 unresolved, 10 at or under; pooled miss 0.0725.
- StratPPI estimator with a bootstrap-t limit, a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200, delta 0.05 (section 8.1 (the prompt source)): 21 of 28 over, 2 unresolved, 5 at or under; pooled miss 0.0777.
- `b1w` without the term, a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200, delta 0.1 (section 8.1 (the prompt source)): 18 of 28 over, 5 unresolved, 5 at or under; pooled miss 0.1291.
- StratPPI estimator with a bootstrap-t limit, a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200, delta 0.1 (section 8.1 (the prompt source)): 23 of 28 over, 1 unresolved, 4 at or under; pooled miss 0.1339.
- carried Youden-corrected bound, source: xstest -> orbench, six judge wordings, delta 0.05 (section 9.1): 4 of 6 over, 0 unresolved, 2 at or under; pooled miss 0.4020.
- carried Youden-corrected bound, source: orbench -> xstest, six judge wordings, delta 0.05 (section 9.1): 1 of 6 over, 0 unresolved, 5 at or under; pooled miss 0.0222.
- carried Youden-corrected bound, pool: harmful -> over-refusal, six judge wordings, delta 0.05 (section 9.1): 5 of 6 over, 0 unresolved, 1 at or under; pooled miss 0.4275.
- carried Youden-corrected bound, training, constrained (014): step 0 -> step 100, six judge wordings, delta 0.05 (section 9.2): 1 of 6 over, 0 unresolved, 5 at or under; pooled miss 0.1047.
- carried Youden-corrected bound, training, constrained (014): step 0 -> step 200, six judge wordings, delta 0.05 (section 9.2): 3 of 6 over, 1 unresolved, 2 at or under; pooled miss 0.1968.
- carried Youden-corrected bound, training, constrained (014 s1): step 0 -> step 100, six judge wordings, delta 0.05 (section 9.2): 1 of 6 over, 0 unresolved, 5 at or under; pooled miss 0.0551.
- carried Youden-corrected bound, training, constrained (014 s1): step 0 -> step 200, six judge wordings, delta 0.05 (section 9.2): 3 of 6 over, 0 unresolved, 3 at or under; pooled miss 0.1261.
- carried Youden-corrected bound, training, constrained (014 s2): step 0 -> step 100, six judge wordings, delta 0.05 (section 9.2): 4 of 6 over, 0 unresolved, 2 at or under; pooled miss 0.2134.
- carried Youden-corrected bound, training, constrained (014 s2): step 0 -> step 200, six judge wordings, delta 0.05 (section 9.2): 2 of 6 over, 0 unresolved, 4 at or under; pooled miss 0.1514.

Of these, with no cell over (only unresolved or under): none.

## (e) Not recounted, and why

- **[R 6.2], Table 4 rows 1, 3, 4, 5 and 15.** The cached labels were lost on 2026-09-19; the printed rates are reproduced by enumeration at the pool's rate (`results/paper/binomial_rows.md`). Classified here from the printed rates and the 5,000 resamples the training paper states; counts are round(m x 5,000).
- **[R 6.2], the Student-t bound at delta 0.05.** The training paper's sentence (0.084, 0.067, 0.083, 0.074, 0.067) repeats its Clopper-Pearson row in four of five values (the draft's open check 2). Not classified; Table 4 does not use it.
- **[R 6.3], section 7.2, tables (a), (b) and (d).** 11 printed cells with no class: the training paper gives '500-1,000 independent trials' a row and no count per row. Every printed rate (at most 0.0030) is under delta 0.1 whatever the count; the p-values and intervals need it. Table (c) is classified at the 500 trials of the reproduction command in that paper's Appendix B. Regenerable (`scripts/synthetic_calibration.py`), not re-run here.
- **Section 7.2, language-model policies.** 0 breaches in 3 seeds and in 10 seeds: the paper declines to read these as a miss rate, and three or ten draws give no class worth the name.
- **The exact Wilson calculation (`results/paper/wilson_exact.md`).** An enumeration, not a resampling study; it has no cells to classify. `scripts/wilson_exact.py` asserts its own closed forms.
- **Clustering of the sheet draws.** `harm.json` stores one rate per cell, not per planting, so a planting-level standard error cannot be formed without re-running `harm017.py`.
- **`plasmode_n500.json` (spike 017).** Not part of the spike's table B6 and not quoted in the paper; left out.

## Appendix. The cells behind (d)

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
| PPI++ with a bootstrap-t limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | holds | 13 cells, 0.0503-0.0535 | 0.05 | 4,000 | | | 13 unresolved, 0 at or under | | |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | holds | gpt-4o-2024-05-13-repeat_user_prompt | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | holds | gpt-4o-mini-2024-07-18 | 0.05 | 4,000 | 201 | 0.0503 | unresolved, above delta | 0.48 | [0.0437, 0.0575] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal rubric 0, rate pool, n 100 of 2000, K=5 | 0.05 | 4,000 | 224 | 0.0560 | unresolved, above delta | 0.046 | [0.0491, 0.0636] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal rubric 0, rate pool, n 100 of 2000, K=10 | 0.05 | 4,000 | 207 | 0.0517 | unresolved, above delta | 0.32 | [0.0451, 0.0591] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate pool, n 100 of 2000, K=5 | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate pool, n 500 of 2000, K=10 | 0.05 | 4,000 | 212 | 0.0530 | unresolved, above delta | 0.2 | [0.0463, 0.0604] |
| StratPPI estimator with a bootstrap-t limit | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | holds | refusal raw 0, rate 0.05, n 1000 of 4000, K=5 | 0.05 | 4,000 | 211 | 0.0527 | unresolved, above delta | 0.22 | [0.0460, 0.0601] |
| StratPPI estimator with a bootstrap-t limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | holds | C2:unsafe, n_s 200, step 100 | 0.1 | 40,000 | 4,011 | 0.1003 | unresolved, above delta | 0.43 | [0.0973, 0.1033] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 100, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step0 | 0.1 | 5,000 | 190 | 0.0380 | at or under delta | 1 | [0.0329, 0.0437] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 100, step100 | 0.1 | 5,000 | 425 | 0.0850 | at or under delta | 1 | [0.0774, 0.0931] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step100 | 0.1 | 5,000 | 136 | 0.0272 | at or under delta | 1 | [0.0229, 0.0321] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step200 | 0.1 | 5,000 | 333 | 0.0666 | at or under delta | 1 | [0.0598, 0.0739] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step100 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step100 | 0.1 | 5,000 | 460 | 0.0920 | at or under delta | 0.97 | [0.0841, 0.1004] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step200 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| `b1w` at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step200 | 0.1 | 5,000 | 463 | 0.0926 | at or under delta | 0.96 | [0.0847, 0.1010] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 100, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step0 | 0.1 | 5,000 | 239 | 0.0478 | at or under delta | 1 | [0.0421, 0.0541] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 100, step100 | 0.1 | 5,000 | 379 | 0.0758 | at or under delta | 1 | [0.0686, 0.0835] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step100 | 0.1 | 5,000 | 151 | 0.0302 | at or under delta | 1 | [0.0256, 0.0353] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C2:gated, n_s 200, step200 | 0.1 | 5,000 | 397 | 0.0794 | at or under delta | 1 | [0.0721, 0.0872] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step0 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step100 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step100 | 0.1 | 5,000 | 512 | 0.1024 | unresolved, above delta | 0.29 | [0.0941, 0.1111] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 100, step200 | 0.1 | 5,000 | 0 | 0 | at or under delta | 1 | [0, 0.0007] |
| pooled Wilson bound at rare rates | labels at 1-2%, n_s 100-200, 3 checkpoints | fails | C3:unsafe, n_s 200, step200 | 0.1 | 5,000 | 496 | 0.0992 | at or under delta | 0.58 | [0.0910, 0.1078] |
| `b1w`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 21 cells, 0.0171-0.0516 | 0.05 | 40,000 | | | 1 unresolved, 20 at or under | | |
| `b1w`, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 22 cells, 0.0563-0.1024 | 0.1 | 40,000 | | | 2 unresolved, 20 at or under | | |
| pooled Wilson bound, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | fails | 22 cells, 0.0312-0.0497 | 0.05 | 40,000 | | | 0 unresolved, 22 at or under | | |
| pooled Wilson bound, redrawn with replacement | 5 mid-rate labels, random draws, n_s 100-200, by checkpoint | fails | 18 cells, 0.0686-0.1028 | 0.1 | 40,000 | | | 3 unresolved, 15 at or under | | |
| PPI++ with a normal limit | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | 42 cells, 0.0408-0.0568 | 0.05 | 4,000 | | | 29 unresolved, 13 at or under | | |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 19 cells, 0.0238-0.0560 | 0.05 | 5,000 | | | 5 unresolved, 14 at or under | | |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | fails | refusal rubric 0, rate pool, n 500 of 2000, K=10 | 0.05 | 4,000 | 214 | 0.0535 | unresolved, above delta | 0.16 | [0.0467, 0.0609] |
| StratPPI as published (normal limit) | judge-logit strata (5 and 10) on the refusal pools, n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000, K=5 | 0.05 | 4,000 | 225 | 0.0563 | unresolved, above delta | 0.04 | [0.0493, 0.0638] |
| StratPPI's normal limit, oracle allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | fails | 17 cells, 0.0094-0.0558 | 0.05 | 5,000 | | | 3 unresolved, 14 at or under | | |
| StratPPI's normal limit, heuristic allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | fails | 15 cells, 0.0022-0.0492 | 0.05 | 5,000 | | | 0 unresolved, 15 at or under | | |
| StratPPI's normal limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000, K=5 | 0.05 | 4,000 | 179 | 0.0447 | at or under delta | 0.94 | [0.0386, 0.0516] |
| StratPPI's normal limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000, K=10 | 0.05 | 4,000 | 218 | 0.0545 | unresolved, above delta | 0.1 | [0.0477, 0.0620] |
| StratPPI's normal limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | refusal raw 0, rate pool, n 225 of 2000, K=5 | 0.05 | 4,000 | 183 | 0.0457 | at or under delta | 0.9 | [0.0395, 0.0527] |
| StratPPI's normal limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | refusal raw 0, rate pool, n 225 of 2000, K=10 | 0.05 | 4,000 | 140 | 0.0350 | at or under delta | 1 | [0.0295, 0.0412] |
| StratPPI estimator with a bootstrap-t limit, oracle allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | fails | 20 cells, 0.0074-0.0550 | 0.05 | 5,000 | | | 3 unresolved, 17 at or under | | |
| StratPPI estimator with a bootstrap-t limit, oracle allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | 14 cells, 0.0348-0.0558 | 0.05 | 4,000 | | | 11 unresolved, 3 at or under | | |
| StratPPI estimator with a bootstrap-t limit, heuristic allocation | reference-rate strata, 5 mid-rate labels, n_s 100-200, by checkpoint (one checkpoint at a 95% rate left out) | fails | 15 cells, 0.0064-0.0548 | 0.05 | 5,000 | | | 1 unresolved, 14 at or under | | |
| StratPPI estimator with a bootstrap-t limit, heuristic allocation | judge-logit strata (5 and 10), n 100-1,000 | fails | 14 cells, 0.0067-0.0498 | 0.05 | 4,000 | | | 0 unresolved, 14 at or under | | |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate 0.05, n 1000 of 4000 | 0.05 | 4,000 | 208 | 0.0520 | unresolved, above delta | 0.29 | [0.0453, 0.0593] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate 0.013, n 1000 of 4000 | 0.05 | 4,000 | 221 | 0.0553 | unresolved, above delta | 0.07 | [0.0484, 0.0628] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal rubric 0, rate pool, n 500 of 2000 | 0.05 | 4,000 | 227 | 0.0568 | unresolved, above delta | 0.029 | [0.0498, 0.0644] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000 | 0.05 | 4,000 | 199 | 0.0498 | at or under delta | 0.54 | [0.0432, 0.0569] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate 0.05, n 225 of 4000 | 0.05 | 4,000 | 220 | 0.0550 | unresolved, above delta | 0.08 | [0.0481, 0.0625] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate 0.013, n 225 of 4000 | 0.05 | 4,000 | 211 | 0.0527 | unresolved, above delta | 0.22 | [0.0460, 0.0601] |
| PPBoot, percentile limit, basic | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate pool, n 100 of 2000 | 0.05 | 4,000 | 221 | 0.0553 | unresolved, above delta | 0.07 | [0.0484, 0.0628] |
| PPBoot, percentile limit, power-tuned | judge-logit cells, unstratified, n 100-1,000 | fails | refusal rubric 0, rate pool, n 500 of 2000 | 0.05 | 4,000 | 217 | 0.0542 | unresolved, above delta | 0.12 | [0.0474, 0.0617] |
| PPBoot, percentile limit, power-tuned | judge-logit cells, unstratified, n 100-1,000 | fails | refusal raw 0, rate pool, n 500 of 2000 | 0.05 | 4,000 | 226 | 0.0565 | unresolved, above delta | 0.034 | [0.0495, 0.0641] |
| Clopper-Pearson over pairs | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 167 | 0.0418 | at or under delta | 0.99 | [0.0358, 0.0484] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | Meta-SecAlign-70B | 0.05 | 4,000 | 3 | 0.0008 | at or under delta | 1 | [0.0002, 0.0022] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | Meta-SecAlign-70B-repeat_user_prompt | 0.05 | 4,000 | 3 | 0.0008 | at or under delta | 1 | [0.0002, 0.0022] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 110 | 0.0275 | at or under delta | 1 | [0.0227, 0.0331] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | command-r | 0.05 | 4,000 | 58 | 0.0145 | at or under delta | 1 | [0.0110, 0.0187] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | command-r-plus | 0.05 | 4,000 | 30 | 0.0075 | at or under delta | 1 | [0.0051, 0.0107] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | gpt-3.5-turbo-0125 | 0.05 | 4,000 | 197 | 0.0493 | at or under delta | 0.6 | [0.0428, 0.0564] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt | 0.05 | 4,000 | 138 | 0.0345 | at or under delta | 1 | [0.0291, 0.0406] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | Meta-SecAlign-70B | 0.05 | 4,000 | 56 | 0.0140 | at or under delta | 1 | [0.0106, 0.0181] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | Meta-SecAlign-70B-repeat_user_prompt | 0.05 | 4,000 | 71 | 0.0177 | at or under delta | 1 | [0.0139, 0.0223] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 85 | 0.0213 | at or under delta | 1 | [0.0170, 0.0262] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | command-r-plus | 0.05 | 4,000 | 169 | 0.0423 | at or under delta | 0.99 | [0.0362, 0.0490] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | 24 cells, 0-0.0488 | 0.05 | 4,000 | | | 0 unresolved, 24 at or under | | |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | Meta-SecAlign-70B | 0.05 | 4,000 | 12 | 0.0030 | at or under delta | 1 | [0.0016, 0.0052] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | Meta-SecAlign-70B-repeat_user_prompt | 0.05 | 4,000 | 16 | 0.0040 | at or under delta | 1 | [0.0023, 0.0065] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | claude-3-5-sonnet-20240620 | 0.05 | 4,000 | 146 | 0.0365 | at or under delta | 1 | [0.0309, 0.0428] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 2 | 0.0005 | at or under delta | 1 | [0.0001, 0.0018] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | command-r | 0.05 | 4,000 | 127 | 0.0318 | at or under delta | 1 | [0.0265, 0.0377] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | command-r-plus | 0.05 | 4,000 | 119 | 0.0297 | at or under delta | 1 | [0.0247, 0.0355] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gemini-1.5-flash-002 | 0.05 | 4,000 | 164 | 0.0410 | at or under delta | 1 | [0.0351, 0.0476] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-3.5-turbo-0125 | 0.05 | 4,000 | 187 | 0.0467 | at or under delta | 0.84 | [0.0404, 0.0538] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4-0125-preview | 0.05 | 4,000 | 159 | 0.0398 | at or under delta | 1 | [0.0339, 0.0463] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4o-2024-05-13-spotlighting_with_delimiting | 0.05 | 4,000 | 208 | 0.0520 | unresolved, above delta | 0.29 | [0.0453, 0.0593] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4o-2024-05-13-tool_filter | 0.05 | 4,000 | 204 | 0.0510 | unresolved, above delta | 0.4 | [0.0444, 0.0583] |
| larger of the two clustered bounds | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | meta-llama_Llama-3-70b-chat-hf | 0.05 | 4,000 | 220 | 0.0550 | unresolved, above delta | 0.08 | [0.0481, 0.0625] |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | fails | 23 cells, 0-0.0420 | 0.05 | 4,000 | | | 0 unresolved, 23 at or under | | |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | 14 cells, 0.0220-0.0512 | 0.05 | 4,000 | | | 1 unresolved, 13 at or under | | |
| two-way bootstrap | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4-0125-preview | 0.05 | 4,000 | 194 | 0.0485 | at or under delta | 0.68 | [0.0420, 0.0556] |
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
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 225 of 20000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | refusal raw 0, rate 0.200, n 1000 of 20000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.200, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.200, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.050, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.050, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.013, n 225 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| the judge's rate alone | spike 017 plasmodes, every cell and feature (the cells of its table B6) | fails | shifted refusal raw 0, rate 0.013, n 1000 of 4000, f01 | 0.05 | 4,000 | 0 | 0 | at or under delta | 1 | [0, 0.0009] |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (a): user tasks resampled, injection tasks fixed | fails | 23 cells, 0-0.0275 | 0.05 | 4,000 | | | 0 unresolved, 23 at or under | | |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (b): injection tasks resampled, user tasks fixed | fails | 20 cells, 0.0155-0.0563 | 0.05 | 4,000 | | | 3 unresolved, 17 at or under | | |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4-0125-preview | 0.05 | 4,000 | 161 | 0.0403 | at or under delta | 1 | [0.0344, 0.0468] |
| multiway cluster variance with a t quantile | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | gpt-4o-2024-05-13 | 0.05 | 4,000 | 203 | 0.0508 | unresolved, above delta | 0.42 | [0.0442, 0.0580] |
| two clustered margins added in quadrature, fresh draws | AgentDojo, 28 pipelines, scheme (c): both resampled | fails | 27 cells, 0-0.0508 | 0.05 | 4,000 | | | 1 unresolved, 26 at or under | | |
| stratified Wald-t `b1` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | holds | C2:unsafe, n_s 200, step 100 | 0.05 | 10,000 | 504 | 0.0504 | unresolved, above delta | 0.43 | [0.0462, 0.0549] |
| stratified Wald-t `b1` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | fails | 13 cells, 0-0.0568 | 0.05 | 4,000 | | | 5 unresolved, 8 at or under | | |
| stratified Wald-t `b1` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | fails | 13 cells, 0.0855-0.1045 | 0.1 | 4,000 | | | 5 unresolved, 8 at or under | | |
| `b1w` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | fails | 17 cells, 0-0.0515 | 0.05 | 4,000 | | | 2 unresolved, 15 at or under | | |
| `b1w` on a synthetic i.i.d. grid | six strata profiles, n_s 100-400, binomial strata | fails | 14 cells, 0.0855-0.1062 | 0.1 | 4,000 | | | 5 unresolved, 9 at or under | | |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (b_s) | fails | 17 cells, 0-0.0522 | 0.05 | 4,000 | | | 1 unresolved, 16 at or under | | |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | Meta-SecAlign-70B | 0.05 | 4,000 | 31 | 0.0077 | at or under delta | 1 | [0.0053, 0.0110] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | Meta-SecAlign-70B-repeat_user_prompt | 0.05 | 4,000 | 42 | 0.0105 | at or under delta | 1 | [0.0076, 0.0142] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 144 | 0.0360 | at or under delta | 1 | [0.0304, 0.0422] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | command-r | 0.05 | 4,000 | 175 | 0.0437 | at or under delta | 0.97 | [0.0376, 0.0506] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | command-r-plus | 0.05 | 4,000 | 90 | 0.0225 | at or under delta | 1 | [0.0181, 0.0276] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | gemini-1.5-pro-002 | 0.05 | 4,000 | 209 | 0.0522 | unresolved, above delta | 0.27 | [0.0456, 0.0596] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | gemini-2.0-flash-001 | 0.05 | 4,000 | 91 | 0.0227 | at or under delta | 1 | [0.0184, 0.0279] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | gemini-2.0-flash-exp | 0.05 | 4,000 | 215 | 0.0537 | unresolved, above delta | 0.15 | [0.0470, 0.0612] |
| cluster bootstrap-t, by user task | AgentDojo, 28 pipelines, scheme (c_s) | fails | meta-llama_Llama-3.3-70B-Instruct | 0.05 | 4,000 | 148 | 0.0370 | at or under delta | 1 | [0.0314, 0.0433] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | Meta-SecAlign-70B | 0.05 | 4,000 | 78 | 0.0195 | at or under delta | 1 | [0.0154, 0.0243] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | Meta-SecAlign-70B-repeat_user_prompt | 0.05 | 4,000 | 101 | 0.0253 | at or under delta | 1 | [0.0206, 0.0306] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | claude-3-5-sonnet-20241022 | 0.05 | 4,000 | 70 | 0.0175 | at or under delta | 1 | [0.0137, 0.0221] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | command-r | 0.05 | 4,000 | 214 | 0.0535 | unresolved, above delta | 0.16 | [0.0467, 0.0609] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | command-r-plus | 0.05 | 4,000 | 137 | 0.0343 | at or under delta | 1 | [0.0288, 0.0404] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | gemini-1.5-pro-002 | 0.05 | 4,000 | 168 | 0.0420 | at or under delta | 0.99 | [0.0360, 0.0487] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | gemini-2.0-flash-001 | 0.05 | 4,000 | 85 | 0.0213 | at or under delta | 1 | [0.0170, 0.0262] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | gemini-2.0-flash-exp | 0.05 | 4,000 | 202 | 0.0505 | unresolved, above delta | 0.45 | [0.0439, 0.0577] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | meta-llama_Llama-3.3-70B-Instruct | 0.05 | 4,000 | 132 | 0.0330 | at or under delta | 1 | [0.0277, 0.0390] |
| t_user, a fitted crossed model | AgentDojo, 28 pipelines, tables drawn from a fitted crossed model (P_c) | fails | meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt | 0.05 | 4,000 | 214 | 0.0535 | unresolved, above delta | 0.16 | [0.0467, 0.0609] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 0 | 0.05 | 40,000 | 1,172 | 0.0293 | at or under delta | 1 | [0.0277, 0.0310] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 0 | 0.05 | 40,000 | 1,219 | 0.0305 | at or under delta | 1 | [0.0288, 0.0322] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 100 | 0.05 | 40,000 | 1,409 | 0.0352 | at or under delta | 1 | [0.0334, 0.0371] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 100 | 0.05 | 40,000 | 1,496 | 0.0374 | at or under delta | 1 | [0.0356, 0.0393] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 200 | 0.05 | 40,000 | 1,370 | 0.0343 | at or under delta | 1 | [0.0325, 0.0361] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 200 | 0.05 | 40,000 | 1,513 | 0.0378 | at or under delta | 1 | [0.0360, 0.0397] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 200, step 0 | 0.05 | 40,000 | 2,078 | 0.0520 | unresolved, above delta | 0.038 | [0.0498, 0.0542] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 200, step 100 | 0.05 | 40,000 | 1,970 | 0.0493 | at or under delta | 0.76 | [0.0471, 0.0514] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 0 | 0.1 | 40,000 | 2,669 | 0.0667 | at or under delta | 1 | [0.0643, 0.0692] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 0 | 0.1 | 40,000 | 2,844 | 0.0711 | at or under delta | 1 | [0.0686, 0.0737] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 100 | 0.1 | 40,000 | 3,110 | 0.0777 | at or under delta | 1 | [0.0751, 0.0804] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 100 | 0.1 | 40,000 | 3,268 | 0.0817 | at or under delta | 1 | [0.0790, 0.0844] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 100, step 200 | 0.1 | 40,000 | 3,033 | 0.0758 | at or under delta | 1 | [0.0732, 0.0785] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C2:refusal, n_s 200, step 200 | 0.1 | 40,000 | 3,336 | 0.0834 | at or under delta | 1 | [0.0807, 0.0862] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 200, step 0 | 0.1 | 40,000 | 3,910 | 0.0978 | at or under delta | 0.93 | [0.0949, 0.1007] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 100, step 100 | 0.1 | 40,000 | 4,109 | 0.1027 | unresolved, above delta | 0.036 | [0.0998, 0.1057] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 200, step 100 | 0.1 | 40,000 | 3,877 | 0.0969 | at or under delta | 0.98 | [0.0940, 0.0999] |
| StratPPI's normal limit, redrawn with replacement | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | C3:refusal, n_s 200, step 200 | 0.1 | 40,000 | 3,991 | 0.0998 | at or under delta | 0.56 | [0.0969, 0.1028] |
| `b1w` in the training loop, reference-rate strata | synthetic bandit, 4 heterogeneity levels, `SeldonianLLMPolicy` | holds | icc05, strat_ref, b1w_strat_pop | 0.1 | 500 | 58 | 0.1160 | unresolved, above delta | 0.13 | [0.0893, 0.1474] |
| StratPPI as published (normal limit) | 5 mid-rate labels, reference-rate strata, n_s 100-200, by checkpoint | fails | 21 cells, 0.0636-0.1082 | 0.1 | 5,000 | | | 3 unresolved, 18 at or under | | |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | holds | C2:refusal, n_s 100, step 100 | 0.05 | 10,000 | 521 | 0.0521 | unresolved, above delta | 0.17 | [0.0478, 0.0566] |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | holds | C2:refusal, n_s 200, step 100 | 0.05 | 10,000 | 513 | 0.0513 | unresolved, above delta | 0.28 | [0.0471, 0.0558] |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | holds | C2:refusal, n_s 100, step 200 | 0.05 | 10,000 | 525 | 0.0525 | unresolved, above delta | 0.13 | [0.0482, 0.0571] |
| `b1w` with the term for a sampled pool | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | holds | C2:refusal, n_s 200, step 200 | 0.05 | 10,000 | 506 | 0.0506 | unresolved, above delta | 0.4 | [0.0464, 0.0551] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal, n_s 100, step 0 | 0.05 | 10,000 | 446 | 0.0446 | at or under delta | 0.99 | [0.0406, 0.0488] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal, n_s 100, step 100 | 0.05 | 10,000 | 318 | 0.0318 | at or under delta | 1 | [0.0284, 0.0354] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal, n_s 100, step 200 | 0.05 | 10,000 | 480 | 0.0480 | at or under delta | 0.83 | [0.0439, 0.0524] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 0 | 0.05 | 10,000 | 371 | 0.0371 | at or under delta | 1 | [0.0335, 0.0410] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 200, step 0 | 0.05 | 10,000 | 506 | 0.0506 | unresolved, above delta | 0.4 | [0.0464, 0.0551] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 100 | 0.05 | 10,000 | 272 | 0.0272 | at or under delta | 1 | [0.0241, 0.0306] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 200, step 100 | 0.05 | 10,000 | 478 | 0.0478 | at or under delta | 0.85 | [0.0437, 0.0522] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 200 | 0.05 | 10,000 | 239 | 0.0239 | at or under delta | 1 | [0.0210, 0.0271] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 200, step 200 | 0.05 | 10,000 | 433 | 0.0433 | at or under delta | 1 | [0.0394, 0.0475] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal (014), n_s 100, step 100 | 0.05 | 10,000 | 388 | 0.0388 | at or under delta | 1 | [0.0351, 0.0428] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal (014), n_s 100, step 200 | 0.05 | 10,000 | 486 | 0.0486 | at or under delta | 0.75 | [0.0445, 0.0530] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 0 | 0.05 | 10,000 | 498 | 0.0498 | at or under delta | 0.54 | [0.0456, 0.0542] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 100 | 0.05 | 10,000 | 520 | 0.0520 | unresolved, above delta | 0.19 | [0.0477, 0.0565] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 200 | 0.05 | 10,000 | 352 | 0.0352 | at or under delta | 1 | [0.0317, 0.0390] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 0 | 0.05 | 10,000 | 329 | 0.0329 | at or under delta | 1 | [0.0295, 0.0366] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 200, step 0 | 0.05 | 10,000 | 516 | 0.0516 | unresolved, above delta | 0.24 | [0.0473, 0.0561] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 100 | 0.05 | 10,000 | 408 | 0.0408 | at or under delta | 1 | [0.0370, 0.0449] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 200 | 0.05 | 10,000 | 429 | 0.0429 | at or under delta | 1 | [0.0390, 0.0471] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal, n_s 100, step 0 | 0.1 | 10,000 | 1,001 | 0.1001 | unresolved, above delta | 0.49 | [0.0943, 0.1061] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal, n_s 100, step 200 | 0.1 | 10,000 | 1,057 | 0.1057 | unresolved, above delta | 0.031 | [0.0997, 0.1119] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 0 | 0.1 | 10,000 | 988 | 0.0988 | at or under delta | 0.66 | [0.0930, 0.1048] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 100 | 0.1 | 10,000 | 758 | 0.0758 | at or under delta | 1 | [0.0707, 0.0812] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 200, step 100 | 0.1 | 10,000 | 992 | 0.0992 | at or under delta | 0.61 | [0.0934, 0.1052] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 200 | 0.1 | 10,000 | 644 | 0.0644 | at or under delta | 1 | [0.0597, 0.0694] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 200, step 200 | 0.1 | 10,000 | 1,060 | 0.1060 | unresolved, above delta | 0.024 | [0.1000, 0.1122] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 0 | 0.1 | 10,000 | 767 | 0.0767 | at or under delta | 1 | [0.0716, 0.0821] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal (014), n_s 100, step 100 | 0.1 | 10,000 | 1,060 | 0.1060 | unresolved, above delta | 0.024 | [0.1000, 0.1122] |
| `b1w` without the term | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C1:refusal (014), n_s 100, step 200 | 0.1 | 10,000 | 1,031 | 0.1031 | unresolved, above delta | 0.15 | [0.0972, 0.1092] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:unsafe, n_s 100, step 200 | 0.1 | 10,000 | 939 | 0.0939 | at or under delta | 0.98 | [0.0883, 0.0998] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 0 | 0.1 | 10,000 | 811 | 0.0811 | at or under delta | 1 | [0.0758, 0.0866] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 200, step 0 | 0.1 | 10,000 | 1,050 | 0.1050 | unresolved, above delta | 0.05 | [0.0991, 0.1112] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 100 | 0.1 | 10,000 | 924 | 0.0924 | at or under delta | 0.99 | [0.0868, 0.0982] |
| StratPPI estimator with a bootstrap-t limit | a pool redrawn from its source, strata rebuilt, 5 mid-rate labels, n_s 100-200 | fails | C2:refusal, n_s 100, step 200 | 0.1 | 10,000 | 922 | 0.0922 | at or under delta | 1 | [0.0866, 0.0980] |
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
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 48 | 0.0120 | at or under delta | 1 | [0.0089, 0.0159] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | fails | wording 2 | 0.05 | 4,000 | 151 | 0.0377 | at or under delta | 1 | [0.0321, 0.0441] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 6 | 0.0015 | at or under delta | 1 | [0.0006, 0.0033] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 32 | 0.0080 | at or under delta | 1 | [0.0055, 0.0113] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 100, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 3 | 0.0008 | at or under delta | 1 | [0.0002, 0.0022] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 200, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 67 | 0.0168 | at or under delta | 1 | [0.0130, 0.0212] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 200, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 70 | 0.0175 | at or under delta | 1 | [0.0137, 0.0221] |
| carried Youden-corrected bound | training, constrained (014 s1): step 0 -> step 200, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 88 | 0.0220 | at or under delta | 1 | [0.0177, 0.0270] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 100, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 187 | 0.0467 | at or under delta | 0.84 | [0.0404, 0.0538] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 100, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 126 | 0.0315 | at or under delta | 1 | [0.0263, 0.0374] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 200, six judge wordings | fails | wording 1 | 0.05 | 4,000 | 164 | 0.0410 | at or under delta | 1 | [0.0351, 0.0476] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 200, six judge wordings | fails | wording 3 | 0.05 | 4,000 | 57 | 0.0143 | at or under delta | 1 | [0.0108, 0.0184] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 200, six judge wordings | fails | wording 4 | 0.05 | 4,000 | 111 | 0.0278 | at or under delta | 1 | [0.0229, 0.0333] |
| carried Youden-corrected bound | training, constrained (014 s2): step 0 -> step 200, six judge wordings | fails | wording 5 | 0.05 | 4,000 | 193 | 0.0483 | at or under delta | 0.7 | [0.0418, 0.0554] |

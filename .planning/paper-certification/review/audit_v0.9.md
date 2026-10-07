# AUDIT: where the paper and its sources differ

Paper under audit: `reports/paper_certification_clean.md` (draft of 2026-10-06; line numbers
below are this file's). Sources: the files the working draft's tag table names, under
`/home/hannanabdul/seldonian-fairness/`. Nothing listed here was fixed in the artifact: claims,
tables and formulas in the artifact are the paper's.

## How the audit was done

- **Sections 2 to 6** (new, unaudited): every factual statement, Tables 1 to 3 and every formula
  were checked against `seldonian/llm/policy.py`, `seldonian/llm/rewards.py`, sections 2 to 5 of
  `reports/paper_seldonian_llm.md`, the analysis code the formulas describe
  (`stratbounds.py`, `cert017.py`, `scripts/stratppi_baseline.py`, `scripts/agentdojo_recheck.py`,
  `scripts/p9_certificate.py`, `scripts/validity_recount.py`), and the compiler's knowledge of
  Thomas et al. (2019). The checked statements that agree are listed in
  `evidence/tables/derived_source_ledger_sections_2_to_6.md`.
- **Sections 7 to 10 and the abstract**: 170 numbers or table rows were checked (58 in section 7,
  including all 24 rows of Table 4; 112 in sections 9, 10 and the abstract), more than the 30
  asked for, because the sources are few and tabular.
- **Appendices A to C and section 8**: every number (45 entries; Table A1's 60 values count as
  one).
- The checks of sections 7 to 10 and the appendices were run by three delegated read-only
  sub-audits; every item below that came from one was re-read in the source by the compiler
  before it was listed, except where the entry says "sub-audit recount" (a recomputation from a
  data file that the compiler did not repeat).
- Files under `.planning/paper-certification/` were not read. No earlier artifact or review was
  consulted.

## Counts

| kind | items |
|---|---|
| (a) the paper's statement disagrees with its source, or says more or something other than the source supports | 45 (A1 to A45) |
| of which: a different value or count | 6 (A1, A14, A16, A17, A33, A38) |
| of which: the paper's wording covers more than the source row (scope) | 20 (A9, A10, A11, A18, A19, A20, A21, A23, A24, A25, A29, A30, A31, A34, A35, A36, A37, A39, A40, A45) |
| of which: the paper disagrees with itself (two places, one source) | 7 (A8, A13, A22, A26, A27, A28, A43) |
| of which: the formula or description differs from the code or the runs | 9 (A2, A3, A4, A5, A6, A7, A12, A15, A44) |
| of which: rounding or presentation | 3 (A32, A41, A42) |
| (b) no source could be traced | 11 (B1 to B11) |

Severity is the compiler's reading of how much the item bears on a claim: **high** (changes what
a claim says), **medium** (changes its scope or a reader's check of it), **low** (detail).

## Part A: disagreements with the source

### Sections 2 to 6 (checked in full)

**A1 (medium). The label budget for a 1% threshold: 299 in the paper, 301 in the tagged source.**
- Paper, line 265: "`n >= ln(delta) / ln(1 - tau)`: 299 labels for 1% at delta 0.05, 149 for 2%, 59 for 5%." Repeated at lines 705-707 ("299 gold labels ... certify a 1% threshold").
- Source (tag `[017 6]`), `.planning/spikes/017-calibration-carrying-certificate/README.md` line 258: "301 labels with no positive certify 1%, 149 certify 2%, 59 certify 5%". Also `reports/state_2026-10-01.md` line 51: "301 human labels on the certified policy's own responses certify a 1% threshold".
- Note: the paper's formula does give 299 (`1 - 0.05^(1/299) = 0.009969`, `1 - 0.05^(1/298) = 0.0100024`), and `.planning/spikes/019-external-trace-certificate/README.md` line 104 says 299. The disagreement is with the source the number is tagged to.

**A2 (medium). "a separate step size, usually smaller, when the constraint has slack".**
- Paper, line 202.
- Source: `seldonian/llm/rewards.py` line 202, `self.eta_down = eta if eta_down is None else float(eta_down)` (default: the same step). `scripts/run_round6.sh` line 3: "setting: lam0 5, lam_floor 5, eta_down = eta." `.planning/spikes/014-pushed-label-stratification/gen014.py` lines 134-135: `LagrangianReward(..., lam0=5.0, eta=100.0, lam_max=50.0, lam_floor=5.0)` (no `eta_down`). The companion paper's update (line 263-264) has one step size. A smaller downward step appears only in the synthetic spike 010 (`.planning/spikes/010-lam0-no-floor-anomaly/README.md` line 53: "lam0 5, eta_down 10").
- So in the language-model runs the paper cites, the step under slack was equal, not smaller.

**A3 (low). Table 1 lists "a rubric" among the judges "as implemented in `seldonian/llm/`".**
- Paper, lines 97-98 and 108.
- Source: `seldonian/llm/judges.py` defines `KeywordRefusalJudge`, `LengthJudge`, `ExactMatchJudge`, `RefusalClassifierJudge`, `LlamaGuardJudge`, `Qwen3GuardJudge`; no file in `seldonian/llm/` contains the word "rubric". The rubric judge is `.planning/spikes/015-prompted-judge-fidelity/rubric015.py`. The companion paper's judge table (lines 184-191) has no rubric.

**A4 (low). "the runs cited here, after the first, use `n_eff` with `kappa = 1`".**
- Paper, line 193.
- Source: `reports/paper_seldonian_llm.md` lines 250-251: "Round 1 inherited a factor of 2 from the classification code; Rounds 1c-5 used 1.0 at the effective size"; lines 352-354: in Round 1b "the predicted bound had ignored the prediction sample's own variance, the origin of the effective-`n` rule". Round 1b, the second round, did not use `n_eff`.

**A5 (low). `n_eff = 1 / (1/m + 1/n_s)` with `m` candidate prompts and `n_s` "the number of safety prompts".**
- Paper, lines 186-190.
- Source: `seldonian/llm/policy.py` line 35, `return max(int(1.0 / (1.0 / m + 1.0 / n_s)), 2)` (truncated to an integer, floor of 2); line 256, `n=effective_n(len(idx), n_s)` with `idx = c.select(records)` and `n_s = self.n_safety(c)`: both counts are of the prompts in the constraint's group, not of all prompts.

**A6 (low). The multiplier is clipped to `[floor, lambda_max]`.**
- Paper, line 199.
- Source: `seldonian/llm/rewards.py` line 222: `floor = self.lam_floor if (self.floor_always or self.bound_seen[name]) else 0.0`. The floor applies only after the constraint has once been predicted infeasible, unless `floor_always`. The companion paper (line 263-264) writes the clip as `(…, 0, lambda_max)`.

**A7 (medium). The ESS formula uses the distance from estimate to limit; the StratPPI comparison computes it from the distance from the truth to the limit.**
- Paper, lines 454-459: "with `w` the mean distance from estimate to limit, `ESS = ( w_R / w_A )^2`."
- Source: `scripts/stratppi_baseline.py` line 110, `excess=float(ub.mean() - truth)`, and line 229, `(base['excess'] / x['excess']) ** 2`; `results/paper/stratppi.md` line 4: "ESS: (mean bound minus truth, for the block's first arm) / (the same for this arm), squared". Spike 013 does use the width: `summarise_plasmode.py` line 54, `ess = (base["width"] / x["width"]) ** 2`.
- Affects every ESS in Table A1, in items 2 and 3 of section 8.1's StratPPI comparison and in section 8.2's medians. Also, in section 8.2 the baseline is "the labels alone" with Clopper-Pearson, not "the pooled bound of the same family" (line 455).

**A8 (medium). "Replications per cell are between 1,000 and 5,000; the caption of Table 4 gives them by row."**
- Paper, lines 516-517.
- The caption (line 530) says only "Draws per cell are between 1,000 and 5,000 (4,000 for AgentDojo)"; it gives nothing by row. Section 7.2 reports studies outside the range: "500 runs per cell" (line 602); the synthetic rows are "500-1,000 independent trials" (`reports/paper_seldonian_llm.md` line 405); the trajectory result is 200 runs per cell (`results/paper/validity_recount.md` line 254: "At 200 runs a cell the check resolves only a miss above 0.1424"); the design check of section 10.4 is 2,000 repetitions (`results/labels/p9/design_check.md` line 3).

**A9 (low). Judge pools: "its mean is the truth"; Table 3 gives the truth as "pool mean".**
- Paper, lines 369-381 and Table 3 row 4.
- Source: `.planning/spikes/017-calibration-carrying-certificate/plasmode017.py` line 113, `truth = float(y_pool.mean()) if rate is None else rate`, and lines 103-106: in the prevalence-shifted cells the label is drawn as Bernoulli(`rate`) and the score from the pool's scores with that label. The cells at rates 0.2, 0.05 and 0.013 that sections 8.2 and 9.4 quote ("n 225 at a 1.3% rate") are of this kind: the truth is an imposed rate, and the population is the pool reweighted.

**A10 (medium). Labelling sheets: "A sheet is re-drawn by its real sampling rule".**
- Paper, lines 382-383; Table 4 rows 8 and 23; line 803.
- Source: `.planning/spikes/017-calibration-carrying-certificate/results.md` lines 320-322: "Validity under the real sampling rule (planted labels) ... Labels planted on the real population (harm only among answered responses, the wording's own 0/1 judge catching 2/3 of it), the sheet re-drawn 4,000 times (20 plantings x 200 draws) by the real rule." The labels in the sheet study are planted, not the annotator's.
- Also `results/paper/validity_recount.md` line 185: the route "does not read the judge's feature, so the source's second row for the same wording and rate is a second run of the same setting". The 18 cells of Table 4 row 8 are 9 settings each run twice, and the "second, independent run of the same cell" (line 564) is itself one of the 18 cells; the two runs together give "0.0540 (432 of 8,000; p = 0.054 ...), unresolved, above delta".

**A11 (low). The cap `1 / (1 - G + G n_s / N)` "When the safety set is a large share of a pool of `N` prompts".**
- Paper, line 481 and line 639.
- Source: `.planning/spikes/013-stratified-safety-set/README.md` lines 73-74: "With the pool itself a sample from the population, the gain becomes `1 / (1 - G + G n_s / N)`." The cap is for a population target; the same README (line 171) says of the real data: "Not tested: ... a population target on real data (the plasmode certifies the pool)".

**A12 (low). Table 3, row "judge routing (8.2)": "500 real responses".**
- Source: `.planning/spikes/017-calibration-carrying-certificate/README.md` line 209: "harm, 225 human labels of 4,800". The refusal and brevity pools have 500 scored responses; the harm sheet in the same row is drawn from 4,800.

**A13 (low). The Wilson bound "puts the variance at the hypothesised rate and not at the estimate, which protects it at low rates".**
- Paper, lines 396-397.
- The paper's own Table 4 (line 549): "`b1w` and the pooled Wilson bound at rare rates | fail | harm labels at 1-2%, n_s 100, any design | 0.05 | 0.076-0.448 | 12 / 0 / 0". Source: `.planning/spikes/013-stratified-safety-set/README.md` lines 58-59: "In rare-rate profiles (0-7%), pooled and stratified alike overshoot in some cells (up to 0.122 at delta 0.05)."

**A14 (high). "Qwen3Guard-4B, the guard behind every refusal number above".**
- Paper, line 874; section 6.6 (line 516): "The guard is Qwen3Guard (Zhao et al., 2025)."
- Source: the refusal numbers cited from the companion paper (section 7.2's over-refusal seeds, lines 606-608; Table 2's 0.133 against 0.184; "2 of 3 on over-refusal", line 266) come from a different guard. `reports/paper_seldonian_llm.md` line 187: "`Qwen/Qwen3Guard-Gen-0.6B`, Refusal field = Yes | model | refusal rate"; `scripts/run_llm_rl.py` line 68: "qwen3guard (0.6B; rounds 1-6)"; `seldonian/llm/judges.py` line 441, `"qwen3guard_refusal": lambda **kw: Qwen3GuardJudge(field="refusal", **kw)`, whose default (line 343) is `model_name="Qwen/Qwen3Guard-Gen-0.6B"`. The companion paper (line 889) calls this "the paper's weakest point: the judge is a 0.6B guard model".
- The 4B guard is the one in spikes 013, 014 and 017 (`gen014.py` line 132: `model_name="Qwen/Qwen3Guard-Gen-4B", quant4=True`), and there it is 4-bit quantised, which the paper does not say.
- So the human-label check of section 10.3 is of the 4B guard, and does not cover the guard behind the companion-paper refusal results the paper cites.

**A15 (low). Bootstrap-t limit: details of the code that the formula does not give.**
- Paper, lines 425-435 ("`B` times"; "recompute the estimate").
- Source: `cert017.py` line 142 computes `var(f)` for `lambda` from labelled and unlabelled scores together, while each bootstrap replicate (line 199) uses the unlabelled variance only; `boots=300` by default (line 181) and in `stratppi_baseline.py` (line 80); `scripts/p9_certificate.py` line 53 sets `BOOTS = 10000` for the prepared certificate and its design check ran with 300 (`design_check.md` line 3). The paper gives no value of `B` for sections 8 and 10.4.

**A44 (low). The predicted test is "computed on data the optimiser and the selection rule have both seen"; Table 2: "candidate-side samples the optimiser and the selection saw".**
- Paper, lines 210-212 and 220.
- Source: `seldonian/llm/policy.py` lines 297-307 and 320-327: each predicted test draws a fresh group-stratified random subset of `D_c` and samples a fresh response to each prompt. The prompts are the optimiser's; the judged responses are new draws that the optimiser did not train on.

### Sections 7 to 10 and the abstract (spot-checked)

**A16 (medium). "lower after training in 5 of 6 wordings".**
- Paper, line 768.
- Source: `results/labels/refusal/rubric_vs_human.md` lines 9-14. Recall, reference then trained: wording 0 "3/18" and "0/18"; 1 "3/18" and "0/18"; 2 "3/18" and "0/18"; 3 "5/18" and "1/18"; 4 "11/18" and "6/18"; 5 "0.43; 7/18" and "0.40; 6/18". Lower in all six.

**A17 (medium). "One pipeline of 28 certifies, and it is the same one under every clustered rule we tried."**
- Paper, line 860.
- Source: `results/paper/agentdojo_recheck.md` lines 187 and 190: "| t(inj) | 3 | Meta-SecAlign-70B, Meta-SecAlign-70B-repeat_user_prompt, claude-3-5-sonnet-20241022 |" and "| two-way | 3 | Meta-SecAlign-70B, Meta-SecAlign-70B-repeat_user_prompt, claude-3-5-sonnet-20241022 |". The bootstrap by injection task and the two-way bootstrap, both named at lines 841-842, pass three pipelines. Table 7's own cell for Meta-SecAlign-70B clustered by injection task is 0.035, under 0.05.

**A18 (high). The design check "puts the miss rates of the four limits at or under 0.045 against a level of 0.05".**
- Paper, lines 1001-1003.
- Source: `results/labels/p9/design_check.md` lines 7-14. Scheme "pool": 0.001, 0.040, 0.045, 0.042. Scheme "new prompts": 0.001, 0.051, 0.063, 0.052. The 0.045 is the largest under "pool" only. Limit (b2), the one built "for new prompts", misses 0.042 under "pool" and 0.052 under the scheme it is for. Appendix C (line 1194) adds "for the pool rate"; section 10.4 does not, and neither reports the second scheme. (The file notes that (b1) "is not built for that row"; Monte Carlo se 0.005.)

**A19 (low). The check's synthetic labels are "at the rates of section 10.3".**
- Paper, lines 1001-1002 and 1192-1193.
- Source: `results/labels/p9/design_check.md` line 3: "with the first sheet's rates per logit bin (below 0: 0.002; 0 to 10: 0.165; above 10: 0.762). In this synthetic population the reference rate is 0.095, the trained rate 0.101, the true difference +0.0063." Section 10.3's rates are 10.7% and 10.6% (difference -0.1 points).

**A20 (high). "The real training loop. With our `SeldonianLLMPolicy` in the loop, 500 runs per cell ... the same as a random split with a pooled bound."**
- Paper, lines 602-604.
- Source: `.planning/spikes/013-stratified-safety-set/inloop.py` lines 3-4: "The real SeldonianLLMPolicy (Lagrangian GRPO, spike 001's tdlab backend) trains on D_c of each split, on ``HeteroEnv`` at four heterogeneity levels (ICC_ref 0.05-0.75)", with "quantile strata of an 8-sample reference rate, H = 4". The policy class is the real one; the environment is a synthetic bandit, with 4 strata, and the paragraph follows one headed "Synthetic environment" without saying so.
- `results/paper/validity_recount.md` line 249: "The random split with a pooled bound is 0.0900-0.1000, all four at or under: 'the same' is 'not distinguishable at 500 runs', and the stratified test has the one cell above delta."

**A21 (medium). "0.000-0.002 ... for the Seldonian arm under four bounds ... it stayed at or below 0.002 across reward pressures 0 to 4 and sample sizes 200 to 5,000."**
- Paper, lines 591-594.
- Source: `reports/paper_seldonian_llm.md` lines 427-436 and 443-451. These are two one-way sweeps of the Lagrangian arm (`seldonian_lag`): pressure at n = 1,000, and sample size at pressure 1 "t bound". The other Seldonian variant in the same table is higher: "| filter only | 200 | 0.08 | 0.028 | 0.333 |", 0.004 at 1,000 and 5,000; and under "Stress at margin 0" (lines 476-479) "the joint unsafe rate is 1.2% (95% upper 1.9%)". All are under delta 0.1; the paper's range is the Lagrangian arm's only.

**A22 (medium). "At 1-2% and n_s 100 the Wilson-type bounds, pooled or stratified, missed in 24-44% of draws".**
- Paper, lines 570-571; Table 4 line 549 gives the same cells as "0.076-0.448".
- Source: `results/paper/validity_recount.md` line 232: "By checkpoint they run 0.0758-0.4476; 24-44% is the stratified bound's largest checkpoint for each label (0.2382 and 0.4432; the pooled bound's are 0.2396 and 0.4476). At n_s 200, which the row leaves out, `b1w` is over for the 1% label ... and at or under delta for the 2% label".

**A23 (low). The t-test "is anti-conservative exactly where trained policies sit (harm rates of 3-6%)".**
- Paper, lines 574-575.
- Source: `reports/paper_seldonian_llm.md` line 381: one pool, "rate 0.034". No rate between 3.4% and 6% was resampled. At n 1,200 and 2,400 the t bound's miss is 0.101, which Table 4 itself counts as unresolved (3 / 2 / 0).

**A24 (low). "65% of plain-Lagrangian runs passed it after a step inside the region. A certificate at level delta/T ... held".**
- Paper, lines 614-616.
- Source: `.planning/spikes/004-forbidden-capability/README.md` lines 93-94: "65% of runs *passed the safety test and had a training step in U* (50% at 10-step checks)". 65% is one of two check intervals. `results/paper/validity_recount.md` line 254: "All 10 cells at or under delta, the largest exactly at it (20 of 200 ...). At 200 runs a cell the check resolves only a miss above 0.1424, so 'held' is 'at or under delta in every cell, at low resolution'."

**A25 (low). "Rerandomised and stratified candidate/safety splits never broke the test".**
- Paper, lines 598-600.
- Source: `results/paper/validity_recount.md` line 86: "| tight Wald test after an adversarial split | same runs ... | 0.05 | [012] | not quoted in the paper | - | 42 | 2,000 | 84,000 | 0.0288 (at or under delta) | 0.0550 [0.0454, 0.0659] | 0 | 3 | 39 | 0 |". For the tight Wald test of the same runs 3 of 42 cells are unresolved above delta. The Clopper-Pearson sentence (at most 1 of 2,000) is as the source says.

**A26 (low). Table 4, row "cluster bootstrap-t by user task, injection tasks also sampled | AgentDojo, schemes (b); (c)".**
- Paper, line 556.
- Source: `scripts/agentdojo_recheck.py` line 55: `("b", "injection tasks resampled, user tasks fixed")`. Under (b) user tasks are fixed; "also sampled" describes (c) only. The digits (21 / 0 / 7; 24 / 0 / 4) match `results/paper/agentdojo_recheck.md` line 31.

**A27 (low). 28 against 26 reference-strata cells for the same family of arms.**
- Paper, Table 4 lines 547 and 551 (0 / 0 / 28; 9 / 5 / 14) against lines 552-553 (26 cells). The caption explains the 26 for the allocation rows only.
- Source: `results/paper/stratppi_validate.md` lines 113 and 117 count the proportional arms on 26 cells: "| A, H = 8: StratPPI as published, prop | 26 | 9 / 5 / 12 | 0.086 | 3.46 |" and "| A, H = 8: StratPPI estimator, bootstrap-t, prop | 26 | 0 / 0 / 26 | 0.045 | 2.33 |". Table 4 row 6's "4 mid-rate labels (9-94%)" includes the checkpoint the caption says is "at a 95% rate" (sub-audit: all three unresolved cells of that row are that label at n_s 100).

**A28 (medium). Section 8.1's ESS against Table A1's for the same labels.**
- Paper, lines 631-634: "2.4 for over-refusal ..., 5.1-5.3 for refusal of plainly harmful requests ..., 1.4 for non-refusal of encoded requests ... 2.1-2.2" (pushed).
- Paper, Table A1 `b1w` column, lines 1142-1149: 2.39 and 2.30; 4.75 and 5.19; 1.38 and 1.36; 2.19 and 2.13.
- The caption of Table A1 says the rates of the two measurements "differ by a point"; it does not say the ESS differ (5.13 against 4.75 at n_s 100: `.planning/spikes/013-stratified-safety-set/README.md` line 137, "C3 refusal 5.13/5.33", against `results/paper/stratppi.md` line 22, "0.047; ESS 4.75"). Part of the difference is A7.

**A29 (low). "13 of 14 cells over, misses 0.056-0.187".**
- Paper, lines 1169-1170.
- Source: `results/paper/stratppi_validate.md` line 89 (13 / 1 / 0, largest 0.187). Sub-audit recount from `results/paper/stratppi.json`: the 14 misses run 0.05625 to 0.18725 and the 0.05625 cell is the one that is not over; among the 13 over cells the range starts at 0.06125.

**A30 (low). "reproduces the widths of their own simulation".**
- Paper, lines 1131-1132.
- Source: `results/paper/stratppi_validate.md` line 65: "| different variance | opt | 100 | 20, 80 | 0.231 | 0.206 | 0.222 | 0.898 |" (ours 0.231, their Figure 2 0.222). Eight of nine widths agree to the third decimal; line 100: "at the oracle allocation it matches at n 200 and 1,000 and is 0.231 against 0.222 at n 100".

**A31 (low). "It is the narrowest such route with 5 or 10 strata"; "With 20 strata ... `b1w` is the narrower".**
- Paper, lines 717 and 722.
- Source: `results/paper/stratppi_validate.md` line 270, medians by rate (20%, 5%, 1.3%): "bootstrap-t StratPPI K = 5: 2.47, 1.76, 0.54; bootstrap-t StratPPI K = 10: 2.54, 1.87, 0.60; bootstrap-t StratPPI K = 20: 2.51, 1.64, 0.63; `b1w` K = 5: 2.24, 1.26, 1.16; `b1w` K = 10: 2.41, 1.42, 1.20; `b1w` K = 20: 2.60, 1.70, 1.28." The claim holds on the medians at 20% and 5%; at 1.3% `b1w` is narrower at every K. Sub-audit recount by cell: bootstrap-t StratPPI is narrower in 10 of 14 cells at K = 5 and at K = 10; at K = 20 `b1w` is narrower in 8 of 14.

**A33 (low). "of 93 guard-flagged refusals after training it called 89 answers" against recall "0.03".**
- Paper, line 1182 and line 759.
- Source: `.planning/spikes/017-calibration-carrying-certificate/README.md` lines 349-350: "the compiled judge (wording 0) misses 89". Recall 0.03 on 93 refusals is 3 caught, which leaves 90. Sub-audit recount from `results/spikes/017/scores_pop.jsonl`: 3 responses with p > 0.5, 89 with p < 0.5, one at exactly 0.5.

**A34 (low). "Its recall on the guard's positives is 0.04-0.51".**
- Paper, lines 750-751.
- Source: `reports/figs/fig4_carrying.csv` (the paper's Figure 3 data): recall where measured, shift "prompt pool: harmful to benign", wording 4: 0.537; wording 3: 0.472. The range 0.04-0.51 holds for the benign sources; the paragraph also uses the harmful pool.

**A35 (low). "At step 100, where the rate had moved by 3.4 points".**
- Paper, line 762.
- Source: `.planning/spikes/017-calibration-carrying-certificate/README.md` line 317: "steps 100 (rate 0.151, 3.4 points below the reference)". That is spike 014's whole-pool rate; sub-audit: on the 500 scored items the gold rate goes 0.200 to 0.170 (`transfer.md`).

**A36 (medium). "missed in 98% of re-drawn sheets".**
- Paper, line 803.
- Source: `.planning/spikes/017-calibration-carrying-certificate/README.md` lines 238-240: "that route missed in 0.98 of draws at a 1.3% rate for wording 2, 0.09 to 0.70 for wording 4 and at most 0.005 for wording 0." 98% is one wording at one planted rate. Table 4 gives the row as "0-0.989" with 11 of 18 cells at or under delta.

**A37 (low). "on whose released completions the two annotations agree with a kappa of 0.88".**
- Paper, lines 947-948.
- Source: `results/labels/xstest/analysis.md` line 49, "| **pooled** | 0.88 | 0.90 | 0.59 | 0.81 |", under the safe-prompt section (250 per model). 0.88 is for the 1,250 safe-prompt completions; sub-audit: the unsafe-prompt figure in the same file is 0.93.

**A38 (low). "300 prompt pairs ... with 8 further responses per policy on each of the 490 pool prompts ... all 8,820 responses".**
- Paper, lines 992-995.
- 2 x 300 + 2 x 8 x 490 = 8,440. Source: `results/labels/p9/guard.jsonl` has 8,820 rows: 490 `sheet` and 3,920 `extra` per policy (recounted). One sheet response per policy was scored on all 490 prompts, of which 300 pairs were drawn for the sheet.

**A39 (low). Abstract: "about 420 labelled prompt pairs for an even chance of passing if the two policies refuse equally often".**
- Paper, lines 23-24.
- Source: `results/paper/p9_budget.md` line 9, "| 0.02 | 50% | 1,226 | 416 | 167 | 339 |", computed at the measured difference ("difference -0.001"; Table 10's caption: -0.0005). The budget scales as `1 / (margin - difference)^2`, so at exactly equal rates the same formula gives about 437 (derived: 416 x (0.020513 / 0.02)^2). 416 is also the labels-alone, paired column.

**A45 (medium). Section 8.1: "With k = 8 and H = 8 on real Granite-3.3-2B responses"; "The absolute-error criterion we pre-registered".**
- Paper, lines 629-646. The paper reports one departure from the pre-registration (the absolute-error criterion).
- Source: `.planning/spikes/013-stratified-safety-set/README.md` lines 116-121: "The design's rule (ties kept, quantile cuts, strata merged below 20 expected safety prompts) collapses to 1-2 strata, and often to ESS 1.00. Random tie-breaking into equal rank strata ... and value strata (exploratory) were run beside it ... **Equal rank strata with random ties at H = 8 are best and most consistent**"; lines 109-115: the coverage truth was redefined after the first results ("a flaw in the pre-registered coverage truth"); lines 178-183 list five "Deviations from DESIGN.md, all reported". The strata rule behind the 1.4 to 5.3 was chosen after the pre-registered rule was seen to fail on the real data, and the validity truth was changed; the spike reports both, the paper does not.

### Appendices A to C, and rounding (every number checked)

**A32 (low). Table A1: what its cells are.**
- Paper, lines 1135-1151.
- Source: `results/paper/stratppi.md` line 10: "Largest miss over the label's checkpoints; ESS and pass at its last checkpoint." The caption says the miss is the largest over checkpoints; it does not say the ESS is the last checkpoint's. In the four bold cells the source prints "**invalid**" in place of an ESS (lines 14, 20, 21, 22; line 5: "Given only where the arm is valid"); the paper's 4.11, 2.59, 2.11 and 6.30 are in no results file (sub-audit recount from `stratppi.json`: 4.1134, 2.5888, 2.1077, 6.3045). The baseline itself is marked invalid in one row: line 18, "C2:refusal (0.93) | 100 | 0.057; **invalid**" for "random + pooled Wilson (R)". Two cells print "0.056", one bold (0.0564) and one not (0.0558; sub-audit).

**A40 (low). Table 2: the selected checkpoint's "predicted rate is optimistic".**
- Paper, lines 211-212 and 222 (two runs: 0.133 against 0.184; 0.141 against 0.168).
- Source: `reports/paper_seldonian_llm.md` lines 800-803, the same three-seed experiment: "At seed 2 no checkpoint was predicted feasible (upper bounds 0.176-0.207 against 0.172) ... the final checkpoint was tested and passed (0.142, upper 0.160): the predicted test was pessimistic by the same 0.01-0.02 it had been optimistic at seed 1." The source has a case of the opposite sign that the paper's table does not show.

**A41 (low). Rounding.**
- 0.0595 is "0.060" at lines 544 and 578 and "0.0595" at line 562; the spike prints it as 0.059 (`.planning/spikes/017-calibration-carrying-certificate/README.md` lines 199-200: "`b1w` on re-drawn sheets 0.059").
- 0.0515 is "0.052" at line 538; the spike prints 0.051 (same lines: "Clopper-Pearson 0.051"); `validity_recount.md` line 18: "0.0515".
- "0.90" at lines 552-553 stands for 0.902 and 0.901 (`stratppi_validate.md` lines 126, 130) in rows otherwise given to three decimals.

**A42 (low). "57% against 32% of the sheet's items run to the token limit".**
- Paper, line 1012.
- Sub-audit: no file prints the percentages; from `results/labels/p9/key.jsonl` the `cut_off` flag is set for 170 of 300 reference and 96 of 300 trained items, which gives 57% and 32%; the flag (`scripts/p9_sheet_build.py` line 51) means at the limit and not ending in terminal punctuation. By token count alone it is 191 and 108 of 300 (63.7% and 36.0%).

**A43 (low). Appendix D: "Every number in this paper is written by a script in the repository."**
- Paper, line 1206.
- The working draft's own note (`reports/paper_certification.md` lines 1255-1259) lists "Numbers the text states that no results file prints, recomputed from the data files named". Items B1 to B5 below are further numbers that no script output prints, and the harm-pool rows are "printed rates whose per-draw data were lost" (line 1224).

## Part B: no traceable source

**B1. "over their level in 1 of 322 cells and unresolved in 23"** (lines 577-578). Printed in no source file (searched `results/paper/*.md` and `scripts/validity_recount.py`). It reproduces from Table 4's own rows 6 to 11: 48 + 4 + 18 + 168 + 28 + 56 = 322 cells, 1 over, 3 + 0 + 1 + 12 + 2 + 5 = 23 unresolved. Of these, the delta 0.10 cells of rows 6 and 7 reuse the draws of their delta 0.05 cells (`validity_recount.md` line 9), and row 8's 18 cells are 9 settings run twice (A10).

**B2. "An ESS of 2.4 is worth 140 labels at n_s 100"** (lines 675-676). No source prints 140; it is 100 x (2.4 - 1).

**B3. "The guard's gain in this check is about 1.4 in labels"** (lines 1199-1200). No file prints it. Sub-audit: from `results/labels/p9/design_check.json` it is 1.42 if the mean margins are taken net of the truth, and (0.037 / 0.032)^2 = 1.33 on the printed margins.

**B4. "The exact bound is about twice as wide as the approximate ones"** (lines 1004 and 1195-1196). Derived from `design_check.md` lines 7-10: 0.065 / 0.037 = 1.76, 0.065 / 0.032 = 2.03, 0.065 / 0.035 = 1.86.

**B5. "At a difference of +0.006 ... about 930 pairs for an even chance and 2,100 for an 80% chance"** (lines 977-978). No file prints these. They follow from Table 10's formula at the design check's +0.0063 (416 x (0.020513 / 0.0137)^2 = 933; 950 x the same = 2,130); at exactly +0.006 the formula gives about 893 and 2,040.

**B6. "narrower than the labels alone in 10 of 210 cells, by at most 0.003"** (lines 797-798). The sentence is in `.planning/spikes/017-calibration-carrying-certificate/README.md` lines 158-159, but the working draft tags it to `results.md` (B6), and `results/paper/validity_recount.md` line 78 counts the block-PPI betting row as 84 cells. Sub-audit: the three plasmode JSON files hold 84 block-PPI cells, 10 of them narrower than Clopper-Pearson, largest gap 0.0026. The denominator 210 could not be reproduced.

**B7. Numbers that no results file prints and that were traced only by recomputation from data files** (all agree with the paper): Table 9's first row, 17.9% and 17.4% (`results/labels/refusal/design.json`: 700 of 3,920; 1,022 of 5,880); "12 refusals, 6 answers and none refuse-then-answer" of 18, "one stratum of 18 labels ... two of 15", "72 and 70", "three responses that appeared twice" (label and key files in `results/labels/refusal/`); 37% and 82% (258 of 700; 840 of 1,022, `design.json`); 56% and 34% (88 and 54 of 158, `results/labels/xstest/analysis.md` line 121); Appendix C's 16.9% and 17.2% (662 and 673 of 3,920, `results/labels/p9/guard.jsonl`); "ahead of `b1w` in seven cells, by 3-35%" and the medians 1.67 and 1.39 (`results/paper/stratppi.json`); "about 40% of the guard's flags" (1 - precision of 0.593 and 0.603); "cuts the budget threefold" (1,226 / 416 = 2.95). These were recomputed by the sub-audits, not by the compiler.

**B8. "Verified against the publisher or arXiv page on 2026-10-04"** (References, line 1102). Appendix D gives the output of `scripts/bib_check.py` as a "printed report"; no stored report was found, and the check was not repeated. Khosravi and Huo (2026, arXiv:2605.20270) in particular could not be checked against anything in the repository.

**B9. Statements about Thomas et al. (2019)** (sections 2, 4.2, 4.3): the requirement `P(g_i(a(D)) <= 0) >= 1 - delta_i` with `g_i(NSF) = 0`; the three-step construction; the name quasi-Seldonian for approximate bounds; `g_hat` as unbiased estimates; "Thomas et al. (2019) use the safety-set size with `kappa = 2`". The *Science* paper is not in the repository. All five agree with what the compiler knows of it (the doubled interval at the safety-set size is in its candidate objective, and `seldonian/llm/policy.py`'s docstring says "the classification models' convention is 2.0 at the safety-set size alone"), but they were not checked against the text.

**B10. One-line descriptions of other cited work** (sections 6.3, 8.2 and 12: PPI++, StratPPI, PPBoot, risk-controlling prediction sets, Learn-then-Test, conformal risk control, Miller, Bowyer et al., Boyeau et al., Zrnic and Candès, Gligorić et al., Csillag et al., Khosravi and Huo). Not in the repository. The working draft's own note says these are "Not yet checked against their sources". The repository's descriptions of PPI++ and StratPPI (`cert017.py` and `stratppi_baseline.py` docstrings) match the paper's formulas.

**B11. "Lagrangian theory is about a saddle point: at convergence, the average of the iterates satisfies the constraint in expectation"** (lines 208-209). No citation in the paper; the only source is the same sentence in `reports/state_2026-10-01.md` lines 62-63.

## Checked and found to agree (summary)

- Sections 2 to 6: 73 statements and formulas are listed in `evidence/tables/derived_source_ledger_sections_2_to_6.md`; 58 agree as written and 15 have an entry above. They include every formula of sections 4 and 6 (the Clopper-Pearson form, `b1w`, PPI++, the bootstrap-t limit, the cluster standard error, the carried bound, the paired difference, the validity rule, the pre-flight `G`, the judge's ESS), Table 2's four numbers, the worked case of section 4.4 (step 175; 0.168; 0.198; 0.206), and Table 3's rows.
- Table 4: all 24 rows' counts and ranges agree with `results/paper/validity_recount.md`, `results/paper/stratppi_validate.md` and `results/paper/agentdojo_recheck.md` (apart from A26, A27, A41).
- Tables 6, 7, 8, 9, 10 and A1: every cell agrees with its source (apart from A32).
- Appendix C's four miss rates and four margins agree with `results/labels/p9/design_check.md`.

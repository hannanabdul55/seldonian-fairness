*A blind review of `reports/paper_certification_clean.md` (v0.9.8), by a fresh agent that read only the manuscript and its figures (2026-10-10). Verdict: weak accept, mean 3.25 of 5 (v0.9.3: 3.83; v0.9: 3.67; v0.3: 3.17). The text below is the reviewer's, unedited; what was done about it is in PLAN.md.*

# Blind referee report

Manuscript: "Certifying behaviour rates of language-model policies: what holds, what a label buys, and what does not carry" (draft of 2026-10-08, clean copy, 1,686 lines).
Read: the whole clean file including Appendices A to E, and the eight PNG files under `reports/figs/`. Nothing else was opened. Line numbers (L) are those of the clean file. Where I say "I recomputed", I did the arithmetic from the numbers printed in the manuscript.

## 1. Summary

The paper takes the last step of a Seldonian algorithm, the held-out safety test, and treats it on its own as a "certificate": a one-sided upper confidence bound at level delta on the rate of a judged behaviour of a fixed language-model policy, with "no solution found" (NSF) as a permitted answer. It does not propose a training method. It audits four assumptions behind the guarantee (an untouched safety set, independent sampling, a bound valid at the sample size and rate, a label that is the quantity claimed) by measuring the miss rate of each bound where the truth is known: a synthetic bandit, resampled pools of real responses ("plasmodes"), and resampled benchmark tables. Its findings are: exact binomial and betting bounds hold; Student-t, Wilson-type limits at some rates, and the normal limits of PPI++, StratPPI and PPBoot run over their level at safety-test sample sizes; a safety set stratified by the reference model's own 8-sample per-prompt rate gives an effective-sample-size gain of about 1.5 at a 12% rate and 2 to 3 at 46% on new prompts, nothing at rare or extreme rates; a judge's calibration does not carry across prompt populations or (one run) across training that targets the label; per-pair bounds fail on AgentDojo's crossed design, where one of 28 pipelines certifies 5% under a user-task cluster bootstrap; 120 robot trials cannot certify 5% after two events; and a certificate of a trained policy in human labels is costed (about 420 pairs for an even chance at a 2-point margin) and not run.

## 2. Scores

| dimension | score | justification |
|---|---|---|
| evidence relevance | 3.5 | Most counts trace to Table 4 and I could reproduce a great deal of arithmetic, but the limit behind the abstract's headline design claim has no row in Table 4, much of section 8.1 is prose-only numbers, and several rows are reprinted from an unpublished companion paper whose data were lost. |
| falsifiability | 3.5 | One explicit over / unresolved / under rule and concrete cell counts; but the eighteen "registered predictions" are never listed, so the reader cannot check "fifteen of eighteen". |
| scope calibration | 3.0 | The body is careful; the abstract, three figure titles, the conclusion and the related-work summary say more than the body, and "makes each label go further" is true only in a setting the paper did not test. |
| argument coherence | 2.5 | The recommendation changes three times inside section 8.1 and is never stated in one place; abstract vocabulary does not map onto the body's; the conclusion omits the paper's largest section; sections 2 to 4 are background for a different paper. |
| exploration integrity | 4.0 | Unusually candid: failed predictions, rules chosen on the data, records without an independent timestamp, lost data and a past implementation bug are all disclosed. Deductions for unlisted predictions and for registrations tuned on a dry run. |
| methodological rigour | 3.0 | Sound validity rule, exact enumeration for the Wilson limit, a with-replacement redraw that corrects its own earlier optimism; against that, heuristic bounds with no theory chosen by which survived resampling, one model and one trained policy, validation of bootstrap limits inside their own bootstrap world, no language model in the in-loop check, one annotator who is an author. |

**Mean: 3.25. Verdict: weak accept**, at the low end. The negative results (sections 7.1, 9, 10.1, 10.2) are solid and useful. The constructive result (section 8.1) is not yet supported as the abstract words it.

## 3. Strengths

- A single, stated decision rule for every bound (L376-382), with the "unresolved" class kept separate from "holds", and an upper interval end reported for rows the authors favour as well as rows they reject.
- The exact enumeration of the Wilson limit's miss probability (L702-709) and the redraw with replacement (L711-718) overturn the paper's own earlier reading of `b1w`. I checked `exp(-z^2)` = 0.067 and 0.194, the zero-count limits 2.6%, 1.3%, 1.6%, and the 0.069 at n 100: all correct.
- The arithmetic I could check is clean. Table 8's five rows reconstruct to integer counts that sum to the pooled row (172 / 123 / 955; 308 flags; 159, 98, 51 flagged), and Cohen's kappa 0.81 and 0.59 and the phi-squared 0.40 and 0.65 follow from them. Table 6's Clopper-Pearson limits (0.137, 0.125, 0.052, 0.044), the Fisher p of 0.003, Table 7's per-pair limits, the 19,380 episodes, the 322 cells and 24 unresolved, the 8,820 responses, the 299 / 149 / 59 zero-count sizes, and Table 10 (to within one or two units of rounding) all reproduce. Figure 1_validity's "N of M over" labels match Table 4 row by row.
- The crossed-design result (sections 9.3, 10.2) is a clear, checkable warning about a public leaderboard, and the paper says what each resampling scheme takes as sampled.
- Appendix D distinguishes records with an independent timestamp from those without. Few papers do.
- Section 8.1's "What the gain costs" paragraph (L873-879) states the main limit of the paper's own design plainly.
- Appendix A's dissection of the 0.90 cell names the authors' own floor of 2 labels and shows the cell still fails without it.

## 4. Weaknesses, ranked

### Critical

**W1. The abstract's headline design claim rests on a limit that Table 4 does not contain, that the paper reports failing elsewhere, and that became the recommendation after the confirmation data were seen (abstract; sections 5, 7.1, 8.1).**
The abstract says (L26-28): "On new prompts, with predictions registered in advance, a certificate computed from this grouped sample with an approximate formula was wrong no more often than promised at breach rates of 12%, 46% and 82%." By elimination the "approximate formula" is the stratified Wald-t limit (L921, L945-946), since `b1w` "is over ... in all 4 at 82%" (L890-891). Four problems:
- The paper says three times that every bound it uses is in Table 4: "with every bound we use measured under one rule (sections 7 and 8, Table 4)" (L321-322); "Table 4 applies that rule to every bound we use" (L606); "Miss rate against delta for every bound used" (L608). There is no stratified Wald-t row in Table 4 (L626-656), nor in the validity figure. Nor are there rows for `b1w` with the sampled-pool term, `b1w` on judge-logit strata (L1007), the routing rule resampled "as one" procedure (L978-980), or the paired betting and bootstrap-t limits of section 10.4.
- The same limit fails where the paper tested it synthetically: "On a synthetic i.i.d. grid (36 cells, 4,000 draws) `b1w` is over in 5 cells and the Wald-t limit in 10" (L697-698); "on the synthetic grid of section 7.1 it did not hold" (L947-948); and "With 4 strata it was over in 1 cell of 13 (0.062 at delta 0.05, on the 9% label)" (L947). The abstract mentions none of this.
- It was promoted after the first confirmation: "Third, the recommendation changes ... the limit to use is the Wald-t one with 8 strata" (L950-955). At 12% and 82% the Wald-t result appears only as a contrast inside the "Refuted" bullet (L900), not as a kept prediction. At 46% its prediction was written after a dry run that "set the floor of a third (the Wald-t limit's gain, raised from 1.5 to 1.7)" (L1650). So "with predictions registered in advance" does not apply cleanly to the limit the sentence is about. This is a recommendation chosen on the data that validate it.
- The evidence is 12 cells (8 at L900, 4 at L921), one model, one trained policy, and the configuration was never run inside a training loop (W6). For the 12% and 82% cells no miss rates are printed, so "no more often than promised" cannot be told from "not more than two standard errors over", which the paper itself calls "weaker than showing that the bound holds" (L380-381).

Fixable by rewording the abstract, adding Wald-t rows to Table 4 (real pools, with replacement, synthetic grid, 4 and 8 strata), and listing the predictions.

### Major

**W2. "A simpler design makes each label go further" (L24) is supported only in a setting the paper did not test (section 8.1).** The body says so: the strata cost "k = 8 judged reference responses on each of the pool's 500 prompts, 4,000 in all ... An ESS of 2.4 is worth 140 labels at n_s 100. So the method saves labels only where the label on the trained policy costs far more than the stratifier's ... We did not test that case, and section 10.3 shows the guard and the annotator disagree on about 40% of the guard's flags. Where both are the same cheap judge, enlarging the safety set is the better use of the budget" (L873-879). Every experiment in section 8.1 uses the same guard for both. In the tested setting the design spends 4,000 judge labels to save 140. The abstract says only "the design costs 8 extra responses per prompt" (L36-37): it omits that each must be judged, and that the tested case is a net loss.

**W3. The abstract leads with gains for a claim of little practical use (section 8.1).** The 1.5 to 1.7 and 2.1 to 2.6 (L29-30) are for "the rate over the pool's own 500 prompts" (L793-794), a finite-population statement about prompts already in hand. A deployment certificate is about the prompt source, where the paper's own figure is 1.2 to 1.5 (L895, L927) and where `b1w` and the bootstrap-t StratPPI limit without the extra term fail in 17 and 21 of 28 cells (L813-814). Section 8.1's opening "Result" (L776-782: 2.4, 5.1-5.3, 1.4) is likewise a pool-rate figure, at delta 0.10, with `b1w`, on the pools that chose the rule, and the 5.1-5.3 is at a 66% rate where the paper later finds `b1w` over its level.

**W4. Universal statements that the manuscript's own tables contradict or do not cover.**
- L1489: "Where we could check against a known truth, no exact bound was over its level". Table 4, L649: "**Clopper-Pearson over pairs** | fails on a crossed design | ... 27 / 0 / 1; 28 / 0 / 0; 28 / 0 / 0". The next clause of the conclusion itself lists "a per-pair bound on a crossed design" as a failure. The abstract has the needed qualifier ("on data sampled as they assume", L17); the conclusion dropped it.
- L1469-1470: "their estimators with a bootstrap-t limit and proportional allocation are not [over their level] when the strata are fixed". L898-900: "With a large pool the bootstrap-t StratPPI limit is over in 1 of 8 mid-rate cells (0.105 at delta 0.10 on the 82% label), and in one more at the 98% rate".
- "every bound we use" (L321, L606, L608): see W1.
- Figure 3's title: "A judge's calibration does not carry, least of all across training that targets the label". Its own left panel shows the opposite ordering: the prompt-population rows reach 0.94 and 1.0 with 4 and 5 of 6 wordings over (L1033-1036), the training rows 0.80 with 3 of 6 (L1047-1048), and the body calls the training result "an existence result from two single runs" (L1043).
- Figure 1's title: "The gain from reference-rate strata can be predicted before labelling". Body: "Before any trained-policy response is labelled those two have to be assumed; with `rho = 1` the prediction ran well above the realised gain" (L827-829), by "27% to 41%" (L942), and "the absolute-error criterion we pre-registered ... failed on real data (0.29)" (L829-830).
- Figure 7's title: "In human terms the two policies refuse equally often". Body: "no difference ... was detected (-0.1 points, prompt-clustered standard error 3.9, so roughly -8 to +8)" (L1296-1297). A null with an 8-point half-width is not equality, and section 10.4 is built on that distinction.
- Figure 2's title: "with an honest limit its estimator holds". Body: "The repair depends on the allocation" (L852-853); Table 4 has it over in 45 of 108 cells under the published allocations (L647).

**W5. Published methods are judged outside the conditions their authors state, and the abstract quotes the worst cell (abstract; sections 6.3, 8.2, 9.4, 10.2; Appendix A).**
- "they were wrong as often as 24% of the time" (L20-21). The 0.241 is PPI++ at "n 225 at a 1.3% rate" (L1083), about three expected positives, where nobody would use a normal limit. The representative figure is in the same sentence of the body: "at a 20% rate and n of 100-500 it missed in 6-11%". StratPPI's 0.239 comes from the judge wording with constant strata; "the cleanest evidence" tops out at 0.187 (L1581-1582).
- PPI++ and StratPPI publish two-sided intervals with asymptotic guarantees. The paper tests a one-sided limit it derives from them ("The published limit is `p_hat + z_{1-delta} sqrt(V)`", L476). One-sided error of a symmetric normal limit on a skewed estimator is first-order worse than two-sided error; that is textbook, and it is the reason the bootstrap-t repair works. "did not keep that promise" (L20) describes a promise those papers did not make at finite n; L1399-1400 concedes this, the abstract does not.
- On reference-rate strata "The predictor is constant in 5 to 7 of the 8 strata" (L1565). StratPPI with a predictor that is constant within strata is a stratified Wald interval; calling the result "Against StratPPI" (L837) and "StratPPI as published" (Table A1) oversells what was compared.
- The 0.90 in Table 4 (L646) depends on "a floor that is ours" (L1591).
- The multiway variance of Cameron, Gelbach and Miller needs many clusters in each direction. The manuscript never states how many injection tasks a pipeline has, so the reader cannot tell whether "over its level for 26 of 28 pipelines" (L1181) is a finding or a known few-cluster failure.

**W6. Assumption A1 is not tested for the proposed design with a language model (sections 4.5, 6.4, 7.2).** The table at L304 sends A1 to section 7.2. There, the stratified test is run only on "a synthetic bandit ... with 4 strata and 500 runs per cell", with `b1w`, and "missed 0.084-0.116 at delta 0.1 ... one unresolved above it ... No language model is in this check" (L744-750). The recommended configuration (8 strata, Wald-t) is not run in the loop at all. Section 6.4 concedes "The resampling studies of section 8.1 hold the candidate fixed, so they do not test a candidate trained on the pool's other prompts" (L545-546). The language-model evidence for A1 is three and ten seeds from the companion paper (L752-757).

**W7. "Registered" covers records of very different strength, and the predictions are not shown (sections 8.1, 9.2, 10.4; Appendix D).**
- The eight and ten predictions are never enumerated with their thresholds. The "Kept" and "Refuted" bullets (L890-903) are prose; which prediction "had no label to test it" (L920) is not said.
- "Fifteen of eighteen registered predictions held" (L34) adds six of eight real tests to nine of ten that "already held on the re-mix" (L914-915) of the same model's responses to the same two sources in the same proportions. Those nine are closer to a replication of a known result than to predictions.
- Of the eight rows of Appendix D, three have "no independent timestamp" (L1645-1647), one of them "written after the termination counts and 12 transcripts had been seen"; one was "replaced" and its criterion "failed" (L1644); two are stated as committed, not pushed, which by the appendix's own standard (L1637-1638) is the weaker record (L1644, L1648).
- Section 10.4: "The analysis was fixed before any label" (L1354). It was fixed after the 220 human labels of section 10.3 on the same two policies, whose rates parametrise the design check ("at the first sheet's rates", L1361). "Before any of the 600 labels" is what is true.
- L788-789: "the truth used for coverage was changed to the mean of the half of the responses the draws come from". A change to the definition of truth after first results deserves more than a clause.
- The Appendix D row for sections 8.2, 9.1 and 9.4 (L1645) has no counterpart in those sections, which mention no prior expectation; section 8.2 says the opposite of its rule (L964-966).

**W8. Approximate limits are validated inside their own resampling world, and the one AgentDojo pass relies on one at a rare rate (sections 6.2, 10.2).** The check for the user-task cluster bootstrap redraws user tasks from the observed table (scheme (a), L427-431), which is the dependence that bootstrap assumes. The paper notes this limit for the pigeonhole bootstrap (L1191-1193) and not for the certificate it actually issues. The pass "rests on 7 successes in 6 user tasks" (L1217) at a rate of 1.1%, exactly where the paper's own advice is "Rare labels take Clopper-Pearson or a betting bound" (L673). Under the paper's sampling assumption the user-task rates are i.i.d. values in [0, 1], so the exact betting bound it applies to six robot tasks (L1123) could be applied to 97 user tasks. It is not reported. The clustered bound is also unstable: "at 4,000 draws it ranged from 0.058 to 0.107 over 30 seeds" (L1220-1221), while the validity check itself used "4,000 bootstrap draws inside every resample" (L1161). The manuscript also never mentions AgentDojo's suites; if injection tasks are suite-specific the table is crossed within blocks, and resampling tasks across the whole table is not the design.

**W9. The motivating regime and the regime where the contribution works do not overlap (sections 1, 8.1).** The abstract opens with "a harmful answer to at most 5% of requests" (L8-9) and the introduction says "The rates of interest are also small" (L72-73). The stratified design "gained little or nothing ... for rare behaviours (under 4%)" (L31). Its gains are for refusal of benign prompts at 12% to 46%. The gain also depends on training barely moving the policy: the correlation of reference and trained per-prompt rates is 0.92 (L782) for a policy whose guard-flag rate moved from 17.9% to 17.4% (L1288). No policy that training changed substantially was tested.

**W10. Dependence on an unpublished companion paper whose data are lost (Table 2, Table 4, section 7.2, section 11).** "The rows on the trained-policy harm pool are the companion paper's printed rates. Their draws were lost" (L613-614); "The pool's count is not recorded ... Computed that way at 124, the count that fits best" (L615-616). Five rows of Table 4, Table 2's selection-bias figures, and all language-model seeds of section 7.2 cannot be checked by a reader, and the reconstruction picks its free parameter by fit to the numbers it is meant to confirm. The exact enumeration needs no simulated rates; print it instead.

**W11. The human-label and trained-policy evidence is thin and its objects do not line up (sections 4.4, 9.2, 10.3, 10.4).**
- "One annotator (an author)" (L1274). Whether the annotator was blind to the policy and to the guard's flag is not stated for the 220 labels.
- Table 9 is "step 200" (L1286); the returned policy is "step 175 of 200" (L285); Table 10's rates "are those of the last checkpoint (step 200); the returned policy that section 10.4's samples come from is step 175" (L1322-1323). The human labels are on a checkpoint the algorithm did not return.
- The paper's only worked certificate of a trained policy (L284-287: rate 0.168, bound 0.198, threshold 0.206) states neither n, delta nor the bound. By my calculation a Clopper-Pearson limit of 0.198 at an observed 0.168 needs about 300 labels at delta 0.10 or more than 420 at delta 0.05; at 196 to 200 labels it is 0.219 to 0.220, which is NSF. If the pass relies on the stratified `b1w` limit, the paper's one returned adapter is returned by an approximate bound, by a margin of 0.008, and the text should say so. The threshold is also "treated as a constant" (L191-192).
- Section 10.1's label is the authors' reading of a harness field: "the termination reason `overheat` ... which we read as the rig's safety stop" (L1101-1102). A4 is not examined for it.

**W12. Structure (whole paper).** Section 8.1 runs from L767 to L959 and its recommendation moves from `b1w` (L771) to "`b1w` kept its level at rates of 9-18% and not at 65% and above" (L865) to "the Wald-t limit is the one the evidence supports" (L817) to "`b1w` is supported only at rates under one half" (L951), with the exact bound for rare and extreme rates. No table states the final rule. Table 5 still routes a stratified sheet to "`b1w`" (L975) with no rate condition. ESS has four operational definitions (estimate-to-limit at delta 0.10, the same at 0.05, truth-to-limit at the last checkpoint, truth-to-limit as a median over checkpoints: L534, L780, L805-806, L1542-1546), so one label has three or four gains. Sections 2 to 4 (L80-307) derive GRPO updates and Lagrange multipliers for a paper that "does not propose a new way to train" (L75). The conclusion (L1487-1495) does not mention stratification, the subject of the abstract's longest paragraph.

**W13. Missing related work.**
- Stratified binomial intervals: stratified Wilson and score intervals (Yan and Su 2010; Miettinen and Nurminen 1985). `b1w` is introduced as "a heuristic and not the inversion of a stratified score test" (L453-454) with no reference to the intervals that are.
- One-sided Wilson coverage: Brown, Cai and DasGupta (2001) and Cai (2005) give the oscillation and the systematic one-sided bias that L702-709 re-derives.
- Survey sampling: two-phase sampling for stratification and its variance (Neyman 1938; Cochran 1977; Särndal, Swensson and Wretman 1992). The "sampled-pool term" of L462-464 is that variance component.
- Bootstrap-t theory (Hall 1988, 1992): the source of the "second order" label in Table 4 (L634).
- Prevalence under misclassification: section 6.5's `p = (q - a) / (s - a)` is the Rogan-Gladen estimator (1978); its failure under shift is the subject of the quantification and label-shift literature (Saerens et al. 2002; Lipton et al. 2018).
- Seldonian follow-ups, especially guarantees under distribution shift (Giguere et al. 2022), which is section 9's topic, and Metevier et al. (2019).
- High-probability certification of language-model behaviour: Prompt Risk Control (Zollo et al. 2024) and Clopper-Pearson certificates of LLM bias and knowledge (Chaudhary et al. 2024) are the nearest prior "certificates of a behaviour rate".
- Stratified and active evaluation: Kossen et al. (2021); Fogliato et al. (2024); Chaganty et al. (2018) for control variates in evaluation.
- References are absent for methods used by name: Clopper-Pearson, Wilson, Bentkus, Anderson, Hoeffding's inequality (only the 1956 Poisson-binomial paper is listed), Satterthwaite, Platt scaling, GRPO, LoRA, and the Granite and Qwen2.5 models.

### Minor

- **m1 (6.1).** The rule's null is "valid", so a multiplicity correction makes failing rows look better and adds nothing for favoured rows. "the test rules out anything above 5.9%" (L18-19) is the largest of 42 uncorrected per-cell 95% ends; "rules out" is too strong, and for an exact bound on i.i.d. draws the check tests the code, not the theorem.
- **m2 (6.4, 8.1).** The ESS baseline is the pooled Wilson limit, which is "itself over its level in 7 of these 28 cells" (L809) and at L930-931. "Width is compared at equal validity" (L525) is not met by baseline or, at 82%, by `b1w`.
- **m3 (7.1).** "over their level in 1 of 322 cells" (L684): the 322 "were grouped after the results" (L1386-1387) and exclude the with-replacement rows (7 of 28), the other eleven settings (8 of 528) and the synthetic grid (5 of 36). The disclosure is there; the headline number should not be.
- **m4 (9.1).** The judge's recall is "0.04-0.54" (L1038) with false alarms "at or near zero", so `s - a` is small and the carried bound is fragile for reasons other than shift; tests are "unadjusted for six wordings" (L1039).
- **m5 (10.1).** Tasks are "the six of 18 on which one of the three models scored highest" (L1109-1110); selection on success may correlate with stops. The time-order confound is stated; good.
- **m6 (10.3).** The interval method of Table 8 is not stated. "0.92 (0.88-0.96)" matches a Wald interval for 159 of 172 (Clopper-Pearson gives 0.87-0.96): a normal interval at a 92% rate, in this paper.
- **m7 (10.4).** The budget uses "pairing correlation 0.66 (the guard's flags on the same prompts)" (L1321) for human labels, and a `rho^2` that "has no interval here" (L1338-1339). Appendix C shows the guard's gain as "about 1.4 in labels, against 2.5 in Table 10" (L1629-1630); the abstract's 420 is the optimistic end.

## 5. Plain errors

1. **L840** "Three things follow." Four numbered items follow (L846-871).
2. **Orphan figure, and file numbers off by one.** `fig1_validity.png` ("Which bounds are over their level") is referenced nowhere; section 7.1 has no figure. Figure 1 is `fig2_strata_ess.png` (L833), Figure 2 is `fig3_stratppi.png` (L842), and so on to Figure 7 = `fig8_refusal_sheet.png` (L1279).
3. **L321-322, L606, L608** "every bound we use" / "every bound used": Table 4 has no row for the stratified Wald-t limit recommended at L954 and in the abstract, nor for the limits listed in W1.
4. **L1489** "no exact bound was over its level" conflicts with Table 4 L649 (Clopper-Pearson over pairs, 83 of 84 cells over).
5. **L1469-1470** "are not when the strata are fixed" conflicts with L898-900.
6. **Rate of the encoded-refusal label.** "93%" at L800, L807, L820, L1548, L1560-1561, L1571; "94%" on Figure 1's point label "refusal, encoded (94%)"; "95%" at L621 ("one checkpoint at a 95% rate") and L631 ("4 mid-rate labels (9-95%)").
7. **Gain on that label.** L807 "At the 93% rate the strata lose (0.88 and 0.93)"; L820 "0.96 to 1.12 at a 93% rate, Table A1"; Figure 1 plots it at about 1.17. One section says the strata lose and lists a 12% gain for the same label.
8. **L937** "the 1.5 to 5.0 that the same measure gives at rates of 9% to 66%": the same measure gives "4.66 and 5.14" (L807) and Table A1 gives 5.19 (L1559). The top is 5.1 or 5.2.
9. **Abstract L32-34** "The gain is smaller, 1.3 to 1.5 at the first two rates": the body gives 1.49 and 1.34 at 46% (L927) and, for the first confirmation, only "the gain with the term is 1.20 to 1.32" (L895) with no limit or rate attached. No Wald-t source gain at 12% is printed, and 1.20 is below 1.3.
10. **L955-956** "at 82% it gave none or lost" conflicts with L949 "0.86 and 1.13" (a 13% gain at a safety set of 200).
11. **Table 4 cell sets for `b1w`.** L631 has "4 mid-rate labels" (24 cells) and L632 the pushed label at "n_s 200" only (2 cells): 26 cells. The rows compared with it use "5 mid-rate labels, n_s 100-200" and 28 cells (L636-638, L642-643). Table A1 (L1554) reports `b1w` on the pushed label at n_s 100 (0.024; 2.19), so the cell exists and is left out of Table 4 and of the 322.
12. **Table 3, L586** lists "225 human harm labels" as a gold label for section 8.2. L424-425 says of the sheets "The labels in this study are planted ... they are not an annotator's", and L346-347 says "The human-label results are in section 10.3". These 225 labels are never described or used by name.
13. **L1056** "The guard's recall and precision against the same labels do not move (17 of 18 and 16 of 18; 0.59 and 0.60)": raw recall is 0.94 and 0.89. Table 9 (L1293) gives recall "0.99" and "0.98", and L1299-1300 says "The guard missed almost nothing". The sentence mixes unweighted recall with design-weighted precision.
14. **L1296** "-0.1 points" against **L1322** "the unrounded difference of the two rates (-0.0005)", which is -0.05 points.
15. **Checkpoint.** L1286 "step 200", L285 "step 175 of 200", L1323 both; L885 "the trained policy is that of section 9.2's constrained run" names no step.
16. **L1046** "recall ... fell from 0.21 to 0.03 for the canonical wording" against **L1609** "of 93 guard-flagged refusals after training it called 89 answers" (4 of 93 is 0.043). Figure 3's right panel, "mean of six wordings", also starts at 0.21 and ends near 0.12: check that the 0.21 is the canonical wording's and not the mean.
17. **L1361-1362** "at or under 0.045 for the pool rate": Appendix C adds "0.001, 0.051, 0.063 and 0.052" when prompts are redrawn (L1624-1625). Limit (b2) is the one "for new prompts" (L1357), so its relevant figure is 0.052, which the main text omits.
18. **L284-287** the worked case gives no n, delta or bound; 0.198 cannot be reproduced from 0.168 with any bound defined before that point (see W11).
19. **Counts that cannot be derived.** L979-980 "over 20 cells (two forms of the refusal judge, rates of 1.3-20%, 225 and 1,000 labels ...)": with the three set rates of L420-421 that is 12. L741 "3 of 42 cells" beside "eleven split rules". L1085 "10 of 210 cells".
20. **L801-802** "3 at 0.10 ... as in section 7.1": section 7.1 gives no such count.
21. **L919** "46% (81% on the hard prompts, 12% on the others)": 193 and 207 of 400 at those rates give 45.3%. Possibly rounding; check.
22. **L862 and L852** `b1w`'s median ESS is "2.16" both with replacement and without. Check that one was not copied.
23. **Dangling referents.** L670 and L641 "the stored design"; L693 "at the other eleven settings in the same file"; L997 "those plasmodes"; L1361 and L1620 "the first sheet"; L1017 "about 45 or more per stratum" (no result supports 45); L1323 "section 10.4's samples" inside section 10.4.
24. **Process notes left in the text.** L3-4 "Generated from the working draft by `scripts/paper_clean.py`; to change the text, edit the working draft ..."; L1499-1502 "Verified against the publisher or arXiv page on 2026-10-04 ..."; L1656-1657 "The working draft carries a source tag on each number"; L125 "`seldonian/llm/`"; L744 "our `SeldonianLLMPolicy` class"; L1368 "are in the repository" with no link.
25. **Undefined terms.** `b1w` (L444, never expanded); "compiled judge" (L986, L1031, L1046); "Anderson" (L630); "Platt score" (L988); "placebo covariate" (L780); "tight Wald test" (L741); "power-tuned" (L648, L1011); "pre-flight" (first at L824, and on Figure 1); the value of `c_H` (L555); "GLIDE" (L1538).
26. **Defined and never used.** "breach rate" (L11) appears only in the abstract, and the body uses "breached" for something else, a training run ending over its threshold (L753-755). "quasi-Seldonian" (L115) is never used again, although it is the right name for every certificate here that uses an approximate limit.
27. **Overloaded symbols.** `k`: constraints (L138), positives (L275), reference samples (L410). `m`: candidate prompts (L208), the Wilson root (L438), clusters (L493), and `m_hat` the miss rate (L374). `R`: replications (L373) and the baseline (L525). `G`: GRPO group size (L199) and variance share (L540). `rho`: reference-to-trained correlation (L553) and label-to-judge correlation (L561). `lambda`: multiplier (L226) and PPI++ weight (L471). `kappa`: inflation (L210) and Cohen's (L1257). `z`: label (L175) and normal quantile (L435). `s`, `a`: recall and false-alarm rate (L569-570), positives `s_h` (L461), the algorithm `a` (L86).
28. **Figure 2 subtitle** "Right of the line the bound misses more often than it claims" conflicts with the rule of L379-381: the blue point at 0.054 (Table A1, L1560) is right of the line and is "unresolved", and L859 says `b1w` "is over its level in no cell".
29. **Validity figure legend** puts "pigeonhole bootstrap-t" under "approximate, used in the paper"; L687 says "which no certificate in this paper uses".
30. **Tables 7 and 8** (L1198, L1240) have no call-out in the text.
31. **References.** Csillag precedes Cameron; Zhao sits between Miller and Owen; "Hoeffding's" bound (L282) has only the 1956 entry, a different result.
32. **L945-947** reads as a contradiction on first pass: "the only stratified limit that was over its level in no mid-rate cell on real responses" is followed by "With 4 strata it was over in 1 cell of 13 ... on the 9% label". The qualifier "with the 8 strata of the rule" is there, but the sentence should lead with it.

## 6. Major items

- [ ] Add the stratified Wald-t limit to Table 4 under the same rule: stored pools, with replacement, both confirmation pools, the synthetic grid, 4 and 8 strata, with miss rates printed. Do the same for `b1w` with the sampled-pool term, the routing rule as a procedure, and the paired limits.
- [ ] Print the eighteen predictions with thresholds and outcomes, and the results of the re-mix beside the fourth pool's.
- [ ] Run the recommended configuration (8 strata, the recommended limit) inside the training loop with enough runs to resolve a miss of delta plus a point or two, on the bandit and on at least one language model, or withdraw the claim that A1 is tested for it.
- [ ] Test the design where it can pay: an expensive label stratified on a cheap one. The XSTest human labels with the guard's reference rate as stratifier are in hand.
- [ ] Rebuild the abstract around source-rate gains, name the two formulas, give the label cost of the strata, and replace "24%" with the range at ordinary rates.
- [ ] A second model family and more than one trained policy for section 8.1, including one that training moved substantially.
- [ ] For AgentDojo, report an exact bound on user-task rates beside the bootstrap-t; state the number of user and injection tasks per pipeline and per suite; run or drop Menzel (2021).
- [ ] Replace the companion paper's reprinted simulation rates by the exact enumeration; move the remaining companion results to clearly marked background.
- [ ] Human labels: a second annotator, blinding stated, and labels on the returned checkpoint (step 175).
- [ ] One decision table: rate regime by claim (pool or source) by label cost, giving the limit to use and the evidence for it. Update Table 5 to match.
- [ ] Cut sections 2 to 4 to about a page; split section 8.1 into design, pilot, confirmation and recommendation; one ESS definition throughout.
- [ ] Add the literature of W13 and compare `b1w` and the Wald-t limit with a published stratified score interval.

## 7. Questions for the authors

1. Was "the Wald-t limit is over in none" one of the eight registered predictions of the first confirmation? If not, which registered prediction supports the abstract's sentence at 12% and 82%?
2. What are the Wald-t miss rates in the eight cells at 12% and 82%, and how many are unresolved?
3. In section 4.4's worked case, what are n, delta and the bound? Would the adapter be returned under Clopper-Pearson?
4. Which checkpoint is "the trained policy" of the confirmation pools (L885)?
5. How many points underlie "Spearman 0.83" (L826), and are the settings independent?
6. Why does the 322 count exclude the pushed label at n_s 100, which Table A1 reports?
7. Was the annotator of the 220 responses blind to policy and to the guard's flag? Who produced the "225 human harm labels" of Table 3?
8. How many injection tasks does each AgentDojo pipeline have, and are they shared across suites? Was resampling done within suite?
9. For the pipeline that passes, what is the miss rate of the cluster bootstrap-t in its own resampling, and how often does it return no limit?
10. Is the "betting mixture" a function of the count alone? The statement that "each of those miss rates is a finite sum" (L614-615) needs that.
11. How was `c_H` obtained, and what is `c_8`?
12. Does the reference model's guard-flag rate of 17.9% (Table 9) match the 0.186 implied by the threshold 0.206 less the 0.02 margin, and on which prompts was each measured?
13. Did any tested trained policy have a reference-to-trained correlation well below 0.9? What does the design return when training moves the policy a long way?
14. Why is the stratifier's judging cost not charged in the ESS, given that both labels are the same guard?
15. Is "overheat" documented by the benchmark as a safety stop?

## 8. Readability for a non-specialist

**Abstract.** The first paragraph (L8-15) is clear and the definitions of "breach rate" and "certificate" work. After that a reader outside the subfield can follow the direction of each claim but cannot tell what was done, and a specialist cannot tell which method is meant. Sentences that are hard to follow:

- L17-19: "promised to be wrong in at most 5% of repeated trials, they were wrong in at most 5.2%, and the test rules out anything above 5.9%." A lay reader sees 5.2 above 5 and concludes the methods failed; "the test" has no antecedent.
- L19-22: "Two published shortcuts that let an automatic judge stand in for some labels" and "A repaired version kept it under narrower conditions". Neither is named. PPBoot, a third published method that the body also finds over its level, is silently dropped.
- L27-28: "a certificate computed from this grouped sample with an approximate formula", and L32: "A sharper formula gave up to 3.0 but ran over its promise at 82%." Two unnamed formulas, and "3.0" has lost its unit.
- L32-34: "when the certificate must cover all prompts of that kind and not only those sampled." A non-specialist assumes a certificate always covers unseen prompts. That the preceding numbers do not is the most important caveat in the paragraph and arrives last.
- L34-35: "Fifteen of eighteen registered predictions held, two failed and one could not be tested; ten were written after a trial run on older responses." Accurate, and opaque without Appendix D.
- L39-41: "nor, in the one training run we examined, after training aimed at what the judge measures." In the body the training targets the guard's flag and the calibration that fails is a different, rubric judge's. The abstract uses "automatic judge" for both, and for the label every certificate rests on (L36, L47). A reader will take them for one model.
- L48-50: "about 420 labelled prompt pairs would give an even chance of certifying". "Prompt pairs" is not explained (two policies' responses to one prompt), and an even chance of certifying is an odd thing to budget for without saying so.

**Does the plain vocabulary connect to the body?** No. The abstract has a vocabulary of its own that the body never picks up:

| abstract | body |
|---|---|
| breach rate | behaviour rate, "the rate"; "breached" means something else at L753 |
| wrong, promise | miss, level, over its level |
| grouped sample, groups | reference-rate strata, stratified safety set |
| starting model | reference policy |
| approximate formula | stratified Wald-t limit |
| sharper formula | `b1w`, stratified Wilson-type bound |
| published shortcuts; repaired version | PPI++, StratPPI; bootstrap-t limit |
| automatic judge | guard (gold label), rubric or "compiled" judge (predictor) |
| agent configurations | pipelines |
| cover all prompts of that kind | a claim about the prompt source; the sampled-pool term |

One parenthesis per term in the abstract, or the plain term carried into the section headings, would close the gap.

**Introduction.** L54-78 is short and readable; "NSF" and "level delta" are introduced adequately. It states no result, and the reader then meets 230 lines of algorithm and training theory (sections 2 to 4) before learning at L315 what the paper proposes. Section 4.3 (GRPO, multipliers, floors) will lose a non-specialist and is not needed for anything that follows. Phrases that need a gloss: "In human terms" (L193, L518, L1296; it means "measured by human labels"), "the stored design" (L670), "plasmode" (defined at L404, well), "pre-flight" (L824), and "over its level", which carries the whole of sections 7 to 10 and is defined only inside a bulleted rule at L379.

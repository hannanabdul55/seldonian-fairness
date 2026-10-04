# Paper plan: Seldonian certification of LLM behaviour

Written 2026-10-04 from the state report (`reports/state_2026-10-01.md`) and the review of what a
certification paper can claim now. The user annotates labels and will recruit a few more
annotators; everything else here is CPU work, writing, and about four local GPU-hours.

## 0. The claim, fixed up front

> A Seldonian safety test certifies a label-defined behaviour of a fixed LLM policy from a
> sample: a one-sided bound at level delta on the rate of a code- or model-defined label, with
> "no solution found" as an outcome. We show where it holds (synthetic truth, plasmodes on real
> labels, public frontier-model traces), how to make the sample go further (stratified safety
> sets, the judge as a variance reducer with a fixed routing rule), and what does not carry
> (calibrations across populations and across training that targets the label, i.i.d. bounds
> on crossed designs, normal limits, judge-assisted finite-sample bounds).

Scope stated in the first section: the label is the guard's or the harness's unless marked
human; policies trained here are 0.5-3B; the frontier-model results are certificates on
published traces, not training. The training half (Rounds 1-6, spike 021's follow-up) is a
separate paper.

## 1. Steps

Owner: the user (U), Claude (C), annotators (A). Times are working time, not calendar time.

| # | step | owner | inputs | output | time | depends on |
|---|---|---|---|---|---|---|
| P1 | **Paper skeleton from existing results.** Sections: claim and scope; the certificate and the Lagrangian contrast (state report 2.3); validity evidence (Round 4 synthetic, 012, 013/014 plasmodes, 004-010 trajectory); stratified safety sets; the routing rule and label budgets; negatives; frontier traces (019, 020); limits; related work. Every number traced to a spike README. Gaps marked `[GAP: Pn]`. | C | spike READMEs, MANIFEST, state report | `reports/paper_certification.md` v0.1 | 2 days | - |
| P2 | **Guarantee table.** One row per bound used: Clopper-Pearson, betting (exact); `b1w`, cluster bootstrap-t, PPI++ bootstrap-t (approximate, with the plasmode miss rate and its Monte Carlo se beside each); normal limits (shown failing). Pull the numbers from 013 `validity_*.md`, 017 `results.md`, 020 `plasmode.md`. | C | existing results | a table in the paper and `guarantees.md` | 1 day | P1 |
| P3 | **Related-work positioning.** Thomas et al. 2019 (Seldonian); Miller 2024 "Adding error bars to evals" and the clustered-SE line; Angelopoulos et al. PPI/PPI++; conformal and risk-control work (Bates et al., Angelopoulos "learn then test"); eval-noise position papers (Bowyer et al. 2025); stratified sampling in evals. The direct neighbours (found 2026-10-04, none in any literature file yet): StratPPI (Fisch et al., NeurIPS 2024, arXiv 2406.04291: stratified sampling with prediction-powered intervals for LLM evaluation), Active Statistical Inference (Zrnic and Candès, ICML 2024) and Confidence-Driven Inference (Gligorić et al., NAACL 2025) for choosing which items humans label, AutoEval Done Right (Boyeau et al.) for PPI in model evaluation, and XSTest's annotation scheme (Röttger et al., NAACL 2024) for the refusal labels. Check every citation against the 013 `LITERATURE.md` conventions (verified, not recalled). State in one paragraph what is new here, without claiming stratification or judge-assisted intervals as such (StratPPI and the active-inference line have them): one-sided certificates with NSF on a trained policy, validity checked by plasmode, the reference model's own per-prompt rate as the stratifier with a pre-flight, and the carrying negatives (calibration under training that targets the label, i.i.d. bounds on crossed designs). | C | web, LITERATURE.md | related-work section, `.bib` | 1 day | P1 |
| P4 | **Labelling guideline.** Refusal: what counts, with the hybrid rule ("a refusal sentence followed by the answer" decided one way, with the reason) and ten worked examples drawn from 014's step-0 and step-200 samples. Harm: extend the existing `results/screen/labels/GUIDELINE.md` (encoded and plain prompts; "disclaimer then real answer" is harm). A `skip`/`unsure` option, and an opt-in note for harmful content. | U decides the rules, C drafts | 014 samples, existing guideline | `labels/GUIDELINE-refusal.md`, updated harm guideline | 1 day | - |
| P5 | **Refusal sheet.** 400 responses (version 2): 200 from 013's step-0 reference samples and 200 from 014's step-200 pushed policy, hybrids over-sampled (the guard's refusals at step 200 are mostly hybrids) and 275 from the guard-negative plain strata, where the guard's misses are; stratum sizes recorded, weights computed at analysis time. A shared 40-item set every annotator labels (agreement), the rest dealt to annotators within each stratum, and a second shared 40 held in reserve. Served as a labelling page like the Harm Label Desk (artifact with the db capability, labels at `labels/<uid>`, owner-only read), or a plain JSONL for anyone who prefers a spreadsheet. | C builds, U + A label | P4, `results/spikes/014/gen_s0.jsonl`, 013 reference samples, `scripts/refusal_sheet_build.py` | `results/labels/refusal/sheet.jsonl`, `labels_<handle>.jsonl` | build 0.5 day; labelling about 3 annotator-hours in all (220 items each for two people) | P4 |
| P6 | **Refusal analysis.** Primary event: refusal in the strict sense (`r`); `r` or `h` reported beside it. Agreement on the shared 40 (kappa with its interval; gate 0.6 before using the labels; if it fails, revise the guideline and use the reserve set); the human refusal rate at step 0 and step 200 with design weights, and the guard's recall for each population separately, never pooled; rho^2 against human labels for the guard's logit (needs the guard's logit per response: one local judge pass, about 20 GPU-minutes, logging the logit not p) and for the compiled judge (017's scores); the hybrid share. This answers 017's follow-up 1 and settles whether spike 018 is needed (M4 rule: guard rho^2 under 0.3). | C | P5 labels | `labels/refusal/analysis.md`; paper section "a certificate in human terms, part 1" | 1 day | P5 |
| P7 | **Harm positives sheet.** 1,000 responses from the capability screen (4,800), stratified by the 4B guard's score so positives are over-sampled, weights carried; the shared 40 for agreement; aim for 30 or more human positives (006's floor for bounding a judge's recall). | C builds, U + A label | P4, `results/screen/`, 017's sampling code | `results/labels/harm/` | build 0.5 day; labelling 15-20 annotator-hours | P4 |
| P8 | **Harm analysis.** Weighted human harm rate; recall and false-alarm rate of the 4B guard with Clopper-Pearson intervals; the label budgets (301/149/59) restated with the judge's measured rho^2 on the logit; the answer-rate-aware correction checked against the human rate. | C | P7 labels | `labels/harm/analysis.md`; paper section | 1 day | P7 |
| P9 | **One human-certified policy.** Spike 014's step-175 returned policy (Round 6's is not on disk): fresh responses from it and from the reference on the safety prompts, labelled by humans under P4. The certificate in human terms, on the difference between the two rates; "no solution found" is a reportable outcome. The event, the bound, the sample and the stopping rule are fixed in section 6. Stratification (013) and PPI++ (017) are run on the same labels as the efficiency comparison, not as the certificate. | C samples, U + A label | P4, P6 (its rates set the cap on pairs), the checkpoint | the paper's headline figure | sampling 0.5 day plus about 1 GPU-hour; labelling 3-8 annotator-hours (up to 600 pairs) | P6 |
| P10 | **Qwen3-1.7B replication of the stratified safety set** (013 dropped it for budget): the C1 pool, 8 reference samples, step-200 side-effect checkpoint, the plasmode. One more model for the 1.4-5.3x claim. Lowest priority after the 2026-10-04 review: another sub-2B model does not answer the small-model objection. | C | 013's `gen013.py`, `real_plasmode.py` | a row in the use-case map | about 4 GPU-hours local, 0.5 day | - |
| P11 | **Library.** `seldonian/llm/certify/`: the bounds (`stratbounds`, `ppipp_boot`, the cluster bootstrap-t), the strata and pre-flight (with `--pushed`), the routing rule, the design-weighted estimator, the plasmode protocol as tests, and a CLI that takes a labels file and prints the certificate. Pin a tag; rent-my-gpu imports the tag. | C | spikes 013, 017, 020 | package, tests, tag `cert-0.1` | 3 days | P2 |
| P12 | **Figures.** Validity (miss against delta, every bound, every setting); ESS against ICC_ref with the pre-flight line; the AgentDojo naive-vs-clustered plot; the RoboDojo resolution table; the carrying failures. Dataviz skill rules, no dual axes, palette validated. | C | results | `reports/figs/` | 1 day | P1-P8 |
| P13 | **Full draft, internal review, submission target.** Workshop (fast, with the scope as stated) if P9 is not in; main venue if it is. | U + C | all | v1.0 | 3 days | all |
| P14 | **StratPPI as a baseline.** Run StratPPI (Fisch et al.) on 013's and 014's plasmodes beside `b1w` and the PPI++ bootstrap-t: miss rate and width, same draws. Without it the stratified-safety-set section has no comparison to the closest published method. | C | P2, their paper and code | rows in the guarantee table | 1 day, CPU | P2 |
| P15 | **The guard against published human labels.** XSTest's repository ships completions from five model variants (GPT-4, Llama 2 with and without its system prompt, Mistral instruct and guard) on all 450 prompts, each with two human annotations and a final label in the three classes our guideline follows (checked 2026-10-04: columns `annotation_1`, `annotation_2`, `agreement`, `final_label`). Score them with Qwen3Guard-4B and report recall, false-alarm rate and rho^2 against those labels: 2,250 human-labelled responses on models we did not train, at no annotation cost. The completions are under the model owners' licences: ask the user before pulling them, keep them out of the repo. | C | XSTest repo, the guard | `results/labels/xstest/analysis.md` | about 20 GPU-minutes, 0.5 day | - |

## 2. Critical path

P4 -> P5 -> P6 -> P9 is the path that turns a workshop paper into a main-venue one, and it is
the only path that depends on other people. Start P4 first. P1-P3, P10-P12 run beside it.

Rough calendar: P1-P3 in the first week; P4 and P5 built in the same week so labelling can
start; P6 as soon as the shared 40 agree; P7 and P9 in weeks two to three as annotators allow;
P10-P12 fill the gaps; P13 in week four.

## 3. Labelling logistics (what the user asked to run)

- **Who:** the user plus a few recruited annotators. Each labels the shared 40 (both sheets)
  and a split of the rest. Agreement on the shared 40 is reported in the paper (kappa with a
  CI); items where the two labels disagree are adjudicated by the user, and the adjudicated
  label is the gold, with the pre-adjudication disagreement rate stated.
- **Tool:** the Harm Label Desk pattern worked (blind items, per-annotator order, labels at
  `labels/<uid>` with owner-only read, a JSONL export fallback). One page per sheet. Annotators
  need claude.ai accounts invited as editors, or they use the JSONL.
- **Content warning and opt-in** for the harm sheet; the refusal sheet is benign prompts.
- **Weights:** every sheet is a stratified sample with recorded weights (017's lesson: the
  sheet read as i.i.d. was wrong in 98% of draws for one wording). The analysis scripts take
  the weights; nobody averages raw labels.
- **Budget check:** 6 (refusal) + 15-20 (harm) + 10-20 (certified policy) annotator-hours,
  about 35-45 hours in all; with four people that is a few evenings each.
- **What is NOT done with the labels:** no judge is trained on them (that would make them
  training data, not gold; spike 018 is deferred by the M4 rule). They are held out for the
  certificates and the rho^2 measurements only.

## 4. What is explicitly out of this paper

Training a certified agent (spike 021's follow-up, RunPod estimate in the handoff), the own
judge (018), the TD-error and forbidden-task ideas (their results are cited where the
trajectory certificate is used), and any claim about human-defined harm beyond what P7-P9
measure.

## 5. Tracking

Each step gets a line in this file when done (date, output path, one-line result).

- 2026-10-04 decisions: refuse-then-answer is its own label (`h`); two annotators now, more later (slot/K scheme).
- 2026-10-04 P4 draft: `results/labels/refusal/GUIDELINE.md` (four labels r/a/h/u, three tests, ten worked examples). Harm guideline extension pending.
- 2026-10-04 P5 built: `results/labels/refusal/sheet.jsonl` (200 items, 8 strata by step x guard x hybrid shape, weights in `key.jsonl`, 40 shared), page https://claude.ai/artifact/RrKcGyzaQT9PHm41ErphGx (db rules: labels read/write owner, labels/{self} interact), `scripts/refusal_labels.py` (flatten, analyze). Labelling not started. The paper
draft carries `[GAP: Pn]` markers until the step lands. Commits outside weekday 9-5 PT.
- 2026-10-04 blind-spot review (section 7). Decisions by the user: the certificate's event is refusal in the strict sense (`r`), with the hybrid share reported beside it; labelling held until the sheet was fixed.
- 2026-10-04 P5 rebuilt as version 2 before any label was collected (the store was empty): `scripts/refusal_sheet_build.py` (the builder, which version 1 never had), 400 items (275 in the guard-negative plain strata), the guideline's ten example prompts removed from the frame (490 prompts), a reserve shared set in `reserve.jsonl`, split items dealt within strata. Page republished at the same link. `scripts/refusal_labels.py` rewritten: weights at analysis time, design and prompt-clustered standard errors, conservative intervals, recall per population, ties and unsure bounded, each annotator alone. Checked on 2,000 simulated draws with a known truth: estimates unbiased, design se matches the spread, intervals cover (conservatively, 100%).
- 2026-10-04 guideline: the labels are tied to XSTest's published scheme; reserve-set and cut-off rules added. Steps P14 (StratPPI baseline) and P15 (guard against XSTest's human labels) added.
- 2026-10-04 P1, P2, P3 drafted as one file: `reports/paper_certification.md` v0.1 (abstract, claim and scope, the certificate, the guarantee table as Table 2 with a miss rate and source for every bound, validity, strata, routing rule, what does not carry, the two frontier benchmarks, limits, related work), `reports/paper_certification.bib` (22 entries, each checked on the publisher or arXiv page; two fields marked CHECK). The guarantee table lives in the paper only, not in a separate `guarantees.md`, so there is one copy to keep right. 10 `[GAP: Pn]` markers (P6, P8, P9, P10, P14, P15). Appendix A lists five open checks, two of them discrepancies found in the sources: 013's README understates one mid-rate miss against its own table (0.097 and 0.054, not 0.093 and 0.047), and the training paper's delta-0.05 sentence in 6.2 repeats another row and cannot be regenerated. Three more neighbours found while verifying: Khosravi and Huo 2026 (anytime-valid risk control for RL-trained models), Zrnic 2024 (a bootstrap for PPI), Csillag et al. 2025 (e-values for PPI).
- 2026-10-04 P15 started with the user's yes: XSTest's five completion files pulled to `/mnt/d/seldonian-runs/p15/xstest/` (outside the repo); `scripts/guard_refusal_score.py` logs the guard's verdict and its Yes/No logit; the same pass scores the refusal sheet for P6 (`results/labels/refusal/guard_logit.jsonl`).
- 2026-10-04 P15 done: `results/labels/xstest/analysis.md`, `guard_scores.jsonl` (ids, human labels, guard verdict and logit; no completion text), `scripts/xstest_guard.py`. About 7 GPU-minutes for 2,250 completions plus the 440 sheet items. On the 1,250 safe-prompt completions the guard flags 92% of human full refusals, 80% of partial refusals and 5% of full compliance: it is a broad-refusal flag. Against full refusal alone: recall 0.92 (0.88-0.96), false alarms 0.14, kappa 0.59 (two humans: 0.90). Against full or partial: recall 0.87, false alarms 0.05, kappa 0.81 (two humans, three classes: 0.88). rho^2 on the logit: 0.46 (0.39-0.53) strict, 0.70 broad. False alarms against the strict label run 0.03-0.44 across models. Consequences: (1) with strict refusal as the certificate's event, the guard is a variance reducer worth about 2x, not the label; (2) by the M4 rule the guard is above 0.3 on other people's models, so spike 018 stays deferred unless P6 finds much less on our own policies; (3) the sheet's surface pattern for refuse-then-answer is a stratifier only (34% of completions with it are human partial refusals). The same pass reproduced all 440 stored guard flags of the refusal sheet and wrote `guard_logit.jsonl`; `refusal_labels.py analyze` now reports the design-weighted rho^2 per population.
- 2026-10-04 P14 done: `scripts/stratppi_baseline.py`, `results/paper/stratppi.md` and `.json` (CPU, about 5 minutes; 5,000 draws per cell with 013's seeds, so the `b1w` column reproduces 013's validity table). (A) Reference rate as stratifier and predictor, 5 mid-rate labels x 2 sizes: `b1w` valid in 10 of 10 cells, median ESS 2.16; StratPPI as published is shorter (3.04 where valid) and over its level in 4 of 10 (up to 0.086 at delta 0.05); the same estimator with a bootstrap-t limit is valid in 10 of 10 with median ESS 2.10. So the gain is the stratification, and `b1w` stays the bound for reference-rate strata. (B) A judge's logit as stratifier and predictor, 017's refusal pools, fixed population strata: StratPPI as published is over its level in 26 of 28 cells (up to 0.239), like PPI++'s normal limit; with a bootstrap-t limit it holds in 28 of 28 and is the most efficient valid route (median ESS 2.54 against 1.67 for unstratified PPI++ bootstrap-t at a 20% rate; 1.87 against 1.39 at 5%). This changes the routing rule: when labels can be drawn after scoring, stratify on the judge's logit. Not run: StratPPI's optimal allocation; their code (the implementation is from the paper's equations). Paper: Table 2 rows, Table 3, sections 5 and 6.
- 2026-10-04 **labels lost, page fixed.** The user labelled all 220 of their items on the version-2 page and none were stored. Cause: the artifact store hands out frozen snapshots; on a return visit the page took its `labels` object from the stored document, so every new label was silently dropped while the status line said "saved to the store". The user's document existed because they had labelled 28 items on the first sheet (18:41-18:51Z) while it was being rebuilt, and those ids were reused by the new sheet. Only a first visit against an empty store had been tested. Fix (page version 3, same link): stored state is copied before use, this browser's copy and the store's are merged label by label instead of one replacing the other, documents carry the sheet version and labels from another version are set aside in `labels_v1`, writes go one at a time, the status line and the end screen report what the store actually holds, an alert appears if a label or a save fails, and a backup button is always available. `scripts/refusal_desk_test.py` runs six scenarios in headless Chrome against a mock store with frozen snapshots; the old page fails the return-visit one, the new page passes all six, including the user's real stored document. The 28 first-sheet labels are kept in `results/labels/refusal/v1_labels_ah.jsonl` (6 of those responses are also on the current sheet). The 220 labels have to be made again.

## 6. P9 analysis rules (fixed 2026-10-04, before any label exists)

1. **Policy.** The returned policy of spike 014's run (`/mnt/d/seldonian-runs/014/s0/checkpoints/feasible-step175`), against the untrained base model as the reference. One training run, one seed: the paper says so, and presents the certificate as one worked case. Round 6's returned policy cannot be used: no Round 6 results or adapters survived the 2026-09-19 loss.
2. **Event.** A human label of `r` (strict refusal) under `results/labels/refusal/GUIDELINE.md`. `h` is reported as its own rate. A second certificate on `r` or `h` is computed and labelled secondary. An unsure label counts against the certificate: as a refusal for the trained policy and as an answer for the reference.
3. **Constraint.** Trained rate minus reference rate is at most the training margin, 0.02 (014's `MARGIN`), both rates in human terms. The reference rate is not a constant: `set_relative_thresholds` measured it once with the guard, and in human terms it has to be estimated from labels like the other.
4. **Sample.** Prompts drawn at random without replacement from the safety pool (013's C1) minus the guideline's ten example prompts; for each prompt one fresh response from each policy (temperature 1.0, 256 new tokens so that fewer responses are cut off than at training's 128). The pair is the unit, so prompt clustering does not arise. Items are labelled blind to the policy, in a random order fixed before labelling starts, each by one annotator, with a shared subset for agreement.
5. **Bound and stopping.** The paired difference per prompt, rescaled to [0, 1], bounded above with the betting bound (`betting_mixture` in `seldonian/llm/constraints.py`), which stays valid when the bound is recomputed as labels arrive. Delta 0.05 (0.1 also reported). Labelling stops when the upper bound is at or below the margin (certified) or at the cap on pairs (no solution found). The cap is set from P6's rates before P9's labelling starts and is at most 600 pairs. No fixed-sample bound is computed on a partial sample.
6. **What the certificate is conditional on.** The policy was chosen on D_c and passed a guard-terms test on the same safety prompts (rate 0.168, upper bound 0.198, threshold 0.206, n 500). The human-terms test uses fresh responses, but the prompts are the same; the paper states this.
7. **Frozen code.** The P9 script is written and committed before the first P9 label, and its commit hash goes in the paper.

By the guard and the hybrid-shape pattern (a proxy, not human labels), the strict rate is about 11.8% for the reference and 3.4% for the pushed policy, and the broad rate 18.6% against 18.2%. If the human labels look like the proxy, the strict certificate needs on the order of 100-150 pairs and the secondary one will not certify within the cap.

## 7. Blind-spot review, 2026-10-04

| # | blind spot | status |
|---|---|---|
| 1 | The certificate's event was undefined (hybrids in or out) | decided: strict, section 6 |
| 2 | The threshold has no value in human terms (reference rate treated as a constant) | section 6, rule 3: both policies labelled |
| 3 | P9's first-choice policy is gone; 014's is one seed with under a point of slack in guard terms | section 6, rule 1; NSF is a reportable outcome |
| 4 | Stratified PPI and model-guided human labelling are published | P3 amended; P14 run: the published interval fails at these sizes, its estimator with a bootstrap-t limit holds and is adopted for judge strata |
| 5 | The scope line says training is out, but the main negative and P9 need a trained policy | **open**: reframe the title and section 0 around what holds and what breaks when certifying behaviour rates, with a short training section. The user has not decided. |
| 6 | Guard recall barely identified; pooled across populations in the script | 275 guard-negative items; recall per population with a conservative interval. Still wide for strict recall at step 200 (simulated lower limit about 0.2 against a truth of 0.78): a second-phase sample aimed by a second judge would be the next fix if the number matters |
| 7 | Agreement gate weak, no fallback, guideline examples on the shared set | kappa interval reported; reserve set; example prompts out of the frame |
| 8 | Prompt clustering ignored in the sheet's standard errors | clustered se beside the design se |
| 9 | Responses cut at 128 tokens shape the hybrid label | `cut_off` in the key, reported; P9 samples at 256 |
| 10 | Peeking at a fixed-sample bound as labels arrive | section 6, rule 5 |
| 11 | Annotator supply (last page: 227 labels, one annotator, 3 harmful) | **open**: P7's 1,000 harm labels and 30 positives need a decision on who labels; P15 gives refusal gold without annotators |
| 12 | The sheet could not be rebuilt | builder and page template in the repo |
| 14 | (found 2026-10-04, after the review) The labelling page was only ever tested on a first visit | `scripts/refusal_desk_test.py`; run it before any labelling page goes to annotators |
| 13 | The `h` label had no outside standing | guideline tied to XSTest's scheme |

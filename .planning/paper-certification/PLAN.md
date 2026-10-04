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
| P3 | **Related-work positioning.** Thomas et al. 2019 (Seldonian); Miller 2024 "Adding error bars to evals" and the clustered-SE line; Angelopoulos et al. PPI/PPI++; conformal and risk-control work (Bates et al., Angelopoulos "learn then test"); eval-noise position papers (Bowyer et al. 2025); stratified sampling in evals. Check every citation against the 013 `LITERATURE.md` conventions (verified, not recalled). State in one paragraph what is new here: one-sided certificates with NSF, validity checked by plasmode, stratified safety sets with a pre-flight, the judge as variance reducer with a routing rule, and the carrying negatives. | C | web, LITERATURE.md | related-work section, `.bib` | 1 day | P1 |
| P4 | **Labelling guideline.** Refusal: what counts, with the hybrid rule ("a refusal sentence followed by the answer" decided one way, with the reason) and ten worked examples drawn from 014's step-0 and step-200 samples. Harm: extend the existing `results/screen/labels/GUIDELINE.md` (encoded and plain prompts; "disclaimer then real answer" is harm). A `skip`/`unsure` option, and an opt-in note for harmful content. | U decides the rules, C drafts | 014 samples, existing guideline | `labels/GUIDELINE-refusal.md`, updated harm guideline | 1 day | - |
| P5 | **Refusal sheet.** 200 responses: 100 from 014's step-0 reference samples and 100 from the step-200 pushed policy, hybrids over-sampled (the guard's refusals at step 200 are mostly hybrids), design weights recorded per stratum as 017 did. A shared 40-item set every annotator labels (agreement), the rest split. Served as a labelling page like the Harm Label Desk (artifact with the db capability, labels at `labels/<uid>`, owner-only read), or a plain JSONL for anyone who prefers a spreadsheet. | C builds, U + A label | P4, `results/spikes/014/gen_s0.jsonl`, 013 reference samples | `results/labels/refusal/sheet.jsonl`, `labels_<handle>.jsonl` | build 0.5 day; labelling about 6 annotator-hours in all | P4 |
| P6 | **Refusal analysis.** Agreement on the shared 40 (kappa; gate 0.6 before using the labels); the human refusal rate at step 0 and step 200 with design weights; rho^2 against human labels for the guard's logit (needs the guard's logit per response: one local judge pass, about 20 GPU-minutes, logging the logit not p) and for the compiled judge (017's scores); the hybrid share. This answers 017's follow-up 1 and settles whether spike 018 is needed (M4 rule: guard rho^2 under 0.3). | C | P5 labels | `labels/refusal/analysis.md`; paper section "a certificate in human terms, part 1" | 1 day | P5 |
| P7 | **Harm positives sheet.** 1,000 responses from the capability screen (4,800), stratified by the 4B guard's score so positives are over-sampled, weights carried; the shared 40 for agreement; aim for 30 or more human positives (006's floor for bounding a judge's recall). | C builds, U + A label | P4, `results/screen/`, 017's sampling code | `results/labels/harm/` | build 0.5 day; labelling 15-20 annotator-hours | P4 |
| P8 | **Harm analysis.** Weighted human harm rate; recall and false-alarm rate of the 4B guard with Clopper-Pearson intervals; the label budgets (301/149/59) restated with the judge's measured rho^2 on the logit; the answer-rate-aware correction checked against the human rate. | C | P7 labels | `labels/harm/analysis.md`; paper section | 1 day | P7 |
| P9 | **One human-certified policy.** Round 6's returned over-refusal policy (or 014's step-175 selection): its safety set relabelled by humans under P4, stratified by the reference rate (013) and routed (017). The certificate in human terms. 300-600 labels with stratification and PPI++ against 1,200 without. | C samples, U + A label | P4, P6 (rho^2 decides the budget), the checkpoint | the paper's headline figure | sampling 0.5 day plus about 1 GPU-hour; labelling 10-20 annotator-hours | P6 |
| P10 | **Qwen3-1.7B replication of the stratified safety set** (013 dropped it for budget): the C1 pool, 8 reference samples, step-200 side-effect checkpoint, the plasmode. One more model for the 1.4-5.3x claim. | C | 013's `gen013.py`, `real_plasmode.py` | a row in the use-case map | about 4 GPU-hours local, 0.5 day | - |
| P11 | **Library.** `seldonian/llm/certify/`: the bounds (`stratbounds`, `ppipp_boot`, the cluster bootstrap-t), the strata and pre-flight (with `--pushed`), the routing rule, the design-weighted estimator, the plasmode protocol as tests, and a CLI that takes a labels file and prints the certificate. Pin a tag; rent-my-gpu imports the tag. | C | spikes 013, 017, 020 | package, tests, tag `cert-0.1` | 3 days | P2 |
| P12 | **Figures.** Validity (miss against delta, every bound, every setting); ESS against ICC_ref with the pre-flight line; the AgentDojo naive-vs-clustered plot; the RoboDojo resolution table; the carrying failures. Dataviz skill rules, no dual axes, palette validated. | C | results | `reports/figs/` | 1 day | P1-P8 |
| P13 | **Full draft, internal review, submission target.** Workshop (fast, with the scope as stated) if P9 is not in; main venue if it is. | U + C | all | v1.0 | 3 days | all |

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

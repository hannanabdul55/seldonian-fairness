# Blind review of the workshop version (AAAI-27 Trust4RSI-Agent), 2026-10-19

Fresh agent, manuscript only (reports/workshop/trust4rsi_aaai27.pdf and sources). Weak accept for the workshop, mean 3.5.

## Scores

| dimension | score |
|---|---|
| evidence relevance | 4 |
| falsifiability | 4 |
| scope calibration | 3 |
| argument coherence | 3 |
| exploration integrity | 4 |
| methodological rigour | 3 |
| **mean** | **3.5** |

Workshop: weak accept (confidence 4/5). Main track: weak reject / borderline (an empirical audit of known interval
methods; small models; one model family per finding; thin human evidence; dense prose).

Fit: good on regression testing of updates; weaker on the "agent OS" view (no sequence of updates, safety-set reuse,
rollback, staged rollout).

## Major items (reviewer's numbering)

1. Repeated gating across a sequence of updates not addressed (fresh vs reused safety sets, delta across versions,
   rollback); suggests a synthetic-bandit simulation.
2. The recommended stratified Wald-t limit's failure region on the synthetic i.i.d. grid (5 of 18, up to 0.111) is not
   characterised; give the failing profiles and a checkable guard (min expected count / min Satterthwaite df).
3. Abstract leads with extreme cells (24% at ~3 positives; 90% at a 2-label floor); lead with the typical regime.
4. Routing table prescribes the Wilson-type limit for stratified sheets while the paper recommends against it; Section 6
   does not follow the table.
5. "Full validity table" omits results quoted in the main text (confirmation pools, routing rule, betting on blocks,
   the smaller-of-two rule); Table 1's StratPPI bootstrap-t row vs the Introduction's "2 of 32".
6. "A second judge family places the failure in the compiled judge" overstates; the WildGuard predictions were
   refuted; the side-effect control is post hoc; weak judge vs rubric judge.
7. Ground truth is a guard; one author-annotator; blinding not stated; per-policy n not given.
8. AgentDojo certificate assumes attacks fixed; the quadrature bound not discussed in the main text; "(repeat)"
   pipelines unexplained.
9. The training runs' relative constraint was gated as an absolute threshold (reference rate plugged in).
10. Missing related work: regression-free / backward-compatible model updates (Bansal et al. 2019; Srivastava et al.
    2020; Yan et al. 2021; Xie et al. 2021), paired non-inferiority intervals (Tango), small-proportion intervals in
    complex surveys (Korn and Graubard 1998), misclassification-corrected prevalence intervals (Lang and Reiczigel 2014),
    PPI under distribution shift.
11. Hard to read: abstract ~280 words; nonstandard terms; no gate diagram; no recommendations table.

## Plain errors (reviewer's list, 20 items)

1 "Its width is zero" antecedent; 2 the 1.5-5.0 range's measure; 3 "mid rate" defined inconsistently; 4 pair-rate 0.060
cited to a figure/table that lacks it; 5 16% vs 82% abstention; 6 mechanism shares vs counts; 7 strict vs broad false
alarms; 8 "2 of 32" untraceable; 9 Table 1 row names and grouping; 10 "seeds of one run"; 11 46% caveat in abstract;
12 two prediction families both labelled Q; 13 Table 2 wording ("8 labels", rare row, "lost", dashes); 14 Table 7 bold
rule; 15 pre-flight gains belong to the Wilson-type limit; 16 the 120-responses break-even scales with pool size;
17 seed retry deviation omitted; 18 provenance of rows 1, 3-5, 15; 19 appendix tables overflow; 20 terms used before
definition (stratified Wald-t, plasmodes, PPBoot, quasi-Seldonian, betting on blocks, reference rate, pre-flight,
canonical wording, XSTest, power ESS).

Anonymity: none found; do not publicise the repository before the decision (dates in the record table).

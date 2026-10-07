## 1. Summary of claims

The paper recasts the held-out Seldonian safety test as a certificate for a fixed language-model policy’s behaviour rate. It audits confidence bounds, studies label savings from stratification and prediction-powered inference, demonstrates failures under calibration shift and dependent sampling, and applies bounds to published benchmarks. A human-label certificate is designed and costed but not executed.

The practical questions are worthwhile, and the disclosure of unsuccessful preregistered choices is unusually good. However, the central validity claims exceed the evidence, and one reported Wilson-bound result is mathematically incompatible with the stated procedure. Substantial correction and independent confirmation are needed.

## 2. Technical correctness

### Seldonian guarantee: correct core, incomplete applicability

The argument in §§2 and 4.4 is correct **if the safety bound has conditional coverage given everything used to select the candidate**. Independence between constraints is unnecessary; the union bound suffices. The guarantee controls the unconditional probability of returning a violating policy, not the probability of violation conditional on returning one. The introduction’s “chance that the statement is wrong” should make this distinction explicit.

Conditioning should include training randomness, as the surrounding prose acknowledges. More importantly, reference scores collected across the entire prompt pool, rank-based strata, and stratified candidate/safety allocation do not automatically preserve the simple independence argument (§§6.4, 8.1). A fixed-population sampling argument or a hierarchical superpopulation argument is needed. “Before training” is not equivalent to “independent of candidate selection.”

The claim in §5 that a method without abstention “cannot promise anything” is false: a known feasible fallback can always be returned. Abstention enables this particular construction without requiring such a fallback.

### A decisive inconsistency: rare-rate Wilson coverage

The pooled Wilson definition in §6.3 is standard. With zero positives its upper endpoint is

\[
U_0=\frac{z_{0.95}^2}{n+z_{0.95}^2}.
\]

At \(n=100\), this is approximately \(0.02634\). All other counts produce larger endpoints. Consequently, at a true rate of 1–2%, the stated pooled Wilson bound **cannot miss at all**, regardless of the sampling distribution of the count.

Table 4 and §7.1 nevertheless attribute miss rates of 0.076–0.448 to “b1w and the pooled Wilson bound” at 1–2% with \(n_s=100\). This cannot be explained by binomial discreteness. Either the implementation, cell description, denominator, quantile, or method attribution is wrong. The upper end resembles the zero-event failure of a plug-in Wald interval, not this Wilson interval. This must be resolved before the broader audit is trustworthy.

### Bounds and implementation details

The Clopper–Pearson formula (§4.4) is correct for \(k<n\), including the stated zero-event formula. Specify \(U=1\) when \(k=n\). The sample requirements 299, 149, and 59 are correct.

“Exact” should mean finite-sample coverage at least nominal, not equality with nominal coverage. Clopper–Pearson is binomial; sampling prompts without replacement from a fixed heterogeneous pool is not binomial sampling. Conservative extensions may apply, but their assumptions and proofs must be supplied rather than inherited from the i.i.d. statement.

The stratified b1w construction (§6.3) is heuristic. Once clipping occurs,

\[
\sum_h W_h p_h(m)
\]

need not equal \(m\), so the variance is not generally evaluated at a parameter configuration whose mean is the hypothesised mean. Specify endpoint/root conventions and distinguish this from a genuine stratified score-test inversion.

The PPI++ estimator and variance expressions are correct for a fixed coefficient and independent labelled/unlabelled samples. The displayed coefficient is estimated from the labelled data, however, so exact unbiasedness does not follow from the fixed-\(\lambda\) argument. Estimated-coefficient inference is generally asymptotic. Clarify coefficient clipping, feature fitting/cross-fitting, constant-feature handling, and whether all fitting is repeated inside each bootstrap.

The bootstrap-t sign convention is correct. Its operational definition is incomplete: what happens when \(V=0\) or \(V_b^*=0\), especially with rare events or sparse strata? Discarding degenerate replicates and retaining infinite statistics can yield very different endpoints. “Approximate (second order)” in Table 4 requires regularity conditions not established for these lattice-valued, boundary-prone problems.

The cluster standard-error formula is the usual ratio-estimator sandwich expression. Its target is an episode-weighted rate, not necessarily the mean user-task rate when cluster sizes vary. State that estimand explicitly.

### Data-dependent routing is itself an inferential procedure

Table 5 chooses Clopper–Pearson or PPI++ according to the observed class counts. Fixing that rule in advance does **not** make it data-independent. Marginal validity of two component bounds does not establish validity of their selected combination.

Audit the entire routed procedure, particularly around ten observed events. Nor can the Clopper–Pearson branch automatically be labelled “exact” as a conditional statement given selection into that branch. Similar care is needed if §10.4’s four analyses permit certification whenever any one passes; reporting four estimates descriptively is different from using four opportunities to certify.

### ESS, calibration, and budget derivations

The oracle PPI variance-reduction formula in §6.4 is correct under its independence and optimal-coefficient assumptions. The width-ratio ESS is only an approximate sample-size equivalence; it is not literally “what the baseline does with \(2n\)” for discrete or nonlinear bounds. Switching between distance from the estimate and distance from truth further compromises comparability.

The reference-stratification expression involving ICC, \(\rho^2\), reliability, and \(c_H\) is a model-based approximation, not a general identity. It also uses candidate ICC and reference–candidate correlation, quantities not ordinarily known before observing candidate responses. The claimed “pre-flight” prediction needs an explicit operational estimator.

The calibration identity (§6.5) is correct, but confidence-bound inversion needs directions, feasibility constraints, and treatment of \(s\approx a\). For \(s>a\), the usual upper bound involves an upper limit for \(q\) and lower limits for \(s,a\), subject to compatible parameter values. Simply invoking three one-sided bounds is insufficient.

Table 10 is approximately reproducible using

\[
n\simeq
\frac{(z_{1-\delta}+z_{\mathrm{power}})^2\operatorname{Var}(d)}
     {(\tau-\Delta)^2}.
\]

The approximately 416 paired observations follow from the stated assumptions. But correlation between two guard flags is not necessarily correlation between human labels. Likewise, per-policy human–guard \(\rho^2=0.6\) does not establish \(\rho^2=0.6\) between the paired human difference and guard-logit difference. These assumptions drive the favourable budget estimates and require sensitivity analysis. “At +0.02 nothing certifies” should instead say that the desired power is unattainable under this approximation; a realised interval can still pass.

Finally, §4.3 contains two inaccurate theoretical summaries: GRPO depends on normalised reward magnitudes, not only rankings; and feasibility of averaged Lagrangian iterates requires assumptions and convergence conditions absent in nonconvex neural-policy optimisation.

## 3. Experimental validity

### The miss-rate audit measures non-rejection, not demonstrated validity

The two-standard-error rule (§6.1) is a reasonable exploratory alarm, but “not detected above nominal” does not establish coverage. At \(R=4{,}000,\delta=0.05\), the threshold is approximately 0.05689, consistent with the stated 0.057. Report binomial confidence intervals for miss probabilities and distinguish theoretical validity, empirical compatibility, and demonstrated undercoverage.

With hundreds of cells, some alarms are expected even for nominal procedures. Multiplicity cuts both ways: a marginal failure is weak evidence, while selecting “usable” methods and regimes after inspecting coverage makes their apparent success optimistic. The post hoc 322-cell subset expressly excludes regimes where the selected methods fail. This is not an independent validation of the resulting recommendations.

The special defence of the b1w sheet cell—row-wise multiplicity correction and pooling with a repeat—is not applied uniformly elsewhere. Predefine how repeated simulations are combined and apply any correction consistently. Large failures such as 24% misses remain compelling; marginal classifications need restraint.

### Plasmodes and finite-population targets

The plasmodes preserve useful empirical structure but do not establish performance on new prompt populations. Sampling without replacement from 500 prompts, sometimes at a substantial sampling fraction, can make bounds conservative relative to fresh-population sampling. Small response pools also suppress or condition away uncertainty.

The change of truth to the sampled response half (§8.1) may be appropriate for avoiding leakage, but it changes the target. Provide a complete data-flow diagram showing response splitting, strata construction, method selection, and coverage evaluation. A genuinely untouched pool is needed to validate the adopted stratification rule.

Bootstrap repairs should be tested against targeted stress cases: zero and one event, approximately ten minority-class observations, constant predictors, small strata, unstable estimated coefficients, and strongly concentrated cluster failures. The current empirical-population bootstrap cannot establish a general repair theorem.

### Benchmark and human-label limitations

AgentDojo’s cluster analysis is useful, but its audit validates procedures under resampling the observed table—not the assumption that curated user tasks are a random sample from deployment-like tasks. The number of clusters, suite structure, missing pairs, and weighting should be reported. Selecting the lone passing pipeline among 28 also requires clarity about marginal versus simultaneous certification; the reported intervals are not automatically a familywise guarantee.

For RoboDojo-RC (§10.1), the paper mentions task structure elsewhere but applies binomial bounds without justifying trial independence and exchangeability. This risks reproducing its own AgentDojo criticism. A benchmark citation, version, trace provenance, and sampling protocol are missing.

The author-labelled sample (§10.3) is useful pilot evidence, not a definitive human construct validation. Different truncation rates and lengths create both measurement changes and policy-identification cues. The step-200 pilot versus step-175 target mismatch (§10.4) weakens budget transfer further.

### Numerical and cross-section checks

Several checks are reassuring: Table 8’s totals reconcile to 172 full refusals, 123 partial refusals, and 308 guard flags; the reported recalls and kappas are approximately consistent. The six designated approximate groups in Table 4 do sum to 322 cells, with 23 unresolved.

Other presentation problems remain:

- The abstract’s “up to 24%” for PPI++/StratPPI omits StratPPI allocation variants reaching 90% in Table 4. Qualify it as proportional allocation.
- Table A1 gives ESS 4.75 where §8.1 reports approximately 5.1. The caption explains different draws/definitions, but a headline comparison should use one analysis.
- Table 4 omits procedures subsequently described as valid, including the finite-sample judge-assisted bounds in §9.4 and Appendix C’s paired analyses, despite claiming to cover every bound used.
- Counts by checkpoint, maxima over checkpoints, and exclusions differ across Table 4, §8.1, and Appendix A. Supply one machine-readable cell inventory.
- Lost draws preclude independent checking of several historical classifications. These should be clearly segregated from reproducible evidence.

## 4. Novelty and positioning

The main proposal is not a new inference method: an independent one-sided confidence bound followed by a threshold decision is classical hypothesis testing and the existing Seldonian safety test. The promising contribution is an application-focused empirical audit, particularly calibration failure after label-targeting training and the practical consequences of crossed benchmark dependence.

The related-work discussion should acknowledge:

- Classical survey sampling, poststratification, Neyman allocation, difference estimators, and generalised regression estimation.
- Classical misclassification correction, including the Rogan–Gladen prevalence identity underlying §6.5.
- Multiway cluster-robust inference and crossed-array/pigeonhole bootstrap methods.
- Binomial interval coverage studies and the limitations of studentised bootstrap inference for discrete rare events.
- Adaptive holdout reuse, simultaneous testing, and confidence-sequence approaches.

The contrast with risk-control work is overstated: conditioning on a separately trained model is routine, and certifying a trained policy rather than a wrapper is not itself a new statistical principle. “A stratifier supplied for free” also conflicts with the 4,000-reference-response cost acknowledged in §8.1.

## 5. Clarity and structure

The manuscript is candid but too diffuse. Sections 2–4 devote substantial space to training that is not the contribution, while crucial sampling and bootstrap details remain unspecified.

Reorganise around: estimand and guarantee; sampling designs; bound audit; independent applications. Move most training mechanics and speculative mechanisms to appendices. Replace “holds,” “honest limit,” and “repair” with language matching the evidence. A correctly constructed one-sided error bar plus a decision rule is precisely the proposed certificate, so the conclusion’s “a rate with an error bar is not a certificate” needs qualification.

## 6. Prioritised required changes

### Major

1. **Resolve the impossible pooled-Wilson result** (§§6.3, 7.1; Table 4), with analytic unit tests and regenerated cells.
2. **Specify and validate complete procedures**, including routing, coefficient fitting, zero-variance bootstraps, and multiple certification opportunities (§§6.3, 8.2, 10.4; Table 5).
3. **Separate finite-population and superpopulation guarantees**, proving the relevant sampling assumptions (§§4.4, 6.2–6.4, 8.1, 10.1–10.2).
4. **Run independent confirmation** of the post hoc stratification and interval recommendations (§8; Table 4).
5. **Replace coverage non-rejection with calibrated evidence**, reporting intervals, multiplicity conventions, and a complete cell inventory (§§6.1, 7.1).
6. **Substantiate the human-label budget**, using paired human pilot information or broad sensitivity analyses; ideally execute the certificate with independent annotators (§10.4; Table 10).
7. **Provide reproducible benchmark provenance and audited implementations**, including Robot trial structure and comparison-method validation (§10; Appendices A, D).

### Minor

8. Correct the GRPO, Lagrangian, abstention, and boundary statements (§§4–5, 10.4).
9. Standardise ESS definitions and reconcile abstract/table scope (§6.4; Tables 4, A1).
10. Expand related work and shorten the training exposition (§§2–4, 12).

## 7. Overall verdict

- **NeurIPS/ICML main track: 3/10 — reject.**
- **Workshop: 5/10 — borderline; potentially valuable after correctness fixes.**
- **Confidence: high** on the mathematical and design criticisms; moderate on implementation-level explanations because code and figures were unavailable.

**Single biggest risk to acceptance:** the paper’s central product is a validity audit, but the stated Wilson procedure cannot produce one of its headline failure results. Until that discrepancy is resolved and the corrected recommendations survive independent testing, the audit itself is not sufficiently reliable.
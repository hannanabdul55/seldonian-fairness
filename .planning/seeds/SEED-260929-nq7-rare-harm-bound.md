---
id: SEED-260929-nq7
status: dormant
planted: 2026-09-29
planted_during: unknown
trigger_when: when a harm constraint (rate below about 5%) must be certified by the Seldonian safety test
scope: small
---

# SEED-260929-nq7: Tightest valid bound for a rare harm constraint, and the safety-set size it needs

## Why This Matters

**Corrected 2026-10-06.** Spike 013 reported that at harm rates of 1-2% the Wilson-type
`b1w`, pooled or stratified, missed 0.24-0.45 at n_s 100. That was a bug: `b1w` returned its
estimate when no positive was drawn (`reports/b1w_fix_and_audit_2026-10-06.md`). With the
bound fixed it is not over its level at delta 0.05 in those cells, and over in one of twelve
at delta 0.1. What still holds at rare rates: the project's t-test and the Wald-type limits
return 0 at zero positives and do miss far more often than delta, and the Wilson limit's
exact miss probability peaks at 0.069 (delta 0.05) and 0.196 (delta 0.1) just above its
zero-count limit (`results/paper/wilson_exact.md`). Exact and distribution-free bounds stay valid there but gain nothing
from stratification (013: no distribution-free stratified bound beat pooling).

So certifying any harm constraint in the project needs two answers:

- Among the bounds already in `seldonian/bounds.py` (Clopper-Pearson, `betting_mixture`,
  `chernoff_kl`, `convex_order`, Bentkus), which is tightest *and valid* at rates of 0.5-5%?
- What safety-set size does a harm threshold of, say, reference + 2 points need for a useful
  solution rate?

Round 6 certified harm alongside over-refusal, so this bears on every harm result the paper
reports. It was split off from spike 014 (queued 2026-09-29), which tests stratification on
over-refusal only.

## When to Surface

**Trigger:** when a harm constraint (rate below about 5%) must be certified by the Seldonian
safety test.

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Small.** It runs on CPU. Spike 013's real harm labels (`results/spikes/013/judged_full.jsonl`,
C2 gated and C3 unsafe, 8 + 3x16 samples per prompt) feed the plasmode directly
(`013-stratified-safety-set/plasmode.py`), plus a synthetic rate grid for the sample-size
curve.

## Breadcrumbs

- `.planning/spikes/013-stratified-safety-set/validity_real.md`: rare-rate coverage (regenerated 2026-10-07)
- `.planning/spikes/013-stratified-safety-set/check_b1.md`: the "H4 rare" rows
- `results/paper/wilson_exact.md`, `tests/test_bound_endpoints.py`: exact miss probability and end-point tests
- `seldonian/bounds.py`: the exact and distribution-free bounds to compare
- `.planning/spikes/012-rerandomized-split/`: the t-test zero-width trap at rates of 0 or 1

## Notes

Captured from a /gsd-explore session (2026-09-29), with its trigger, why and scope filled in.

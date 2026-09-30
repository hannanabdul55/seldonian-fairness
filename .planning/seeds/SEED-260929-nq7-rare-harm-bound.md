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

Spike 013 found that at harm rates of 1-2%, every approximate bound (the project's t-test,
Wald, and the Wilson-type `b1w`) misses far more often than delta. The random-split baseline
missed 0.45 at n_s 100 on plain-PKU harm, and 0.24 on encoded gated harm. That is the bound
failing, not the split. Exact and distribution-free bounds stay valid there but gain nothing
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

- `.planning/spikes/013-stratified-safety-set/validity_real.md`: the rare-rate coverage failures
- `.planning/spikes/013-stratified-safety-set/check_b1.md`: the "H4 rare" rows (discreteness)
- `seldonian/bounds.py`: the exact and distribution-free bounds to compare
- `.planning/spikes/012-rerandomized-split/`: the t-test zero-width trap at rates of 0 or 1

## Notes

Captured from a /gsd-explore session (2026-09-29), with its trigger, why and scope filled in.

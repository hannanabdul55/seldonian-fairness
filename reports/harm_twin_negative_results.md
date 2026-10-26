# Harmful-twin experiments: a negative result

**Status:** shelved 2026-10-22. Not submitted to any venue. Registration and run artifacts
are preserved (see "What is preserved" below). This note records what was tried, what was
measured, and why the idea does not work, so it is not re-proposed without a new angle.

## The idea

Train a second LoRA adapter (the "twin") on the same frozen base as a main model `p`, to
serve as an importance-sampling (IS) proposal for certifying the main model's harmful-breach
rate. The twin's KL-regularized RL objective

    reward = log(J + 1e-6) + log p(y|x) - log q_old(y|x)

has optimum `q* = p_H` (the main model conditioned on harm), value `log P(H|x)`. The hope
was that sampling breaches from the twin instead of from `p` would make a high-confidence
upper bound on the breach rate much tighter than plain Monte Carlo at the same sample budget.

Two experiments tested this, in two different sample spaces. Both are decisive negatives.

## E1 — response-space twin (refuted)

The twin proposes **responses** `y` for a fixed prompt; the certificate reweights by
`p(y|x) / q(y|x)`. For that bound the proposal must *cover* the main model's harmful mass,
i.e. small forward KL `KL(p_H || q)`.

Measured on the 41 brute-force prompts whose `P(H)` the reference could resolve
(`/mnt/d/seldonian-runs/twin/e1/forward_kl.json`):

- forward `KL(p_H || q)` for the trained twin: **median 187.2 nats** (mean 191.4, range
  130.6–232.0).
- the base model `p` itself: forward `KL(p_H || p) = -log P(H) ~ 2 nats` on these prompts.
- a warm-start SFT twin vs the RL twin were per-prompt near-identical (e.g. 204.5 vs 208.7),
  so the objective (RL vs MLE) is **not** the culprit
  (`forward_kl_warm_vs_rl.json`).

The RL twin is ~180 nats *worse* a proposal than the base model it was trained from. Cause:
RL is mode-seeking (minimizes reverse KL), so the twin collapses onto short harmful outputs
with `log q > log p`, while `p`'s actual harmful samples are long with `log q << log p`. The
IS weight `p/q` then explodes on exactly the samples that carry the breach mass. As an IS
proposal the response-space twin is worse than sampling from `p`.

## E6 — prompt-space twin + defensive-mixture certificate (valid, not competitive)

Reframed to the space where "need" and "trainability" might align: the twin scores
**prompts** `x` (a logistic reward model on the frozen base's mean-pooled last hidden state
`phi(x)`, isotonic-calibrated), and the certificate uses a defensive mixture
`m(x) = alpha*D(x) + (1-alpha)*q(x)`, summand `w*1_H` in `[0, 1/alpha]`, certifying
`B = E_{x~D}[P(H|x)]`. Registered as E6 in `.planning/harm-twin/REGISTRATION-E1-E5.md`
(2026-10-21). Reference: 1200 prompts (300 PKU-adversarial candidate split + 900 benign),
32 WildGuard-labelled responses each, 36,148 s (~10.0 h), clean. Results in
`results/twin/e6/`:

| prediction | target | result | |
|---|---|---|---|
| E6a validity | coverage >= 0.90 all cells | 0.99–1.00 everywhere | PASS |
| E6b efficiency | betting width <= 0.5x CP @ (pi=.05, n=300, alpha=.5) | 0.0308 vs 0.0126 (2.4x wider) | FAIL |
| E6c generalization | Spearman rho >= 0.40 on M_cert | rho = 0.301 | FAIL |
| E6d gate | var_ratio <= 0.5 AND missed < 0.25 @ (pi=.05, alpha=.5) | var_ratio 0.609 (missed 0.181 ok) | FAIL |

The decisive line is the proposal comparison: at every cell the learned twin is
statistically indistinguishable from both the zero-training lexical baseline and the
**uniform** proposal, e.g. at (pi=0.05, alpha=0.5, n=300):

    twin bet_w 0.0308   lexical 0.0312   uniform 0.0314   | plain-MC Clopper-Pearson 0.0126

So the reward model buys nothing over drawing prompts uniformly, and all three mixtures are
1.35x–6.0x *wider* than plain Monte Carlo (corrected 2026-10-25; an earlier
version said 2.4x–5.8x).

Mechanism: `B` is a rare event (~0.018–0.028), so a plain binary Clopper-Pearson at n=300 is
already tight. The mixture summand lives in `[0, 1/alpha]`, and the empirical-Bernstein /
betting range penalty from that `1/alpha` ceiling swamps any variance saved. The twin only
tilts the variance ratio to ~0.57–0.61 (barely below 1), and uniform sampling tilts it just
as far, because at rho = 0.30 the reward model cannot pick out the harmful prompts a uniform
draw would miss.

## Conclusion

A KL-regularized twin gives a **valid** certificate but is not a competitive IS proposal for
a rare breach rate — refuted in response space (E1: forward KL ~185 vs `p`'s ~2) and shown
valid-but-no-better-than-uniform in prompt space (E6: twin ≈ uniform, 1.35–6.0x wider than
plain MC). For rare, low-variance breach quantities, plain Monte Carlo with a tight binary
bound is hard to beat with a bounded-weight mixture, because the `1/alpha` range penalty
costs more than the variance it saves. The right diagnostic is missed harmful mass and the
variance ratio, not `KL(p_H || q)`.

This does not rule out a twin helping in a different regime (e.g. `P(H)` small enough that MC
itself is the bottleneck, or a stronger prompt feature than mean-pooled `phi`). No such
regime was tested; if one is pursued it is a new registration, not a retry of E1/E6.

## What is preserved

- Registration: `.planning/harm-twin/REGISTRATION-E1-E5.md` (E1–E5 + the 2026-10-21 E6 section).
- E1 artifacts: `/mnt/d/seldonian-runs/twin/e1/` (brute-force reference, adapters, forward_kl*.json).
- E6 artifacts: `/mnt/d/seldonian-runs/twin/e6/` (pool, 1200-prompt reference, features) and
  `results/twin/e6/` (report.md, e6.json, run.log).
- Drivers: `scripts/twin_e1.py`, `scripts/twin_e6.py`, `scripts/run_twin_e1.sh`,
  `scripts/run_twin_e6.sh`.
- Prior write-ups (not submitted): `reports/proposal_adversarial_harm_twin.md`,
  `reports/aaai_harm_twin_brief.md`.

## Stage A — is it the model and prompts? (registered 2026-10-22, NO-GO)

Question: E6 used a mildly aligned 1.7B model and weak single-turn prompts. Does a setting
with real attack structure change the answer? Zero-GPU reanalysis of spike-013 step 0
(Granite-3.3-2B, 1,000 PKU prompts in 7 attack families, 24 Qwen3Guard-4B labels each),
registered in `.planning/harm-twin/REGISTRATION-STAGE-A.md` before the split was computed.
Results in `results/twin/stage_a/`.

- Gated harm (the real label): leetspeak 8.2%, plain 0.9%, the other five encodings 0%.
  Family explains 4.3% of the Monte Carlo variance (P1 NO-GO); 62% of the variance is
  within-prompt noise that no prompt selection can remove.
- Family-stratified Clopper-Pearson at n = 300 is 28.6x (oracle Neyman) / 6.5x
  (proportional) wider than plain CP (P3 NO-GO). Zero-rate strata each pay a
  "cannot certify zero" floor at delta/7.
- The non-refusal label is strongly structured (leetspeak 44%, family explains 23.5% of the
  variance, more than E6's oracle 19.7%), and stratifying it is still 4.5x wider.
- Post hoc, the most favourable split ({leetspeak, rest}) at its best allocation is still
  1.27-1.33x wider than plain CP.

So the negative is not specific to the model: at these breach rates and budgets, plain Monte
Carlo with an exact binary bound beats every stratified or reweighted certificate tried,
including one built on real, design-time attack structure. This matches spike 013's earlier
finding that rare labels (1-2%) get no stratification gain. Stage B (Qwen3-4B-Instruct-2507,
~10-12 GPU-h) is not licensed by this result and is not planned.

## Conclusion of the line (2026-10-23)

Closed. All three certificates above certify the **judge's** flag rate (WildGuard in E1/E6,
Qwen3Guard-4B in Stage A), and for that quantity the comparison is settled: plain Monte Carlo
with an exact binary bound wins. Every loss had the same mechanism: a stratum or mixture
component whose rate is near zero still costs many labels to bound near zero, and splitting
delta across strata makes each of those bounds looser.

## Angles not tested

1. **The labels are the judge's, not ground truth.** On the user's 225 encoded-sheet labels the judges'
   false-alarm rate was 18-23%; their recall on harm is unmeasured (3
   positives). Responses to *benign* prompts are flagged at least as often as responses to
   plain harmful ones (Stage A: 1.36% vs 0.90%; E6: benign 1.5%), which suggests much of
   the measured "breach" mass on these pools is false alarms. Cleaner labels most likely make
   harm rarer and more concentrated; the E6 gate's rarefied test (Part 3) found plain CP
   gains the most in exactly that case, so better labels are unlikely to rescue a sampler.
   They do change what the certificate means.
2. **Human labels as the bottleneck.** With judge labels nearly free there is nothing for a
   sampler to save. If ground truth is a human label, labels are scarce and the judge (or a
   twin) becomes the stratifier for where to spend them. The C1 human certificate already
   shows the same failure there: its cleared stratum (0/70 labelled, 4,809 responses) makes
   the exact bound 0.20 against an estimate of 0.106. The open problem in that regime is the
   certificate for the near-zero stratum, not the sampler.
3. **A per-family (worst-subgroup) certificate.** Stage A found real family structure
   (leetspeak 8.2% vs plain 0.9% gated; 44% vs 0.9% non-refusal). An average-rate
   certificate hides it; a certificate on the worst family, or on each family, is where that
   structure is the point rather than a nuisance. **Tested 2026-10-23: GO** (registration
   `.planning/harm-twin/REGISTRATION-PER-FAMILY.md`, results `results/per_family/`): choosing
   where to spend labels certifies every family with 0.35x the labels of uniform sampling.
4. **A joint stratified bound.** Every stratified certificate here was per-stratum
   Clopper-Pearson at delta/K. A single exact bound on the weighted sum would avoid the
   delta split. Spike 013 found no distribution-free stratified bound that beat pooling at
   mid rates; rare rates with a joint exact bound were not tried.

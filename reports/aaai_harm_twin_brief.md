# Certifying a Language Model's Breach Rate with a Likelihood-Coupled Harmful Twin

**Brief paper (short-paper form), prepared for AAAI. Draft 2026-10-20.**

*Status: theory and the exact-toy validation (E0) are complete; the large-model experiments E1–E5 are pre-registered (`.planning/harm-twin/REGISTRATION-E1-E5.md`, 2026-10-18) and in progress. No large-model result is reported below; registered predictions are stated as predictions, not findings.*

## Abstract

We study how to put a high-confidence upper bound on the rate at which an aligned
language model emits content a harm judge flags — its *breach rate* `P(H)` — in the regime
where that rate is small (≈1%) and plain Monte Carlo would need tens of thousands of clean
samples to certify. Our instrument is a *harmful twin*: a second LoRA adapter on the same
frozen base as the model under test, trained by KL-regularised reinforcement learning to
write judge-flagged responses, with a KL penalty of coefficient one to the *live* model
under test. A standard control-as-inference identity makes the twin's optimum the model
*conditioned on harm*, `p_H`, and its objective value the log breach probability `log P(H)`.
From this we derive three uses of one object: (i) the twin is an amortised sampler of the
model's own harmful tail; (ii) importance-corrected twin samples give a variance-reduced
estimate of the gradient of `log P(H)`, which drives a defender step that does not stall as
the model becomes safe; and (iii) a *defensive-mixture* importance sampler using the twin as
proposal yields a confidence upper bound on `P(H)` whose validity does not depend on the
twin's quality — only its width does. We prove each identity on an exact tabular toy where
truth is computable by enumeration (7/7 checks pass), and we pre-register a protocol on
1.7B–2B models to test the one assumption that can refute the bound: that the trained twin's
forward KL to `p_H` is small enough for the sample size to be feasible. The novelty is
narrow and stated as such: the RL identity is standard; what is new is anchoring the twin to
the *live* model, alternating it with a defender, and turning it into a twin-independent
certificate of the breach rate.

## 1. Introduction

Safety evaluation of a generative model usually reports a point estimate of some harm rate
from a sample of outputs. When the rate is deliberately small — a well-aligned model under
adversarial prompts might breach on the order of 1% of the time — a point estimate is both
noisy and, for a *safety* claim, pointing the wrong way: a developer needs a high-confidence
*upper* bound on the breach rate, and getting one to within a factor of two at `P(H) ≈ 0.01`
by naive sampling takes on the order of 10⁴ labelled generations per model checkpoint.

Red-teaming addresses the related but distinct problem of *finding* breaches. A learned
attacker that maximises a judge score will reach harmful regions, but it tells you nothing
calibrated about how *likely* the model under test was to produce that content on its own —
it is an attacker, not an estimator. Our question is whether a single adversarial object can
be made to serve estimation and certification at once.

We answer yes, under a known identity used in a specific way. Train a twin `q` — a second
adapter on the same frozen base as the model `p` under test — to maximise

```
F(q) = E_{y ~ q}[ log 1_H(y) ] − KL( q(·|x) ‖ p(·|x) ),            (1)
```

the expected log harm indicator minus a unit-coefficient KL to the *current* `p`. This is
exactly KL-regularised RL with `p` as the reference [korbak2022rlkl] [khalifa2021gdc]. Its
optimum is `p` tilted onto harm,

```
q*(y) = p(y | x) · 1_H(y) / P(H | x) = p_H(y | x),                 (2)
```

and its optimal value is `log P(H | x)` (the log-partition / Gibbs variational identity).
The twin is therefore not merely an attacker: at convergence it is the model's own harmful
conditional, and its objective value is calibrated to the model's harmful mass.

**Contributions.**

1. *A calibrated adversary.* We make precise, and validate on an exact toy, that the
   KL-anchored harm objective (1) has optimum `p_H` and value `log P(H)`, and we enumerate
   the assumptions that break the identity (length normalisation, a KL to a fixed base, an
   additive rather than log harm term, a soft judge under the entropy form).
2. *A defender step that does not stall.* At the twin's optimum,
   `∇_θ log P_θ(H) = E_{y ~ p_H}[∇_θ log p_θ(y)]`, so lowering the model's likelihood on
   importance-corrected twin samples is unbiased descent on `log P(H)`. The on-policy
   baseline — train the model on its own judged samples — loses its gradient once the model
   refuses confidently and most sampled groups contain no breach; the twin keeps a usable
   gradient because the *excess* variance from importance reweighting vanishes as `q → p_H`
   (the weights become constant on H) — the estimator's own variance does not vanish.
3. *A twin-independent certificate.* A whole-sequence defensive mixture
   `m = α p + (1−α) q` with the twin as proposal gives an unbiased estimator of `P(H)` whose
   summand lies in `[0, 1/α]`, so empirical-Bernstein and betting bounds apply. Validity
   holds for *any* twin; only width improves as the twin approaches `p_H`.
4. *A pre-registered test of the one fragile assumption.* The certificate's sample cost
   grows like `exp(KL(p_H ‖ q))`. We pre-register a protocol whose first decisive experiment
   measures this forward KL directly on real models, before any bound is claimed.

## 2. The twin objective and its optimum

**Setting.** Prompts `x ~ D`. The model `p` and twin `q` are two LoRA adapters on one
frozen base; only adapters train. A judge `J(x,y) ∈ {0,1}` defines the harmful set
`H(x) = {y : J=1}`; everything is relative to the judge's label, not a platonic notion of
harm. All sequence quantities are summed over tokens with a fixed length cap `L`. The breach
rate is `P(H) = E_{x~D}[P(H|x)]`, with `P(H|x) = E_{y~p(·|x)}[1_H(y)]` at temperature one and
no truncation.

**Derivation.** Expanding the KL in (1),

```
F(q) = E_{y~q}[ log 1_H(y) + log p(y|x) − log q(y|x) ].
```

Maximising over distributions with a Lagrange multiplier for normalisation gives
`q*(y) ∝ p(y) exp(log 1_H(y)) = p(y) 1_H(y)`, i.e. (2); and the maximised value is the
log-partition `log E_{y~p}[1_H(y)] = log P(H|x)`. A suboptimal twin gives
`F(q) = log P(H|x) − KL(q ‖ p_H)`, an evidence lower bound on `log P(H|x)` whose gap is the
reverse KL to the target [korbak2022rlkl] [zhao2024twisted].

**What the identity needs** (each is a known way to break it):

- *Summed, not length-normalised, log-probs.* Per-token averaging gives a length-dependent
  effective coefficient and no proper optimum short of the length cap.
- *Coefficient one, to the live `p`, and no KL to a fixed base.* A KL to the base tracks the
  base, not the current model; a KL to both double-counts the likelihood. Under an indicator
  judge a coefficient `β ≠ 1` leaves the optimum at `p_H` and only rescales the value to
  `β log P(H)` (since `q* ∝ p·J^{1/β}` and `J^{1/β}=J`); the *entropy* form is not robust
  this way, tempering the optimum to `(pJ)^{1/β}`.
- *Log harm term.* An additive bonus `λ J` yields a twin that is mostly benign unless
  `λ > log(1/P(H))`; a smoothed `log(J+ε)` needs `ε ≪ P(H)`.

**Implementation.** We realise (1) as GRPO with reward
`r(y) = log(J+ε) + log p(y|x) − log q_old(y|x)`, `ε = 10⁻⁶`, both sequence log-probs from one
extra forward pass each, TRL `beta=0` (no second KL), `scale_rewards="none"` (group mean as
baseline; dividing by the group s.d. would rescale the coefficient), and a token-summed loss
(`loss_type="dr_grpo"`; length normalisation would break the identity). A warm start on a few
hundred judge-flagged samples is required: REINFORCE from `p` alone sees no gradient when `H`
has tiny probability and sits at `p`.

## 3. The defender step

At `q = p_H`, the breach-gradient identity `∇_θ log P_θ(H) = E_{y~p_H}[∇_θ log p_θ(y)]` makes
likelihood descent on twin samples unbiased, with no baseline since `E_p[∇ log p] = 0`. Away
from the optimum, self-normalised importance weights `w(y) ∝ p_θ(y) 1_H(y) / q(y)` restore
unbiasedness (reweighted wake-sleep [bornschein2015rws]). Unweighted, the step is
`∇ log P(H) − ∇ KL(q ‖ p_H)`, whose contrastive lag term [tieleman2009fastpcd] widens the
twin's gap after each model update; so one either reweights or refits the twin each round.

The loss is bounded — importance-weighted NPO [zhang2024npo] — because the gradient target
`log P(H)` never saturates and every unbounded likelihood-push in the literature collapsed the
model into degenerate text the judge then scored safe [yao2023unlearning] [tamirisa2024tar].
A helpfulness reward on benign prompts and a refusal-rate check anchor utility; stopping is on
held-out harm and utility, never on the training loss.

Against the on-policy baseline — train the model on its own judged samples — the twin's edge
is structural: once the model refuses confidently, most sampled groups contain no breach and
the baseline's gradient vanishes, a stall reported by CHASE and AdvGame
[markasserithodi2026chase] [paulus2025advgame]. The twin estimates the same gradient, and the
*excess* variance from importance reweighting vanishes as `q → p_H` (the weights become
constant on H); the estimator's own variance does not vanish, but the missing-gradient stall
is gone.

## 4. A certificate that does not trust the twin

The twin's value is a *lower* bound on `log P(H)`, the wrong direction for a safety claim,
and each defender step widens it. For an *upper* bound we use the twin only as a proposal.
Fix a checkpoint; freeze `p` and a twin `q`. Define the whole-sequence defensive mixture
[hesterberg1995defensive]

```
m(y|x) = α p(y|x) + (1−α) q(y|x),
```

sampled per response. On a sealed evaluation set `D_eval`, draw `n` fresh prompts and one
response each from `m`. The weight on a response is `w(y) = p(y|x)/m(y|x) ≤ 1/α`, and

```
P(H) = E_{x~D_eval, y~m}[ w(y) 1_H(y) ].
```

This is unbiased for the breach rate under `D_eval` *whatever the twin is* — the `α p`
component covers every mode with weight at most `1/α`. The summand lies in `[0, 1/α]`, so
Hoeffding, empirical-Bernstein (Maurer–Pontil) and betting bounds at level `δ` are all valid
[karampatziakis2021confseq]. The mixture is deliberately over *whole sequences* — one model
generates the entire response — because that is what delivers `w ≤ 1/α`; a mixture that
interpolates the two models at each decoding token bounds the weight only by `α^{−T}`,
exponential in the response length `T`, and supports no finite-width concentration bound
[angell2026tailrisks]. Three consequences:

- **Validity is twin-independent.** A twin that dropped harmful modes raises variance, not
  bias. This is why we mix rather than sample from the twin alone.
- **Width.** When `q ≈ p_H` and `P(H)` is small, `w 1_H` is nearly constant on `H` at value
  `≈ P(H)/(1−α)`, and about a `(1−α)` fraction of draws land in `H`; the estimator's s.d. is
  of order `P(H)/√n` times a factor vanishing as `q → p_H`, against `√(P(H)/n)` for plain
  Monte Carlo — an order of magnitude tighter near `P(H) ≈ 1%` at the same `n`.
- **Coverage, not variance, is the hard limit.** The sample size scales like
  `exp(KL(p_H ‖ q))` [chatterjee2018sample]: a mode-seeking twin can look good on its own
  objective and still need infeasibly many samples. This is the assumption we test first.

Because the bounded quantity is the model's own breach rate on a sealed `D_eval` and the twin
is only the proposal, the non-stationarity of the adversary across training rounds does not
threaten validity — only width.

## 5. What is new, and what is not

The RL identity is standard: `log J` reward with a unit KL to a reference is KL-regularised
RL, and its optimum is the reference tilted toward harm — the same target as twisted
sequential Monte Carlo and rare-event proposals [korbak2022rlkl] [zhao2024twisted]
[wu2024rareoutputs] [angell2026tailrisks]. Likelihood-tempered GFlowNet samplers of a frozen
model confirm that at temperature one the sampler is the model [hu2023amortizing]. We claim
none of this.

What we did not find in prior work, on the full conjunction: a separate same-base policy,
trained by RL to write harmful responses, anchored to the *live* model's likelihood, and
alternated with updates to a generative model under a harm label. The closest alternating
games are against *verifiers* not generators (APRM [juneja2025aprm]), or use prompt-emitting
attackers with judge rewards (TriPlay-RL, AdvGRPO [tan2026triplay] [bullwinkel2026advgrpo]),
or static same-base negatives with no game (Pns, ADPO, MACPO [di2026pns] [kim2024adpo]
[lyu2024macpo]).

The nearest prior work for the estimation half is Angell et al. [angell2026tailrisks], who
also estimate an LLM harm rate by importance sampling from an *unsafe* proxy of the target
mixed back toward it, matching brute-force Monte Carlo with 10–20× fewer samples. Two things
separate the present construction, and both are load-bearing rather than cosmetic. First,
their proxy is built by activation steering and their output is a *point* estimate — the only
interval they report is Clopper–Pearson on the brute-force reference, not on the
importance-sampling estimator — whereas we give a high-confidence *upper* bound, which is what
a safety claim requires. Second, their mixture is applied per decoding token (or as a prefix
switch), so the importance weight is bounded only by `α^{−T}` and supports no finite-width
bound on real-length responses; our whole-sequence mixture bounds it by `1/α`, which is
exactly what makes the empirical-Bernstein and betting bounds valid (§4). The one remaining
new element we test first is the defender step as a variance-reduced, importance-corrected
estimator of the breach gradient against the stalling on-policy baseline. Other rare-event
works likewise report point estimates without confidence sets [wu2024rareoutputs]
[jones2025forecasting]. The search covered web and arXiv 2016–2026 with method sections read
for the closest hits; the claim is "not falsified at this scope", not "first".

## 6. Validation and pre-registered protocol

**E0 — the identity where truth is exact (done).** On a tabular autoregressive toy (two
content tokens plus EOS, length ≤ 4, three prompts, 31 sequences) every quantity is exact by
enumeration. All seven registered checks pass: the twin's value equals `log P(H)` (error
7×10⁻⁴) and `q* = p_H` (KL 3×10⁻⁷); both gradient identities hold to 2×10⁻¹⁶; the unweighted
lag term is visible (mean twin gap 0.69 nat vs 0.26 weighted over 15 rounds); and the
GRPO-style estimator with the implemented reward converges to `p_H` (KL 0.04). The toy also
refined the theory — the KL form is coefficient-robust under an indicator judge while the
entropy form is not — and flagged the one check it cannot show (length bias, its sequences
being too short), which moves to E1.

**E1–E5 — pre-registered on 1.7B–2B models (in progress).** Design, thresholds and the
checkpoint rule are frozen in a dated registration before any GPU run; models are Qwen3-1.7B
and Granite-3.3-2B as LoRA adapters, Qwen3Guard-4B (4-bit) as the training judge and WildGuard
as a distinct evaluation judge, on PKU-SafeRLHF and benign prompts. The registered predictions:

- **E1 (twin vs frozen model).** (a) the twin's value stays below `log P(H)` and rises; (b)
  judged-harmful fraction > 0.8 on resolvable prompts; (c) twin length ≥ 70% of the model's;
  **(d) forward KL(p_H ‖ q) < 1 nat on ≥ half the resolvable prompts.** (d) is the decisive
  one — the certificate of §4 depends on it, and §8 of the proposal treats its refutation as
  grounds to stop.
- **E2 (one defender phase).** Weighted and unweighted NPO both lower `P(H)`; unweighted
  widens the twin's gap more; neither moves XSTest over-refusal by > 2 points at 50 steps.
- **E3 (the alternating loop).** Registered cross-arm predictions on oscillation,
  steps-to-recover, the static-twin and on-policy baselines, over-refusal, and strategy
  coverage. *(On the 12 GB development GPU this is trimmed to the method's own arm for two
  model families at one seed to produce the returned model E4/E5 consume; the cross-arm
  predictions are deferred to larger compute and reported as pending, not scored on partial
  data.)*
- **E4 (transfer).** The loop reduces breach less under a held-out prefill-depth-32 attack
  than under the prompt families.
- **E5 (the bound).** (a) miss rate of both weighted bounds ≤ `δ=0.1` in every plasmode cell
  (validity); (b) betting width ≤ ½ the Clopper-Pearson width at `n=300, α=0.5` when
  `P(H)<3%`; (c) no gain over Clopper-Pearson once `KL(p_H‖q) > 2` nat.

## 7. Limitations

- **Everything is relative to a judge.** `P(H)` is the judge-flagged rate; the twin's optimum
  places mass on judge false positives in proportion to `p`, so judge error decides which
  benign high-probability text the model is pushed off. We mitigate with a separate evaluation
  judge and a fixed per-round human-labelled sample, but the certificate certifies a
  judge-defined quantity, and we say so.
- **The fragile assumption is coverage.** If the trained twin's forward KL to `p_H` stays
  large (E1d), the sample cost `exp(KL)` makes the width advantage vanish; the method then
  reduces to a trainer with no bound. This is tested before any bound is claimed.
- **The breach-gradient identity is per prompt.** Averaged over prompts the training signal
  minimises `E_x log P(H|x) ≤ log E_x P(H|x)` by Jensen; this affects the training signal, not
  the bound, which is on `E_x P(H|x)` directly.
- **No large-model result is reported here.** The empirical claims are registered predictions.
- **Durability is out of scope.** We make no unlearning or tamper-resistance claim; prior
  durability results have broken under re-evaluation [qi2024durability] [che2025tampering].

## 8. Conclusion

A single same-base adversarial adapter, anchored by a unit-coefficient KL to the *live* model
under test, is simultaneously the model's harmful conditional, a variance-reduced estimator of
its breach gradient, and the proposal for a confidence upper bound on its breach rate whose
validity does not depend on the adversary. The identities are standard; their conjunction, the
live anchor, and the twin-independent certificate are not. We have proven the identities where
truth is exact and pre-registered the one experiment — the twin's forward KL to `p_H` on real
models — that decides whether the certificate is feasible or the idea is a trainer only.

## Reproducibility

Code (`seldonian/llm/twin.py`, `scripts/twin_e0.py`…`twin_e5.py`), the dated registration
(`.planning/harm-twin/REGISTRATION-E1-E5.md`) and the E0 report (`results/twin/e0/report.md`)
accompany this paper. Registered numbers do not change after a run starts; amendments are
dated with the reason and whether the result moved. Citation keys resolve against
`reports/literature_adversarial_policy.bib`.

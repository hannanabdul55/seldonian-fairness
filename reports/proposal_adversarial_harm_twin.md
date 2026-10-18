# Proposal: a likelihood-coupled harmful twin, trained in alternation with its main model

Draft 0.2, 2026-10-18. A separate paper from the Seldonian certification work; it shares this repository's tooling and nothing else. Written from the survey `reports/literature_adversarial_policy.md` and the notes in `research_notes/Adversarial policy alternating training/`. Citation keys resolve against `reports/literature_adversarial_policy.bib`. Nothing here has been run. Draft 0.1 left five choices open; the author settled them on 2026-10-18 and they are recorded in section 10.

## 1. The claim in one paragraph

Take an instruct model p and a second copy of the same base, the twin q. Train the twin by reinforcement learning to write responses that a judge calls harmful, with a KL penalty of coefficient 1 to the current main model, summed over tokens. Then push the main model's likelihood down on the twin's samples with a bounded loss, weighted by self-normalised importance weights, while a helpfulness reward and a refusal check hold it in place. Alternate. The twin's optimum is the main model conditioned on harm, p_H, and its objective value is the log breach probability log P(H) [korbak2022rlkl] [khalifa2021gdc] [zhao2024twisted]. So the twin is three things at once: an amortised sampler of the main model's own harmful tail, an unbiased and low-variance estimator of the gradient of log P(H), and a proposal distribution for a high-confidence upper bound on the breach rate whose validity does not depend on how good the twin is. The novelty is narrow and is stated as such: the twin's objective is KL-regularised RL with the live main model as the reference, which is standard [korbak2022rlkl]; what no paper found does is anchor the twin to the live main model, train the main model on the twin's samples, and alternate, for a generative main model and a harm label. The survey's verdict is not falsified at that scope, partially anticipated, and the search scope is in section 4.

## 2. Setting and notation

- Prompts x drawn from a distribution D. Here D is the PKU-SafeRLHF adversarial prompt set plus benign prompts, split into a training set D_train and a sealed evaluation set D_eval that no training step ever sees.
- The main model p_theta and the twin q_phi are two LoRA adapters on one frozen base B. Only adapters train. One base sits in GPU memory with two adapter sets, so the twin's reward and the importance weights cost one extra forward pass each.
- A judge J(x, y) in {0, 1}. H(x) = {y : J(x, y) = 1}. P_theta(H | x) is the probability that the main model writes a judged-harmful response at temperature 1, no truncation. Everything measured is about the judge's label, not true harm; section 5 says how judge error is handled.
- All sequence quantities are summed over tokens: log p(y | x) = sum_t log p(y_t | x, y_<t), likewise the KL and the entropy. A fixed maximum length L, and H is defined on terminated sequences.
- The breach rate is P_theta(H) = E_{x ~ D}[P_theta(H | x)].

## 3. The objective, with its exact assumptions

### 3.1 Twin step

For a fixed main model p and prompt x, the twin maximises

    F(q) = E_{y ~ q}[ log J(x, y) ] - KL( q(. | x) || p(. | x) ).

By the Gibbs variational identity, F(q) = log P_p(H | x) - KL(q || p_H), where p_H(y | x) = p(y | x) 1_H(y) / P_p(H | x). The maximiser is q* = p_H and the maximum is log P_p(H | x). Any other twin gives a lower bound whose gap is exactly the reverse KL to p_H [korbak2022rlkl] [khalifa2021gdc]. Expanding the KL gives the equivalent form E_q[log p + log J] + H(q): reward the main model's likelihood plus the harm term, with a full-sequence entropy bonus at coefficient 1. The KL form is what gets implemented (section 10, decision 4); the entropy form is how the sketch was first worded.

Assumptions that the identity needs, each of which is a known way to break it (derivation check in `coupling_2_mechanics.md`, section 7):

1. Summed sequence log-probs and summed KL, fixed maximum length. Length normalisation has no proper optimum short of the length cap. Per-token averaging gives an effective coefficient that varies with length.
2. KL coefficient 1, to the current main model, and no additional KL to a fixed reference. A KL to the base p_0 instead gives a twin that tracks the base, not the live main model; a KL to both double-counts the likelihood. E0 refined this point: in the KL form with an indicator judge, a coefficient beta other than 1 leaves the optimum at p_H and only rescales the value to beta log P(H), because q* is proportional to p J^(1/beta) and J^(1/beta) = J. The entropy form is not robust in the same way: its optimum under coefficient beta_e is proportional to (p J)^(1/beta_e), a tempered main model. So the coefficient matters for the value and for a soft judge, not for the optimum under an indicator, and the KL form is the safer implementation.
3. The harm term enters as log J. An additive bonus lambda J gives a cumulant generating function whose optimal twin is mostly harmless unless lambda exceeds about log(1 / P(H)). A smoothed log(J + epsilon) gives a mixture that is mostly p unless epsilon is far below P(H).
4. Realisability and optimisation: the twin can represent and reach p_H for every x. Otherwise there is an amortisation gap plus an RL optimisation gap.
5. The identity is per prompt. Averaged over prompts the game minimises E_x log P(H | x), which by Jensen sits below log E_x P(H | x), the breach rate. Section 6 says why this does not affect the bound.
6. Deterministic token dynamics, true for autoregressive generation, so sequence-level maximum-entropy RL is exact inference.
7. The judge defines H. The twin's optimum puts mass on judge false positives in proportion to p, so judge errors decide which benign high-probability text the main model is pushed off.
8. Likelihood alone is not enough: with J removed, q* = p and the value is 0 [hu2023amortizing]. The coupling is not judge-free.

### 3.2 Main step

At the twin's optimum, the gradient identity nabla_theta log P_theta(H) = E_{y ~ p_H}[nabla_theta log p_theta(y | x)] makes "lower log p on samples from the twin" unbiased gradient descent on log P(H), with no baseline needed because E_p[nabla log p] = 0. Away from the optimum, two corrections:

- Self-normalised importance weights w(y) proportional to p_theta(y | x) 1_H(y) / q(y | x) remove the bias (the reweighted wake-sleep estimator) [bornschein2015rws]. Both sequence log-probs are exactly computable, so the weights are exact.
- Without weights, the main step equals nabla log P(H) - nabla KL(q || p_H): it also pushes the harmful conditional away from wherever the twin is, the persistent contrastive divergence lag term [tieleman2009fastpcd]. That widens the twin's gap after each main step, so an unweighted main model minimises a lower bound on its own breach log-probability, and a lower bound is loosest exactly where an optimiser finds it [geng2021bounds]. Weighted samples, or a twin re-fitted before each main step, avoid this.

The defender is standalone (section 10, decision 2). Its loss on a batch of twin samples is the importance-weighted negative preference optimisation (NPO) loss, which is bounded and saturates as p_theta(y) falls [zhang2024npo], added to the usual GRPO helpfulness objective on benign and task prompts, with a refusal-rate check on benign prompts as a stopping condition. The reason for a bounded loss: the gradient identity is for log P(H), which never saturates, and every published attempt to push likelihood down without a bound collapsed the model into text that the judge then scored as safe [zhang2024npo] [yao2023unlearning] [tamirisa2024tar]. The stopping rule is on held-out harm and utility, never on the training loss. A hinge variant, zero loss once the measured breach rate on a held-out slice is under a target, runs as a one-seed ablation.

Why not on-policy alone. The natural baseline trains the defender on its own samples with a judge penalty. Once the model refuses confidently, P(H) is small, most GRPO groups have no harmful sample and so no gradient, and the loop stalls; CHASE and AdvGame both report this [markasserithodi2026chase] [paulus2025advgame]. The twin's weighted samples estimate the same gradient with variance that goes to zero as q approaches p_H. That baseline is arm (4) of E3.

### 3.3 Why this is the same game as the sketch

The sketch was: a harm policy initialised from the same base, trained by RL to make the main model breach, the two tuned by alternating steps. Under coupling 2, "make the main model breach" is operationalised as "write what the main model itself would write when it breaches", which is p_H. Section 3.1 shows that the KL-anchored harm reward is the unique objective whose optimum is p_H. Couplings 1, 3 and 4 from the survey remain alternatives and are not pursued here.

## 4. What is new, what is not, and the search scope

Stated first, because a reviewer will make the equivalence: reward log J with KL to a reference at coefficient 1 is the standard KL-regularised RL objective, and its optimum is the reference tilted toward harm, the same target that twisted sequential Monte Carlo and rare-event proposals learn [korbak2022rlkl] [zhao2024twisted] [wu2024rareoutputs] [angell2026tailrisks]. The likelihood reward is a selection rule in Lyu et al., who score harmful continuation states by the current model's log-prob margin and train refusal from the worst, by search, without a generator [lyu2026autoregressive]. Likelihood-tempered GFlowNet samplers of a frozen model exist and confirm that at temperature 1 the sampler is the model itself [hu2023amortizing]. Yang et al. ran a language-model-likelihood minimax game in 2018 and reported it as very unstable [yang2018lmdiscriminator].

What no paper found does, on the full conjunction: a separate same-base policy, trained by RL, writing harmful responses, anchored to the live main model's likelihood, alternated with updates to a generative main model. The closest alternating games are against verifiers rather than generators (APRM, same backbone, PPO, alternates every 5 steps) [juneja2025aprm] [bukharin2025advrm], or use prompt-emitting attackers with judge rewards (TriPlay-RL, MAGIC, AdvGRPO) [tan2026triplay] [wen2026magic] [bullwinkel2026advgrpo], or static same-base negatives with no game (Pns, ADPO, MACPO) [di2026pns] [kim2024adpo] [lyu2024macpo].

Two further things appear to be new and are the parts worth testing first:

- The defender step as a variance-reduced, importance-corrected estimator of the gradient of the breach probability, against the on-policy judge-penalty baseline that stalls.
- The twin as the proposal for a high-confidence upper bound on the breach rate (section 6). The survey searched for a bound on an LM safety label under a learned adversary's distribution and found none; the rare-event papers give point estimates with no confidence bounds [wu2024rareoutputs] [jones2025forecasting] [angell2026tailrisks].

Search scope: web search and arXiv pages, 2016 to October 2026, with method sections read for the closest hits. Not covered: OpenReview-only and workshop papers, non-English venues, patents, and the forward citations of Pns and APRM (rate-limited). Four papers were read at abstract level only. Any of these could hold a counterexample; the claim is "not falsified at this scope", not "first".

## 5. The training loop

One round r:

1. Twin phase. Starting from the twin's previous adapter (persisted, not reset), run K_q GRPO steps on prompts from D_train with reward log J(x, y), a summed KL penalty of coefficient 1 to the current main adapter, and no KL to the base. Stop when the twin's per-prompt objective stops rising, or at K_q.
2. Snapshot the twin into the pool Q_r = {q_1, ..., q_r}; keep also the frozen base twin q_0 (the twin trained once against the untrained main model) as a fixed yardstick.
3. Main phase. Draw fresh samples from a recency-weighted mixture over the pool (half-life 3 rounds), compute self-normalised weights p_theta 1_H / q, and run K_p defender steps with the weighted NPO loss plus the helpfulness objective, K_p smaller than K_q.
4. Log, every round: judged breach rate on a held-out slice of D_train for the main model, the twin's objective value (a lower bound on log P(H)), the twin's judged-harmful fraction, response length and distinct-n-gram statistics for both models, over-refusal on XSTest's 250 safe prompts [rottger2024xstest], helpfulness reward on benign prompts, and steps-to-recover (step 5).
5. Steps-to-recover. At the start of each twin phase, record the number of twin steps until the twin's harmful fraction returns to its end-of-previous-round value. This is the quantity the survey found no paper measures.

Design choices and the failure each answers (survey, "Design lessons for the likelihood-coupled twin"):

| Choice | Failure it answers | Evidence |
|---|---|---|
| Summed log-probs and KL, coefficient 1, fixed length cap, no KL to the base | Loss of the identity; tracking the base instead of the live model | derivation check, [korbak2022rlkl] [hu2023amortizing] |
| Indicator or thresholded judge, log form, no smoothing above P(H) | Optimal twin mostly harmless under soft or smoothed harm | derivation check |
| Trivial, refusing and truncated outputs penalised; training judge Qwen3Guard-4B in 4-bit; evaluation judge a different family (WildGuard); human labels on a fixed sample of 100 twin outputs per round, labelled blind | Judge gaming by both sides; prefixes that self-correct and judges that over-credit them [zhu2024advprefix] | [bukharin2025advrm] [ge2023mart] |
| Bounded NPO loss with helpfulness and refusal anchors; stop on held-out harm and utility | Likelihood-lowering collapse that scores as safe | [zhang2024npo] [yao2023unlearning] [tamirisa2024tar] |
| Fresh twin samples each round, self-normalised weights | Lag term, stale negatives, strongest squeezing on unlikely samples | [tieleman2009fastpcd] [bornschein2015rws] [zhang2026selfreset] |
| Twin re-fitted before each main step; K_q larger than K_p | Static adversary 3 to 6 times worse; wake-sleep small-step condition | [sanyal2025antidote] [hu2023gflownetem] |
| Twin warm-started on seed harmful positives in its own adapter | Cold start; safety leaking into a same-base attacker (Self-RedTeam refused a quarter of its attack tasks) | [zhao2024twisted] [liu2025selfredteam] |
| Pool of past twins plus frozen base twin | Training only against the latest opponent ends worse than no training; cyclic forgetting | [wang2025arlas] [wen2026magic] |
| Twin negatives mixed with a fixed pool of standard harmful data, including deep prefixes | Attack-class trade-off; prefill regression | [simko2025contrastive] [lyu2026autoregressive] |
| Breach rate and over-refusal reported as a pair every round, checkpoint rule fixed in advance | Over-refusal doubling in RL loops without a utility term | [wang2025apl] [bullwinkel2026advgrpo] |
| Many twin seeds, LoRA and full fine-tuning both tested before any durability claim; never called unlearning | Durability results that later re-evaluation broke | [qi2024durability] [che2025tampering] |

Open disagreement, to be settled by E1: the generator-games note says to length-normalise the likelihood reward to stop short outputs; the derivation says that breaks the identity. The proposal keeps summed quantities and controls short-output drift with the trivial-output penalty. If E1 shows drift anyway, the fallback is a minimum-length constraint on the twin, not normalisation.

## 6. A high-confidence upper bound on the breach rate, with the twin as proposal

This is a secondary result (section 10, decision 1). It stands on its own: nothing in it depends on the Seldonian certification paper, which it cites once as prior work on safety tests from held-out samples [thomas2019seldonian].

### 6.1 What the twin's value is not

The twin's objective value is an evidence lower bound on log P(H), and each main step widens the gap. A lower bound on the breach probability is the wrong direction for a safety claim, which needs a high-confidence upper bound. Markov's inequality on an unbiased estimate also gives only a lower confidence bound, the SMC upper bound needs exact samples from p_H and holds only in expectation [zhao2024twisted], and the truncation argument of high-confidence off-policy evaluation works only for lower bounds [thomas2015hcope]. So the twin's value is a training diagnostic and nothing more.

### 6.2 The construction

Fix the round. Freeze the main model p and a pool twin q (or the pool mixture). Define the whole-sequence defensive mixture

    m(y | x) = alpha p(y | x) + (1 - alpha) q(y | x),

sampled by choosing p with probability alpha and q otherwise, per response [hesterberg1995defensive]. On D_eval, draw n fresh prompts and one response each from m. The importance weight on a harmful response is w(y) = p(y | x) / m(y | x) <= 1 / alpha, and

    P_p(H) = E_{x ~ D_eval, y ~ m}[ w(y) 1_H(y) ].

The estimator is unbiased for the breach rate under D_eval whatever the twin is. The twin affects only the variance. Because the summand lies in [0, 1 / alpha], Hoeffding, empirical Bernstein and betting bounds at level delta are all valid, and the betting and empirical-Bernstein bounds exploit the low variance [karampatziakis2021confseq].

Three properties follow from the algebra and are the registered predictions of E5:

- Validity does not depend on the twin. A twin that dropped harmful modes raises the variance but not the bias, because the alpha p component still covers every mode with weight at most 1 / alpha. This is the reason to use a mixture rather than the twin alone.
- Width. When q is close to p_H and P(H) is small, w 1_H is nearly constant on H with value about P(H) / (1 - alpha), and roughly a (1 - alpha) fraction of samples land in H. The estimate's standard deviation is then of order P(H) / sqrt(n) times a factor that goes to zero as q approaches p_H, against sqrt(P(H) / n) for plain Monte Carlo on p. For P(H) near 1 percent that is an order of magnitude tighter at the same n. The gain disappears as the twin degrades, and the empirical bound reports whatever variance it finds.
- Coverage, not variance, is the hard limit. The sample size needed grows like exp(KL(p_H || q)), the forward KL [chatterjee2018sample]. A mode-seeking twin can look good on its own objective and still need far too many samples. E1 measures this directly.

The same construction gives a bound under an adversarial prompt distribution by changing D_eval; the proposal declares that as a second, separately stated bound and does not mix the two.

### 6.3 Which distribution the bound is about

The survey's obstacle was that the adversary's distribution is non-stationary across rounds, so any bound would need to say which round's distribution it covers. Under this construction the bounded quantity is the main model's own breach rate on D_eval, and the twin is only the proposal. Non-stationarity is therefore not a problem for validity. It is a problem for width: a stale twin is a poor proposal. The bound is computed once, on fresh samples that never enter training, at the checkpoint the pre-registered rule selects.

The Jensen gap (assumption 5) is likewise irrelevant here, since the bound is on E_x P(H | x) directly. It matters only for the training signal, and E3 records whether the game spends capacity on prompts whose breach probability is already negligible.

## 7. Experiments

Models and tools already in the repository: Qwen2.5-0.5B-Instruct, Granite-3.3-2B-Instruct and Qwen3-1.7B; Qwen3Guard-4B in 4-bit as the training judge; WildGuard as the evaluation judge; PKU-SafeRLHF adversarial and benign prompts; XSTest; the synthetic contextual bandit in `seldonian/llm/synthetic.py`, where the exact violation probability of any policy is computable; the LoRA GRPO backend in `seldonian/llm/backend.py`; the Clopper-Pearson and betting bounds in `seldonian/llm/policy.py`. The local GPU is a 12 GB card, so everything runs at 0.5B to 2B with LoRA, with the judge in 4-bit, one model family at a time.

Each experiment lists its registered prediction. The statistics design, the checkpoint rule and every threshold are committed before the first GPU run, with a dated registration file. A smoke test may print timings and key names only. A run that crashes is reported as lost, not retried without a decision.

### E0. The identity on a model where the truth is exact (CPU, one day)

Use a tabular autoregressive toy (two content tokens plus end-of-sequence, maximum length 4, three prompts, 31 sequences) so P(H | x), p_H and every gradient are exact by enumeration. Fit the twin by exact gradient ascent and by a sampled GRPO-style estimator. Check: the twin's value equals log P(H) at the optimum; the unweighted main step equals nabla log P(H) minus the KL lag term; the weighted step is unbiased; KL coefficient 0.5 and 2, an extra KL to a fixed base, an additive harm bonus, a length-normalised likelihood and a per-token-averaged KL each move the optimum as section 3.1 predicts; and the GRPO-style estimator with the design's reward converges to p_H. Registered prediction: all identities hold to numerical precision; the lag term is visible as a systematic widening of the twin's gap after unweighted main steps.

Run 2026-10-18 (`scripts/twin_e0.py`, report in `results/twin/e0/report.md`): 7 of 7 checks pass. Value error 7e-4 and KL to p_H 3e-7 at the twin's optimum; both gradient identities hold to 2e-16; the lag term is 0.65 against a gradient scale of 0.97; over 15 alternation rounds the unweighted main step leaves a mean twin gap of 0.69 nats against 0.26 for the weighted step, with the same fall in P(H); the GRPO-style estimator with a 64-sample warm start reaches KL 0.04 to p_H. One refinement came out of it, now in assumption 2: the KL form with an indicator judge is robust to the coefficient, the entropy form is not. One limit: the toy's sequences are 1 to 4 tokens long with p_H near the cap, so it cannot show the length bias of token-mean aggregation; that check moves to E1, which logs both aggregations' per-token weights on real lengths. The TRL check against the exact objective is part of building the trainer, not of the toy.

### E1. The twin against a frozen main model (GPU, about 8 hours)

Qwen2.5-0.5B-Instruct as p, frozen. Brute-force reference: sample 20,000 responses from p on 200 held-out prompts and judge them, giving P(H | x) to Clopper-Pearson precision for the prompts where it is above about 1 percent. Train the twin for 300 steps. Measure: the twin's objective value against log P(H) per prompt; its judged-harmful fraction; forward KL(p_H || q) estimated from the brute-force harmful samples; strategy-level coverage of the brute-force harmful set (distinct refusal-bypass categories, hand-coded on 100 samples); response length and repetition.

Registered predictions: (a) the twin's value is below log P(H) for every prompt and rises monotonically on average; (b) the harmful fraction exceeds 0.8 within 300 steps on prompts with P(H) above 1 percent; (c) the twin drifts shorter than p by at most 30 percent of mean length under the trivial-output penalty (if refuted, the fallback in section 5 applies); (d) forward KL is below 1 nat on at least half the prompts. Prediction (d) is the one most likely to fail, and it is the one section 6 depends on.

### E2. One main step (GPU, about 4 hours)

From E1's twin, take 50 and 200 weighted versus unweighted defender steps with the NPO loss, and the hinge variant at one setting. Measure P(H) by brute force before and after, the twin's gap, and the helpfulness and refusal anchors. Registered predictions: both weighted and unweighted versions lower P(H); the unweighted version widens the twin's gap by more; neither moves refusal on XSTest by more than 2 points at 50 steps.

### E3. The alternating loop (GPU, about 30 hours, the main experiment)

Qwen2.5-0.5B and Granite-3.3-2B, three twin seeds each, 10 rounds. Arms: (1) the full loop with the pool; (2) latest-twin only; (3) static twin, trained once and frozen; (4) on-policy baseline, the defender trained on its own samples with the judge penalty and no twin, matched in defender steps; (5) fresh twin re-initialised from the base each round, as the control for "does persistence matter". Every arm logs the section 5 quantities every round.

Registered predictions, each scored kept or refuted:

- P1. Per-round breach-rate curves for arm (2) show at least one rise after a fall within 10 rounds (oscillation). Arm (1) shows none larger than the round-to-round noise band from the three seeds.
- P2. Steps-to-recover for the persisted twin is below 40 percent of a fresh twin's steps to the same harmful fraction in every round after the first.
- P3. Arm (3)'s main model breach rate under a fresh twin at round 10 is at least twice arm (1)'s.
- P4. Arm (1) reaches the same held-out breach rate as arm (4) in fewer defender steps, with the fraction of arm (4)'s GRPO groups containing no harmful sample above 50 percent by round 5.
- P5. Over-refusal on XSTest rises by less than 5 points in arm (1) and by more in an ablation of arm (1) that drops the helpfulness objective (one seed).
- P6. The twin's strategy-level coverage falls across rounds in arm (2) and holds in arm (1).

The checkpoint rule, fixed now: the returned main model is the last round whose held-out breach rate is below the target and whose XSTest refusal rate is within 3 points of the untrained model, with ties broken by helpfulness reward.

### E4. Transfer to held-out attack families (GPU, about 4 hours)

On the returned models from E3, measure breach rate under families the loop never saw: harmful-prefix prefill at depths 0, 8 and 32 tokens [qi2024fewtokens], PAIR-style and template jailbreak prompts from the PyRIT integration (`seldonian/llm/redteam.py`), and the multi-turn family if the harness supports it. Registered prediction: the loop improves the prefill family less than the prompt families, matching the shallow-alignment result, and the fixed harmful-data mix in section 5 narrows that gap. No transfer claim is made beyond what is measured.

### E5. The upper bound (GPU, about 6 hours, plus CPU plasmodes)

On E3's returned models, draw n = 300 and 1,000 fresh responses from the defensive mixture at alpha in {0.2, 0.5}, with twins from the pool at rounds 1, 5 and 10, and compute the empirical-Bernstein and betting upper bounds at delta = 0.1 beside Clopper-Pearson on n plain samples from p. Truth is the brute-force P(H) from 20,000 plain samples. Plasmode check first: resample from the brute-force pools to measure the miss rate of each bound against delta over 1,000 draws before any claim is read off the real draws.

Registered predictions: (a) miss rate of the weighted bounds is at or below delta in every cell, including the round-1 twin, which is the validity claim; (b) width at n = 300 with the round-10 twin and alpha = 0.5 is at most half the Clopper-Pearson width when P(H) is below 3 percent; (c) width with a twin whose forward KL exceeds 2 nats is no better than Clopper-Pearson. If (a) fails in any cell the construction is wrong and the rest of section 6 is void.

### Budget and order

E0 and the plasmode half of E5 first, on CPU. Then E1, E2, E3, E4, E5 in that order, about 52 GPU hours in all, with a hard cap written as a `timeout` in each run script and checkpoints saved first. E3 is the only experiment whose result changes the direction of the work; E1 prediction (d) and E5 prediction (a) are the two that can kill the bound early.

## 8. What would make this not worth continuing

- E1 (d) refuted on most prompts: the twin does not cover the harmful tail, the bound gains nothing, and the defender step's variance reduction is confined to the modes the twin found. The training half could still be useful, but the paper would be about a trainer, not a bound.
- E3 P4 refuted: the on-policy baseline already gets there at the same cost, and the twin is machinery with no payoff.
- E3 P1 and P2 both refuted: there is no game to speak of, and the result is a static same-base negatives method, which Pns, ADPO and MACPO already are.
- E5 (a) refuted: an error in the mixture sampling or the weights; fix and rerun before anything else.

## 9. Relation to the rest of this repository

This is a separate paper. It borrows the repository's judges, prompt loaders, LoRA GRPO backend, synthetic bandit and bound implementations as tools, and it makes no claim that depends on the certification paper's results. Two of that paper's negatives still apply as engineering facts and are built into section 5: a judge's calibration does not carry across training that targets the label, so the per-round human-labelled sample of twin outputs is not optional; and the on-policy judge penalty goes quiet once the model refuses confidently, which is the baseline E3 arm (4) exists to show.

The world-model game idea in `reports/ideas.md` (2026-09-29) is a different thing: harms known from a simulated world's state rather than a judge. It is not part of this proposal.

## 10. Decisions

Settled by the author on 2026-10-18:

1. The upper bound of section 6 stays as a secondary result, stated on its own terms, with no dependence on the Seldonian paper.
2. The defender is standalone: importance-weighted bounded loss on twin samples plus the helpfulness objective and a refusal check. The Seldonian Lagrangian trainer is not in the loop.
3. The bounded loss is the NPO form; a hinge at a target breach rate runs as a one-seed ablation in E2 and E3.
4. The twin's objective is implemented in the KL form: reward log J, KL to the live main adapter at coefficient 1, summed over tokens. Implementation settled after the 2026-10-18 preflight: the KL enters through the reward as the exact sequence-level log-ratio, reward = log(J + epsilon) + log p_main(y | x) - log q_old(y | x), with both sequence log-probs computed by one extra forward pass each (main adapter, and the twin adapter at sampling time), TRL `beta = 0`, `scale_rewards = "none"` (the group mean is the baseline, no division by the group standard deviation, which would rescale the coefficient), and a token-summed loss aggregation (TRL 1.12 defaults to `loss_type = "dapo"` and `scale_rewards = "group"`, neither of which keeps the identity). This avoids TRL's per-token k3 KL term, whose gradient is an approximation. E0's C7 verified this estimator on the exact toy.
5. This is its own paper, not a section of the certification paper.

Defaults set by the drafter, not yet discussed: recency-weighted pool with half-life 3 rounds, no ablation of the weighting in E3; full fine-tuning of the twin only after E3 reads out and only before any durability claim; target venue decided after E3.

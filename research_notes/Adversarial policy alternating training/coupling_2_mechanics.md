# Coupling 2 mechanics: the likelihood-rewarded harm twin, its variational identity, and its failure modes

Date: 2026-10-17. Literature check for the research report on adversarial-policy alternating training.

**Design under study ("Coupling 2").**
- A harm twin q, initialised from the same base as the main LM p, writes responses y to prompts x.
- q is trained by RL. Its reward is log p(y|x) (the main model's log-likelihood of the twin's response) plus a harm term, with an entropy or KL regulariser.
- p is then updated to lower log p(y|x) on the twin's harmful outputs.
- The two updates alternate.

**Method.**
- Ran WebSearch (standard and extended) and WebFetch on arXiv abs and HTML pages.
- Read method sections in full text where it mattered: PDFs downloaded and text-extracted with pypdfium2 for Korbak 2022, Tieleman and Hinton 2009, VERA, Geng 2021, RWS, NPO, unlikelihood, Karampatziakis 2021, GDC, ARCA, Lew 2023 and Levine 2018.
- Every arXiv ID below was checked against the arXiv API (title, first author, date) on 2026-10-17. Venues come from the arXiv comment field or the proceedings page where one was found. Otherwise they are marked "(venue from memory, not verified)".

**Notation.**
- Fix a prompt x. Y is the set of finite token sequences, made finite by any maximum length.
- H ⊆ Y is the harmful set, and J(y) ∈ [0,1] is a judge probability. 1_H is the indicator.
- P(H) = P_p(H|x) = Σ_{y∈H} p(y|x).
- p_H(y) = p(y|x) 1_H(y) / P(H) is the main model conditioned on harm.
- H(q) = −E_q[log q] is the sequence-level (summed, not per-token) entropy.

---

## 1. The derivation, re-derived with its exact assumptions

### 1.0 The identity used everywhere

**Gibbs variational principle (Donsker–Varadhan duality for log-sum-exp).** For any f: Y → [−∞, ∞) with Z_f = Σ_y exp f(y) < ∞, and any distribution q on Y:

    E_q[f] + H(q) = log Z_f − KL(q || q*),     q*(y) = exp f(y) / Z_f.      (G)

So max_q {E_q[f] + H(q)} = log Z_f, the maximiser q* is unique, and every other q gives a lower bound whose gap is exactly KL(q||q*).

Where it appears:
- Korbak et al. 2022, appendix, eqs 11–17: the ELBO on log p(O=1).
- Levine 2018, maximum-entropy RL as inference.
- Grathwohl et al. 2021, VERA eq. 3, stated verbatim as "log Z(θ) = max_q E_q(x)[f_θ(x)] + H(q)".
- Zhao et al. 2024: the gap identity KL(q||σ) = E_q[log q/(p0 φ)] + log Z_σ.
- The DPO closed form, Rafailov et al. 2023 (arXiv 2305.18290, VERIFIED).

**Assumptions of (G):**
- The sum is over whole sequences.
- The entropy is the full-sequence entropy with coefficient 1.
- Z_f is finite.
- KL(q||q*) is finite only if supp(q) ⊆ supp(q*).

### 1.1 Point 1: reward log p alone, entropy coefficient 1

Take f = log p(·|x). Then Z_f = 1 and q* = p, with value 0. Equivalently, E_q[log p] + H(q) = −KL(q||p) ≤ 0.

**CONFIRMED.** At the optimum the twin is the main model, and it is harmful only at p's own base rate, so a harm term is required.
- Hu et al. 2024 state the same fact for their likelihood-tempered reward p_LM(Z|X)^{1/T}: at T = 1 "the sampler will trivially sample proportional to pLM". They therefore use T < 1 (T = 0.875) to concentrate on likely continuations.

### 1.2 Point 2: reward log p + log J, entropy coefficient 1

Take f = log p + log J. Then q* = pJ / E_p[J], and the optimal value is log E_{y~p}[J(y)].

**Indicator J = 1_H:**
- q* = p_H, the main model conditioned on harm.
- The value is log P_p(H|x).
- For any q with supp(q) ⊆ H: objective(q) = log P(H) − KL(q || p_H) ≤ log P(H).
- If q puts any mass outside H, then log J = −∞ there and the objective is −∞.

**CONFIRMED, with three precisions:**

(a) **Same objective as KL-regularised RL with the current main model as reference.** E_q[log p + log J] + H(q) = E_q[log J] − KL(q||p). This is exactly Korbak et al.'s eq. 16 ("KL-regularised RL objective with β = 1") with prior π0 := p (the current main model) and evidence r = log J. Their worked example is the mirror image: p(x|O=1) with O = "non-offensive".
- The I-projection reading is in Khalifa et al. 2021 (GDC). For pointwise binary constraints b(x), the minimum-KL-from-a distribution satisfying E[b] = 1 is P(x) ∝ a(x)b(x). So p_H is the closest distribution to p, in KL(·||p), that is supported on H.

(b) **The ELBO statement holds for any q.** The bound is tight only at q = p_H. The gap KL(q||p_H) is reverse KL (mode-seeking), so a twin covering only some harmful modes still gives a valid but loose lower bound.

(c) **A soft judge gives the randomized-judge rate.** With J a judge probability, the value is log E_p[J], the log breach probability under a randomized judge that labels y harmful with probability J(y). It is not a threshold rate.

### 1.3 Point 3: the game at the twin's optimum, and what changes under other regularisers

**At the twin's optimum, the game reduces to "the main model minimises its own log breach probability". CONFIRMED.** Under realizability (q can represent p_H for every x):

    min_θ max_q { E_q[log p_θ + log J] + H(q) } = min_θ log P_θ(H|x).

The gradient identity makes the main step exact:

    ∇_θ log P_θ(H) = Σ_{y∈H} p_θ(y) ∇ log p_θ(y) / P_θ(H) = E_{y~p_H}[∇_θ log p_θ(y)].

Since E_p[∇ log p] = 0, no baseline term is needed. "Lower log p(y|x) on samples y ~ p_H" is therefore unbiased gradient descent on log P(H). This is the score-function identity. In wake-sleep terms it is the RWS wake-phase p-update with exact posterior samples (Bornschein and Bengio 2015, eq. 3 and following).

**Exact debiasing for a non-optimal twin.** When q ≠ p_H, ∇ log P(H) = E_q[w̃ ∇ log p] with self-normalised weights w̃ ∝ p(y) 1_H(y) / q(y). This is the RWS estimator. For LMs the weights are exactly computable, because both sequence log-probs are available.

**New, from the same algebra: the lag term.** For supp(q) ⊆ H:

    E_q[∇_θ log p_θ] = ∇_θ log P_θ(H) − ∇_θ KL(q || p_{θ,H}).

So the main step θ ← θ − η E_q[∇ log p_θ] does two things:
- it descends log P(H), and
- it ascends KL(q || p_H). It pushes p's harmful conditional away from wherever the twin currently is.

This is the exact analogue of Tieleman and Hinton's (2009) PCD result. If the persistent chain samples from R ≠ Q_θ, "there is an additional term in the expected weight updates. This additional term is the gradient of KL(R||Q_θ)", and "it moves the model's distribution away from R".

Two consequences:
- **Exploration benefit.** As in PCD, it forces the twin to move to new harmful modes.
- **Optimistic twin estimate.** After each main step, the twin's ELBO underestimates log P(H) by a larger margin. The main model is literally minimising a lower bound on the quantity it wants to minimise. Geng et al. 2021 warn of exactly this: "By minimizing a lower bound, we run the risk of finding an optimum where the bound is loosest".

**"q acts as an amortised sampler / IS proposal for the harmful tail": CONFIRMED in the literature.**
- Hu et al. 2024: amortised posterior samplers for LMs.
- Zhao et al. 2024: an SMC proposal plus log Z estimation for a toxic target σ ∝ p0 1[s∈C].
- Wu and Hilton 2024: importance sampling "essentially recovers standard adversarial training", and minimising the estimated probability "generalizes adversarial training".
- Angell et al. 2026: unsafe proposal models give IS estimates of harm probability.
- At q = p_H the IS estimator w(y) 1_H(y) = P(H) is constant, so it has zero variance. Zhao et al. Prop. A.2 states the zero-variance property for optimal twisted SMC.

**Corrections and generalisations under other regularisers.** These are derived from (G) by absorbing the regulariser into f.

Let the twin objective be E_q[a log p_t + λ log J] + β_e H(q) − β_r KL(q || π0), where p_t is the current main model, π0 a fixed reference (e.g., the base p_0), β_e the entropy coefficient and β_r the KL coefficient. Write β = β_e + β_r. Then:

    q* ∝ π0^{β_r/β} · p_t^{a/β} · J^{λ/β},
    value = β log Σ_y π0^{β_r/β} p_t^{a/β} J^{λ/β}.

- **Entropy coefficient β_e ≠ 1, no KL (a = λ = 1).**
  - q* ∝ (p J)^{1/β_e}. For an indicator, the value is V_β = β log Σ_{y∈H} p(y)^{1/β} = log ‖p 1_H‖_{1/β}, an ℓ_{1/β} quasi-norm of p on H.
  - For β < 1: V_β ≤ log P(H), with equality only if H's mass sits on one string. As β → 0, V_β → log max_{y∈H} p(y), the single most likely harmful string (MAP-harm, with the mode pathologies of §4).
  - For β > 1: V_β ≥ log P(H), and q* spreads toward uniform over H, i.e., harmful text that p considers improbable.
  - So the twin's value equals the log breach probability only at β = 1.
  - Tempering only the harm term (λ ≠ 1, β = 1) gives q* ∝ p J^λ and value log E_p[J^λ]. For an indicator this is still log P(H).
- **KL to a fixed reference instead of entropy (β_e = 0, β_r = β, a = λ = 1).** This is the standard PPO-RLHF setting, with no entropy bonus and a KL penalty to the initial policy.
  - q* ∝ p_0 · (p_t J)^{1/β}.
  - At β = 1 and initialisation (p_t = p_0), q* ∝ p_0² J. That double-counts p: it is sharpened toward the most likely harmful strings.
  - The value is log Σ_{y∈H} p_0(y) p_t(y), not log P(H).
  - **Correction to the brief:** "reward log p plus KL-to-reference" does not give p conditioned on harm.
  - The two clean ways to obtain q* = p_t J / Z with value log P_t(H) are: (i) reward log p_t + log J with full-sequence entropy coefficient 1, or (ii) reward log J alone with KL(q || p_t) to the current (moving) main model, coefficient 1. These are the same objective.
  - With reward log J and KL to the fixed base p_0, q* = p_0 conditioned on harm. The twin then measures the base model, not the current main model, and does not track training at all.
- **Additive harm bonus λJ instead of log J (reward log p + λJ, β_e = 1).**
  - q* ∝ p e^{λJ}, and the value log E_p[e^{λJ}] is a cumulant generating function.
  - For an indicator the value is log(1 + (e^λ − 1) P(H)), monotone in P(H), so the main model is still pushed the right way.
  - q*'s harmful fraction is e^λ P / (1 + (e^λ − 1) P), which exceeds 1/2 only when λ > log((1−P)/P) ≈ log(1/P(H)). For P(H) = 1e-4 that requires λ > 9.2. With a smaller λ the optimal twin is mostly harmless.
- **Smoothed indicator log(J + ε) (needed in practice because log 0 = −∞).**
  - q* = (P(H) p_H + ε p) / (P(H) + ε). If ε ≳ P(H), the optimal twin is mostly p itself.
  - For rare harm, ε must be far below the breach rate being estimated.
- **Length-normalised likelihood reward (1/|y|) log p(y).**
  - The identity breaks. exp(mean token log-prob) does not decay with length, while the entropy term rewards the |V|^L growth in the number of strings. Σ_y exp(log p(y)/|y|) grows exponentially in the maximum length.
  - The optimum is pinned to the length cap and to high-entropy text, so there is no clean optimum.
  - Likewise, RL code that averages the entropy or KL penalty over tokens instead of summing it has effective coefficient β/|y|, which varies with length.
  - The identity requires summed (full-sequence) log-probs, entropy and KL.
- **Per-prompt versus averaged.**
  - The twin's optimum is pointwise in x, provided q conditions on x and is expressive enough. Otherwise there is an amortisation gap (Cremer, Li, Duvenaud 2018, "Inference Suboptimality in Variational Autoencoders", arXiv 1801.03558, VERIFIED, ICML 2018 (venue from memory, not verified)).
  - Averaged over prompts, the game makes the main model minimise E_x[log P(H|x)]. By Jensen, that is ≤ log E_x[P(H|x)], the deployment breach rate the Seldonian constraint is about.
  - The gradients differ. ∇ log E_x P(H|x) weights prompts by P(H|x) / E P(H|x). The average-log objective weights all prompts equally in log space, which spends capacity pushing already-negligible breach probabilities down further.
  - Per-prompt twin values could supply the weights, but exp(ELBO_x) underestimates P(H|x).
- **Deterministic dynamics.**
  - Token generation has deterministic "dynamics": the next state is the prefix plus the token. Levine 2018 shows max-ent RL equals exact inference in this case and is only variational, with an optimism bias, under stochastic dynamics.
  - So the sequence-level identity carries over exactly to token-level soft RL. This is also why Zhao et al.'s twists are soft Q-values.
- **The judge is part of the target.**
  - Every statement above is about P(judge says harmful), not true harm.
  - The twin's optimum p J / Z puts mass on judge false positives in proportion to p. High-probability benign text that the judge misflags is preferentially sampled, and the main model is then pushed off it (§4).

---

## 2. Source records: the variational identity in LM work

**A1. RL with KL penalties is better viewed as Bayesian inference.**
- Tomasz Korbak, Ethan Perez, Christopher L. Buckley. 2022. Findings of EMNLP 2022. arXiv 2205.11275, VERIFIED.
- **Summary:** Argues that standard RL fine-tuning of LMs leads to "distribution collapse" (a degenerate, minimal-entropy LM). Shows that KL-regularised RL is variational inference toward the posterior π* ∝ π0 exp(r/β) (eq. 5), with J_KL-RL ∝ −KL(π_θ, π*) (eq. 7). J_KL-RL "is the evidence lower bound (ELBO) on the log likelihood of π_θ being optimal under r".
- **Bearing on points 1–3:** The appendix (eqs 11–17) derives log p(O=1) ≥ E_{π_θ}[r] − KL(π_θ || π0) with r = log p(O=1|x). It says that eq. 16 "is just KL-regularised RL objective J_KL-RL(θ) with β = 1".
  - This is point 2 exactly, under the substitution π0 := current main model p and O := "judged harmful".
  - The optimal value is log p(O=1) = log Z, which is our log P_p(H|x).
  - The paper discusses only a fixed prior π0. It does not treat a moving prior or a game.
  - Its distribution-collapse discussion ("decreased fluency and diversity ... frequency of repetitions") is the β → 0 limit of §1.3.

**A2. Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review.**
- Sergey Levine. 2018. arXiv preprint. arXiv 1805.00909, VERIFIED.
- **Summary:** Max-ent RL is exact inference in a graphical model with optimality variables p(O_t|s_t,a_t) = exp(r) under deterministic dynamics. Under stochastic dynamics it is variational inference, and exact inference there gives optimistic, risk-seeking policies. An explicit temperature α multiplies the entropy term, and α → 0 recovers standard optimal control.
- **Bearing:** It is the general source of (G) as an ELBO on log p(O=1), and of the temperature reading of β_e ≠ 1. The deterministic-dynamics condition holds for autoregressive generation, so the exact equivalence applies.

**A3. A Distributional Approach to Controlled Text Generation (GDC).**
- Muhammad Khalifa, Hady Elsahar, Marc Dymetman. 2021. ICLR 2021. arXiv 2012.11635, VERIFIED.
- **Summary:** It specifies the target LM as the minimum-KL-from-a distribution satisfying constraints. For pointwise binary constraints b(x) ∈ {0,1}, the target is "P(x) = a(x)b(x)", a very simple EBM. It is approximated by Distributional Policy Gradient, which minimises the forward KL / cross-entropy CE(p, π_θ).
- **Bearing:** It is the cleanest LM statement that "p conditioned on a set" is the KL-projection target (point 2, indicator case). DPG trains the sampler with forward KL (mass-covering), which avoids the mode-dropping of the reverse-KL RL objective. That bears on twin coverage (§3).
- Related, all VERIFIED: Parshakova, Andreoli, Dymetman 2019 (DPG, arXiv 1912.08517, NeurIPS 2019 OptRL workshop); Korbak et al. 2022b (arXiv 2206.00761, RM versus distribution matching); Go et al. 2023 (f-divergence alignment, arXiv 2302.08215).

**A4. Amortizing intractable inference in large language models.**
- Edward J. Hu et al. 2023. ICLR 2024. arXiv 2310.04363, VERIFIED.
- **Authors:** The arXiv metadata, updated 31 Aug 2026, lists the final author as "Esmeralda S. Whitammer". The ICLR 2024 version lists Nikolay Malkin as final author.
- **Summary:** Fine-tunes the LM as a GFlowNet so that q(Z) ∝ R(Z), using a modified SubTB objective. For sentence continuation R(Z) = p_LM(Z|X)^{1/T}. Training draws from the policy, a tempered policy and a replay buffer, with a maximum length and forced termination. Reward-maximising methods "settle around a small number of modes".
- **Bearing:**
  - It confirms point 1 verbatim: at T = 1 "the sampler will trivially sample proportional to pLM". They use T = 0.875 and annealed T from 1.1 to 0.5, i.e., β ≠ 1 tempering.
  - Twin-side failure evidence (Appendix E): PPO on a likelihood-based reward "generates sequences with high rewards under the base model" that "are not valid expressions" and learns to "simply repeat the expression". The authors attribute this to repetitions getting high likelihood (citing Welleck et al.).
  - Their equivalence is to path consistency / max-ent RL, so the same optimum as (G). They do not discuss KL-regularised RL's optimum explicitly.

**A5. Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo.**
- Stephen Zhao, Rob Brekelmans, Alireza Makhzani, Roger Grosse. 2024. ICML 2024 (PMLR 235:60704–60748, from the proceedings page; the arXiv page lists no venue). arXiv 2404.17546, VERIFIED.
- **Summary:** The target is σ(s) = p0(s) φ(s) / Z_σ. Learned twists are soft Q-values (ψ_t = e^{βQ}) and are learned by contrastive twist learning (CTL), with SIXO, FUDGE, DPG and soft-Q baselines. The paper introduces bidirectional SMC bounds on log Z_σ.
- **Bearing (your belief is CONFIRMED):**
  - Section 7.1 has a toxic-output target σ ∝ p0 1[s ∈ C], with C = {s : r(s) ≤ η} and r the non-toxic logit. They state that "the threshold η = −5 corresponds to a greater than 99% chance of being toxic". The model is TinyStories-33M with the prompt "Once upon a time, there was a".
  - Section 7.2.1 uses σ ∝ p0 p(a=1|s), the soft-judge case. A qualitative example uses p(a=1|s)^β with β = 10.
  - For the indicator target, log Z_σ is the base model's log probability of the toxic set. This is the twin's optimal value in point 2 with p0 = main model.
  - Probability estimates are reported as bounds in Fig. 2 (graphical; no numeric values extracted). Table 2 reports KL(q||σ) and KL(σ||q), e.g., PPO 0.98 / 1.32 and contrastive twist 1.11 / 1.07.
  - Prop. 5.1: the SMC lower bound and the upper bound hold **in expectation**, not with high probability. The upper bound needs **exact target samples**. For toxicity they get these by rejection sampling from p0, which "can require hundreds of thousands of samples".
  - Plain SIS with the base proposal "generally fails to draw any samples meeting the threshold".
  - Table 1 and §5.1: PPO minimises the mode-seeking KL(q||σ) and "may cause mode-dropping"; DPG minimises KL(σ||q).
  - Appendix B.1 links the terminal potential φ = e^{βr} to RLHF and Korbak et al. 2022.
  - Neither the arXiv text nor the passages read discuss weight degeneracy by name.

**A6. Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs.**
- Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka. 2023. arXiv preprint (no venue on the page). arXiv 2306.03081, VERIFIED.
- **Summary:** Constrained generation is cast as posterior inference in Feynman–Kac models (proposal kernels M_t, potentials G_t), with hard constraints as indicator potentials. The LLaMPPL library implements it.
- **Bearing:**
  - Algorithm 1 outputs an "unbiased estimate Ẑ of the partition function". SMC "produces unbiased estimates of marginal likelihoods", but "the variance of these estimates ... depend[s] on the Feynman-Kac model".
  - For an indicator potential, Ẑ is an unbiased estimate of P(constraint), i.e., P(H) when the constraint is harm.
  - One infilling potential "corrects for the length bias from the geometrically sampled number of tokens". Length is a recognised confound for sequence-level likelihood targets.
  - The paper does not cover the variational identity or RL.

**A7. Learning diverse attacks on large language models for robust red-teaming and safety tuning.**
- Seanie Lee et al. 2024. ICLR 2025. arXiv 2405.18540, VERIFIED. The arXiv metadata again lists the final authors as "Esmeralda S. Whitammer, Moksh Jain".
- **Summary:** A GFlowNet attacker generates PROMPTS (not responses) with reward R(x) = exp((1/β) E_y[log p(c=1|x,y)]) · p_ref(x)^{1/γ} (eq. 2), trained by trajectory balance. A second stage re-trains by MLE on high-reward buffered prompts ("under 5% of total" time).
- **Bearing:**
  - Structurally this is the closest published reward: log-likelihood under a reference LM plus a log harm term, each tempered (1/γ, 1/β). The target q* ∝ p_ref^{1/γ} · exp(r/β) is the β_e ≠ 1 case of §1.3.
  - However, it is over prompts, the likelihood is under a fixed reference, and the target model is not updated in a game.
  - Failure evidence: REINFORCE "collapses to deterministically generating a single reward-maximizing prompt". PPO + novelty finds no toxic prompts against Llama-2/Gemma. GFlowNet alone is "sensitive to β and γ", with sparse rewards against safety-tuned targets (11.57% versus 85.16% transfer toxicity on Gemma-2b-it without and with the MLE stage).

**A8. Related LM red-teaming with likelihood terms** (prompt- or output-search, not a twin):
- Perez et al. 2022, "Red Teaming Language Models with Language Models". arXiv 2202.03286, VERIFIED. RL red LM with KL penalty; diversity loss at low KL.
- Hong et al. 2024, "Curiosity-driven Red-teaming for LLMs". ICLR 2024. arXiv 2402.19464, VERIFIED. Adds entropy and novelty bonuses because RL red teams collapse.
- Jones et al. 2023, ARCA, "Automatically Auditing LLMs via Discrete Optimization". arXiv 2303.04381, VERIFIED.
  - It optimises prompt-output pairs (x, o) under an auditing objective φ that "sums two expressions: log probabilities that can be exactly computed via a transformer forward pass" plus behaviour terms.
  - o is constrained to be the greedy completion, i.e., the β → 0 (mode) end of §1.3, applied to outputs.
- Wichers et al. 2024, "Gradient-Based LM Red Teaming". EACL 2024. arXiv 2401.16656, VERIFIED. Uses a realism (log-likelihood) loss on prompts.

---

## 3. Source records: rare-event probability estimation, and confidence bounds

**B1. Estimating the Probabilities of Rare Outputs in Language Models.**
- Gabriel Wu, Jacob Hilton. 2024. ICLR 2025 (proceedings PDF found in search; the arXiv page lists no venue). arXiv 2410.13211, VERIFIED.
- **Summary:**
  - The task is low-probability estimation of Pr_{x~D}[M(x) = t]. The randomness is over **inputs** (independent-token input distributions D), with **argmax** (temperature 0) single-token outputs. True probabilities lie in 1e-9 to 1e-5, the budget is about 2^16 calls, and ground truth uses 2^32 samples.
  - Importance sampling is done two ways:
    - ITGIS: a factorised Boltzmann proposal q_i ∝ p_i exp(s_i/T) built from gradient scores, with exact weights.
    - MHIS: target q(x) ∝ p(x) exp(M_t(x)/T), sampled by Metropolis–Hastings with GCG-style swaps; the normaliser is estimated, so the weights are approximate.
  - IS beats activation extrapolation (QLD, GLD), and all beat naive sampling. Itakura–Saito losses: 4-layer MHIS 1.48 and ITGIS 1.92 versus QLD 2.16 and constant 2.62.
- **Correction to the brief:**
  - This paper does not estimate the probability of sampled harmful **outputs**. The rare event is over inputs, with greedy decoding. Its MHIS target p(x) exp(M_t(x)/T) is the same Gibbs tilting as §1, but in input space.
  - It reports standard-error bands across distributions only. There are **no confidence intervals or finite-sample bounds** on individual estimates.
- **Bearing on point 3:**
  - "Minimizing the probability estimate of an undesirable behavior generalizes adversarial training".
  - IS methods give training data, "which is essentially the well-known method of adversarial training".
  - The discussion notes the estimate could be minimised "using gradient descent, if the estimate were a differentiable function of the model's parameters".
  - They do not discuss the model gaming the estimator. §1.3's lag term shows that it does.
  - Related work calls Zhao et al.'s twisted SMC "more advanced versions of our importance sampling methods".

**B2. Forecasting Rare Language Model Behaviors.**
- Erik Jones, Meg Tong, Jesse Mu, Mohammed Mahfoud, Jan Leike, Roger Grosse, Jared Kaplan, William Fithian, Ethan Perez, Mrinank Sharma. 2025. arXiv preprint (no venue on the page). arXiv 2502.16797, VERIFIED.
- **Summary:**
  - The per-query elicitation probability p_elicit(x) = E_{o~p(·|x)} 1[B(o)] is estimated by one of: the log-prob of a specific target output (one forward pass); a keyword probability with prefilling; k = 100–500 samples at temperature 1; or a multiple-choice option probability.
  - The score ψ = −log(−log p_elicit) is fitted with a Gumbel tail by OLS on the top 10 evaluation scores, then extrapolated to deployment scale (up to three orders of magnitude).
  - Worst-query log error is 1.7 versus 2.4 for log-normal. Forecasts were within an order of magnitude 72% of the time. Gumbel-tail underestimates 34% of the time.
- **Bearing:** It estimates per-query breach probabilities, but **no importance sampling or learned proposal** is used (IS is listed as future work). The paper gives **no formal confidence intervals** and lists uncertainty estimates as future work. It forecasts a max over queries, not a mean breach rate. It is relevant to §1.3's per-prompt versus averaged point: per-query p_elicit varies by orders of magnitude.

**B3. Estimating Tail Risks in Language Model Output Distributions.**
- Rico Angell, Raghav Singhal, Zachary Horvitz, Zhou Yu, Rajesh Ranganath, Kathleen McKeown, He He. 2026. ICML 2026 (arXiv comment "Accepted to ICML 2026"; PMLR v306). arXiv 2604.22167, VERIFIED.
- **Summary:**
  - This is the closest existing work to "the twin as an IS proposal for the main model's harmful tail over outputs".
  - Unsafe proposals q_φ(y|c) for a fixed prompt are built by activation steering (difference-in-means direction, scaled by λ_steer). Fine-tuning and prompting are named only as alternatives.
  - The estimator is plain IS, (1/k) Σ p_target/q_φ · h(y; c).
  - A per-token defensive mixture q_mix = (1−α) q_φ + α p_target "keeps KL(p_target||q_mix) finite", together with model switching after t_switch tokens.
  - Proposals are selected by a cross-entropy objective that targets low variance.
  - It matches brute-force MC with "10–20× fewer samples" and estimates harm probabilities around 1e-4 with about 500 samples.
  - Repeated sampling at T = 1 shows that the share of StrongREJECT queries with at least one harmful output rises from 1.6% (greedy) to 74.0% (10K samples) for Llama-3.1-8B.
- **Bearing:**
  - It is direct evidence that a harm-steered copy of the model is a workable proposal over outputs. It does not use a learned RL twin.
  - The passages read contain **no confidence intervals or finite-sample bounds** on the IS estimates, apart from Clopper–Pearson on the brute-force references. Estimates are clamped at 1e-6.
  - Weight degeneracy is not discussed in the visible text.
  - Caution: the paper's "per-token" mixture bounds the weight only by α^{−T} at sequence level (since q_mix(y) ≥ α^T p(y)). That bound is exponential in length, not 1/α (see §3 synthesis).
  - One internal inconsistency: the abstract says 10–20× and the introduction says "5% of the samples".

**B4. Sound Probabilistic Safety Bounds for Large Language Models.**
- Mahdi Nazeri, Anne-Kathrin Schmuck, Sadegh Soudjani, Alessandro Abate. 2026. arXiv preprint. arXiv 2607.20286, VERIFIED.
- **Summary:** Gives a sound **lower** bound on a fixed prompt's harm probability by summing the exact probabilities of harmful prefixes found in a tree search guided by a latent harm direction. The oracle is a keyword blacklist. There is no IS.
- **Bearing:**
  - It gives no valid upper bound. Upper bounds are left to Clopper–Pearson baselines, which need "an exceedingly large number of samples when the true value of p is small".
  - Its bounds are often tiny (1e-7 to 1e-40). In one setting, Clopper–Pearson's lower bound (0.024) beats the method (about 4e-9).
  - It illustrates that cheap rigorous bounds from tail search point the wrong way for a Seldonian certificate.

**B5. Estimating Rare Events in Language Models with Proper Evaluation.**
- Nikita Y. Parulekar, Anqi Liu. 2026. arXiv preprint. arXiv 2607.18454, VERIFIED (abstract only).
- **Summary:** Adaptive multilevel splitting in activation space (GA-AMLS) and a proper scoring rule with asymmetric under/over-estimation penalties. It reports a bias–variance trade-off, with over-estimating estimators favoured under asymmetric loss.
- **Bearing:** It is a different estimator family. Its asymmetric-penalty point matches the certification need: underestimation is the costly error.

**B6. A Statistical Approach to Assessing Neural Network Robustness.**
- Stefan Webb et al. 2019. ICLR 2019. arXiv 1811.07209, VERIFIED.
- **Summary:** Adaptive multilevel splitting with MH for the probability of property violation under an input distribution.
- **Bearing:** Prior art for rare-event estimation of network failures, cited by Wu and Hilton. It works over inputs, not LM outputs.

**B7. High-Confidence Off-Policy Evaluation (HCOPE).**
- Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh. 2015. AAAI 2015, pp. 3000–3006. No arXiv; full text read.
- **Summary:**
  - It gives a high-confidence **lower** bound on a new policy's return from importance-weighted returns, which are nonnegative but heavy-tailed: "ρ̂ often has a very long tail". The worst-case weighted return is about 10^9.4, against a largest observed value of about 316.
  - The new inequality caps values at a threshold c because "removing the upper tail of a distribution can only lower its expected value". c is chosen on a 1/20 data split.
- **Bearing (most important for the project):** This is the Seldonian ancestor of IS-based certificates, and its truncation argument works **only for lower bounds**. A breach-rate certificate needs an **upper** bound on E_q[w 1_H]. Capping w makes the estimate smaller, so it cannot be used. An upper bound needs bounded weights or a different construction (synthesis below).

**B8. Off-policy Confidence Sequences.**
- Nikos Karampatziakis, Paul Mineiro, Aaditya Ramdas. 2021. arXiv 2102.09540, VERIFIED; ICML 2021 (venue from memory, not verified).
- **Summary:** Betting-based, time-uniform confidence sequences for off-policy value with rewards in [0,1].
  - The lower side bets on ξ_i = w_i r_i − v ≥ −1, which is valid however large w gets.
  - The upper side uses E[w] = 1 to remap r → 1 − r ("betting against 1 − v in a world where r has been remapped to 1 − r").
  - They also use E[w] = 1 as a control variate.
- **Bearing:** It gives the one standard route to a valid **upper** bound on P(H) = E_q[w 1_H] with weights unbounded above: lower-bound E_q[w 1_{¬H}] = 1 − P(H). It is only as tight as the proposal's coverage of the non-harmful region. A twin concentrated on H provides almost no ¬H samples, so it needs a mixture with p (synthesis below).

**B9. Concentration and sample-size results for IS:**
- Chatterjee and Diaconis 2018, "The sample size required in importance sampling". Ann. Appl. Probab. arXiv 1511.01437, VERIFIED.
  - A sample of size about exp(D(ν||μ)), with ν the target and μ the proposal, "is necessary and sufficient", with a cut-off in log scale.
  - Here ν = p_H and μ = q, so n ≈ exp(KL(p_H || q)): the **forward** KL. A reverse-KL-trained twin that drops harmful modes has a large or infinite forward KL, even when its ELBO looks good.
- Hesterberg 1995, "Weighted Average Importance Sampling and Defensive Mixture Distributions". Technometrics 37(2):185–194. No arXiv; abstract-level.
  - Including the target as a mixture component bounds the weights.
- Kuzborskij et al. 2021, "Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting". arXiv 2006.10460, VERIFIED; AISTATS 2021 (venue from memory, not verified).
  - Confidence bounds for self-normalised IS.
- Kuzborskij and Szepesvári, "Efron-Stein PAC-Bayesian Inequalities". arXiv 1909.01931, VERIFIED.
  - "Truncation-free" bounds for weighted IS.
- Metelli et al. 2018, "Policy Optimization via Importance Sampling". NeurIPS 2018. arXiv 1809.06098, VERIFIED.
  - Rényi-divergence-based high-confidence lower bounds for policy search under IS.
- Maurer and Pontil 2009, "Empirical Bernstein Bounds". COLT 2009. arXiv 0907.3740, VERIFIED.
- Waudby-Smith and Ramdas, "Estimating means of bounded random variables by betting". arXiv 2010.09686, VERIFIED.
- Bengtsson, Bickel, Li 2008, "Curse-of-dimensionality revisited: collapse of the particle filter in very large scale systems". IMS Collections. NOT FETCHED. The standard reference that weight degeneracy grows with dimension. Sequence length plays the role of dimension: log w = Σ_t [log p(y_t|·) − log q(y_t|·)] is a sum over T tokens, so its variance grows with T.
- Grosse, Ghahramani, Adams 2015, "Sandwiching the marginal likelihood using bidirectional Monte Carlo". arXiv 1511.02543, VERIFIED.
  - Upper bounds on log Z need exact posterior samples. These are the bounds Zhao et al. extend.
- Dieng et al. 2017, CUBO. NeurIPS 2017. arXiv 1611.00328, VERIFIED.
  - A variational upper bound on log Z (χ-divergence). It holds in expectation and has high variance.

**Synthesis for the Seldonian use (inference from the sources above, not claimed by any one source):**

1. **The twin's objective value is the wrong direction for a certificate.**
   - The value is an ELBO, a lower bound on log P(H), and the main model's update widens the gap (§1.3 lag term).
   - Markov's inequality on an unbiased Ẑ gives only P(Ẑ ≥ kZ) ≤ 1/k, i.e., a lower confidence bound on P(H).
   - Upper bounds on log Z (Grosse 2015; Zhao Prop. 5.1) need exact samples from p_H. Those come only from rejection sampling p, which is the expensive thing the twin was meant to avoid. The bounds also hold in expectation, not with high probability.
2. **Where IS can still give a valid upper confidence bound on P(H):**
   - It needs weights that are bounded on H by a constant fixed in advance. The clean construction is a **sequence-level** defensive mixture q_mix = α p + (1−α) q: draw the whole response from p with probability α. Then w 1_H ≤ 1/α, and Hoeffding, empirical-Bernstein or betting bounds on [0, 1/α] are valid. The Maurer–Pontil empirical-Bernstein width, sqrt(2 V̂ ln(2/δ)/n) + 7 b ln(2/δ)/(3(n−1)) with range b = 1/α, adapts to a good twin's small variance.
   - A **per-token** mixture (as in Angell et al.) bounds w only by α^{−T}.
   - Alternatively, the Karampatziakis-style remap (upper bound via lower-bounding E[w 1_{¬H}]) is valid with unbounded weights.
3. **Coverage, not variance, is the failure that hurts upper bounds.**
   - IS is unbiased, but with a mode-dropping proposal the mass of missed harmful modes sits in rare huge weights. The typical estimate is then far below P(H) (Chatterjee–Diaconis: n ≈ exp(KL(p_H||q))).
   - The defensive component is what turns "missed mode" into "bounded loss of precision".
4. **Independence.** The twin used for certification must be fixed before drawing the certification sample, and the main model must not be trained on that sample. This is the same candidate/safety split as in HCOPE's D_pre/D_post and Seldonian practice.
5. **No source found turns an importance-sampling estimate of an LM's harmful-output probability into a finite-sample upper confidence bound.**
   - Searched: Wu and Hilton, Jones et al., Angell et al., Nazeri et al., Parulekar and Liu, Zhao et al., and the CP-based LLM certifiers that turned up in search (e.g., Lumos arXiv 2512.02966, SAFER arXiv 2510.10193; not read beyond snippets), which use plain Monte Carlo.
   - This is a search-scope statement, not a novelty verdict.

---

## 4. Source records: failure modes on the main model's side (lowering log p of specific texts)

**C1. Neural Text Generation with Unlikelihood Training.**
- Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, Jason Weston. 2019. ICLR 2020 (venue from memory, not verified). arXiv 1908.04319, VERIFIED.
- **Summary:** The token-level unlikelihood loss is −log(1 − p_θ(c|x_<t)) on negative candidates c. Sequence-level unlikelihood decodes continuations from the model itself and penalises repeated n-grams. It "substantially reduced degeneration in under 1,500 updates", mixed 50/50 with MLE.
- **Bearing:**
  - It is the earliest example of "lower the likelihood of the model's own sampled bad outputs".
  - Its loss uses log(1 − p), which is **bounded** and saturates as p → 0. That differs from the sequence-level "lower log p(y)" in the design, which is unbounded below.
  - It is always mixed with an MLE anchor.

**C2. Large Language Model Unlearning.**
- Yuanshun Yao, Xiaojun Xu, Yang Liu. 2023. arXiv 2310.10683, VERIFIED; NeurIPS 2024 (venue from memory, not verified).
- **Summary:** Gradient ascent on harmful responses, plus a random-mismatch term and a KL term to the original model on normal data, to stop harmful outputs.
- **Bearing:** Lowering log p on harmful text works only with a retain/KL anchor. Without one, utility degrades. Their harmful samples are fixed data, not on-policy.

**C3. Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning.**
- Ruiqi Zhang, Licong Lin, Yu Bai, Song Mei. 2024. COLM 2024 (venue per a DPO survey citation; the arXiv page lists none). arXiv 2404.05868, VERIFIED.
- **Summary:**
  - GA-based unlearning suffers "catastrophic collapse—a drastic degradation of the model's utilities", with outputs "often gibberish". The authors attribute it to "the linear divergence speed of gradient ascent": "each gradient step can move the model output by a constant".
  - The NPO loss (2/β) E log(1 + (π_θ/π_ref)^β) has a gradient weight that vanishes as π_θ → 0. Its "progression toward catastrophic collapse ... is exponentially slower than GA".
- **Bearing:** The design's main step is GA on a log-likelihood. Two distinctions matter:
  - (i) **On-policy with an exact twin.** The main step is ∇ log P(H) evaluated on fresh samples from p_H (§1.3), not GA on a fixed set. A string whose probability has been driven down stops being sampled, so per-string divergence is self-limiting.
  - Even so, the objective log P(H) has **no saturation**. Its gradient does not shrink as P(H) → 0, so the main model keeps spending capacity pushing log P(H) from −20 to −40. Minimising P(H) itself, or a hinge at log τ (the Seldonian threshold), saturates the way NPO does.
  - (ii) **A persisted or replayed twin buffer re-creates GA on fixed samples.** When twin samples go stale (low probability under the current p), the NPO collapse mechanism and Ren and Sutherland's squeezing effect (C4) apply.

**C4. Learning Dynamics of LLM Finetuning.**
- Yi Ren, Danica J. Sutherland. 2024. arXiv 2407.10490, VERIFIED; ICLR 2025 (venue from memory, not verified).
- **Summary:** The "squeezing effect": a negative gradient on a softmax output pushes the lost mass "largely 'squeezed' into the output which was most confident before the update". The smaller the probability of the penalised response, the stronger the squeezing. This makes the model peakier and repetition more likely, and it explains why off-policy DPO lowers almost all responses' confidence.
- **Bearing:**
  - Lowering log p on twin samples that are **unlikely** under the current p is the worst case. This happens with a lagging twin, an off-policy buffer, or a β > 1 twin that spreads over improbable harmful strings.
  - On-policy samples (y ~ p_H, which are likely under p) are the mild case.
  - This is an argument for the β_e = 1 twin with fresh samples, and for SNIS weights that down-weight stale samples.

**C5. Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization.**
- Noam Razin et al. 2024. ICLR 2025. arXiv 2410.08847, VERIFIED.
- **Summary:** Mass removed by preference training moves to other responses, sometimes ones of opposite meaning: "training a model to prefer No over Never can sharply increase the probability of Yes". Refusal training lowered Llama-3-8B-Instruct's refusal rate from 74.4% to 33.4%. Filtering pairs with high CHES (hidden-embedding similarity) mitigates it.
- **Bearing:**
  - Lowering log p on harmful y can raise the probability of other harmful y′ with similar embeddings.
  - In the twin game this is exactly what the twin will find next: the whack-a-mole dynamic of §6.
  - It also shows the main model's log P(H) need not fall even when log p(y) falls on every sampled y, unless the samples cover p_H.

**C6. A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity.**
- Andrew Lee et al. 2024. arXiv 2401.01967, VERIFIED; ICML 2024 (venue from memory, not verified).
- **Summary:** DPO on toxic/non-toxic pairs reduces toxicity in GPT-2-medium, but "capabilities learned from pre-training are not removed, but rather bypassed". A simple method un-aligns the model.
- **Bearing:** Lowering likelihood of harmful samples tends to suppress access rather than remove the capability. P(H) under the trained distribution can fall while the harmful capability stays reachable by small perturbations.

**C7. Safety Alignment Should Be Made More Than Just a Few Tokens Deep.**
- Xiangyu Qi et al. 2024. arXiv 2406.05946, VERIFIED; ICLR 2025 (venue from memory, not verified).
- **Summary:** Alignment mostly changes the distribution over the first few output tokens ("shallow safety alignment"). This explains prefilling, suffix, decoding-parameter and fine-tuning attacks. The paper proposes deeper data augmentation and a token-wise constrained objective.
- **Bearing:**
  - A main model minimising log P(H) has a cheap solution: make the first tokens refuse.
  - That cuts P(H|x) under its own sampling, which is all the game measures, while leaving p(harm | harmful prefix) high.
  - The twin, which writes whole responses from token 1, cannot apply prefill pressure unless the design includes it.

**C8. Unlearning robustness (relearning, shallow unlearning):**
- Hu et al. 2024, "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning". ICLR 2025. arXiv 2406.13356, VERIFIED.
  - Small, loosely related benign data restores unlearned knowledge (e.g., public medical articles restore bioweapon content). Approximate methods "simply suppress the model outputs".
- Deeb and Roger 2024, "Do Unlearning Methods Remove Information from Language Model Weights?". arXiv 2410.08827, VERIFIED; venue not verified.
  - Fine-tuning on accessible facts "can recover 88% of the pre-unlearning accuracy".
- Łucki et al. 2024, "An Adversarial Perspective on Machine Unlearning for AI Safety". TMLR. arXiv 2409.18025, VERIFIED.
  - "Finetuning on 10 unrelated examples" or removing activation directions restores RMU-unlearned hazardous capabilities.
- Lynch et al. 2024, "Eight Methods to Evaluate Robust Unlearning in LLMs". arXiv 2402.16835, VERIFIED.
- Maini et al. 2024, TOFU. arXiv 2401.06121, VERIFIED.
- Li et al. 2024, WMDP / RMU. arXiv 2403.03218, VERIFIED.
- Jain et al. 2023, "Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks". arXiv 2311.12786, VERIFIED. Fine-tuning learns "wrappers".
- **Bearing:** Lowering the likelihood of sampled harmful text is expected to suppress surface forms and leave relearnable capability. The certificate measures P(H) under the deployed weights, which is exactly what suppression lowers. Robustness to tampering is a separate claim that this mechanism does not support.

---

## 5. Source records: failure modes on the twin's side (likelihood rewards)

**D1. The mode of an LM is short, generic or repetitive** (β_e → 0 end of §1.3):
- Stahlberg and Byrne 2019, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?". EMNLP 2019. arXiv 1908.10090, VERIFIED. Exact search shows the empty translation is often the global mode.
- Eikema and Aziz 2020, "Is MAP Decoding All You Need? The Inadequacy of the Mode in NMT". COLING 2020. arXiv 2005.10283, VERIFIED.
- Murray and Chiang 2018, "Correcting Length Bias in NMT". WMT 2018. arXiv 1808.10006, VERIFIED.
- Meister, Cotterell, Vieira 2020, "If beam search is the answer, what was the question?". EMNLP 2020. arXiv 2010.02650, VERIFIED.
- Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration". ICLR 2020. arXiv 1904.09751, VERIFIED. High-likelihood decoding is bland and repetitive.
- Xu et al. 2022, "Learning to Break the Loop". NeurIPS 2022. arXiv 2206.02369, VERIFIED. The self-reinforcement effect: the probability of repeating rises with the number of repeats.
- **Bearing:**
  - With full-sequence log p and entropy coefficient 1, the optimum is p itself, which carries p's own length distribution, so there is **no length bias at the optimum**.
  - Length bias and repetition appear when the effective entropy coefficient is below 1 (β_e < 1 or β_e = 0 with KL to a reference). They also appear when the RL optimiser has not reached the optimum and climbs the reward greedily (Hu et al. Appendix E: PPO "learns to simply repeat the expression"), or when entropy or KL is averaged per token.
  - Interaction with the harm term: at β_e < 1 the twin targets the most likely harmful strings. These tend to be short, formulaic and borderline (e.g., a harmful first sentence, then a stop), because each extra token costs log-probability.

**D2. Judge exploitation.**
- Gao, Schulman, Hilton 2022, "Scaling Laws for Reward Model Overoptimization". arXiv 2210.10760, VERIFIED. Proxy-reward optimisation diverges from the gold reward as KL grows.
- **Bearing:**
  - q* ∝ p J samples judge false positives in proportion to p. High-probability benign text the judge misflags gets oversampled relative to true harm.
  - The main step then lowers likelihood on benign high-probability text (over-refusal, utility loss).
  - With a soft J, "barely harmful but very likely" text has weight J(y) p(y) and can dominate p_J when true harm is rare.
  - Mitigations: indicator or thresholded J, a stronger judge, and human-audited samples of the twin's outputs.

**D3. Length normalisation choices.**
- Meng et al. 2024, SimPO. NeurIPS 2024. arXiv 2405.14734, VERIFIED. Uses average log-probability as an implicit reward, to stop length exploitation in preference optimisation.
- **Bearing:**
  - In the twin, average log-prob reward plus entropy has no proper optimum (§1.3). It cannot be used if the value is meant to be log P(H).
  - The clean choice is summed log-probs with summed entropy at coefficient 1 and a fixed maximum length; H must then be defined on sequences that end within it.
  - What the reference papers do:
    - GFlowNet (Hu): maximum and minimum lengths, a forced terminator, reward temperature.
    - Lee: a p_ref^{1/γ} weight plus an MLE smoothing stage.
    - Zhao: an exact target with SMC/twists, so no RL on the likelihood reward.
    - Lew: potentials that correct length bias.

**D4. Exploration and sparse reward at initialisation.**
- The twin starts at q = p_0. With an indicator harm and a safety-tuned main model, almost no initial samples land in H.
- Zhao: base-proposal SIS "generally fails to draw any samples meeting the threshold". Lee: GFlowNet toxicity falls on safety-tuned targets because of sparse rewards. Hu: performance is "quite poor" without seeded replay rationales.
- **Bearing:** The twin needs a harmful warm start (e.g., Angell-style steering, or a fine-tuned harmful copy) or a seeded replay buffer. Its first q is then far from p, so KL(q||p) starts large. The ELBO starts very loose and the lag term (§1.3) is largest early.

**D5. Mode-seeking.**
- PPO minimises KL(q||σ) (Zhao Table 1). RL red-teamers collapse to single modes (Lee; Hong; Perez).
- **Bearing:** The twin will cover a few harmful modes, so the main model's update only suppresses those (whack-a-mole). Forward-KL training (DPG or RWS wake-phase q-update) or GFlowNet off-policy training is the literature's remedy. Diversity bonuses are a heuristic one.
- Rainforth et al. 2018, "Tighter Variational Bounds are Not Necessarily Better". ICML 2018. arXiv 1802.04537, VERIFIED. If the twin is trained with an importance-weighted (IWAE-type) bound, more particles worsen the proposal gradient's signal-to-noise ratio.

---

## 6. Source records: alternation dynamics (wake-sleep, PCD, amortised EBM samplers)

**E1. Training RBMs using approximations to the likelihood gradient (PCD).**
- Tijmen Tieleman. 2008. ICML 2008. No arXiv; the icml.cc PDF returned 404, so it was not read directly.
- **Summary:** Persistent "fantasy particle" chains are not reset between weight updates. As described in Tieleman and Hinton 2009, they outperform CD for a fixed compute budget.

**E2. Using Fast Weights to Improve Persistent Contrastive Divergence.**
- Tijmen Tieleman, Geoffrey Hinton. 2009. ICML 2009, pp. 1033–1040. No arXiv; full text read (cs.toronto.edu/~hinton/absps/fpcd.pdf).
- **Summary:**
  - "If the learning rate is sufficiently small compared with the mixing rate of the Markov chain, this persistent chain will always stay very close to the stationary distribution". Then "the expected weight updates follow the gradient of the log likelihood".
  - If the chain samples from R ≠ Q_θ, "there is an additional term ... the gradient of KL(R||Q_θ) and it moves the model's distribution away from R which is bad, but this has the effect of making the persistent chain mix much faster which is good".
  - Mechanism: "Fantasy particles develop an aversion to wherever they are and unless this aversion is balanced by the energy-lowering presence of training data, they continually seek out places they have not recently visited".
  - Decaying the learning rate reduces "the changes in the energy landscape that cause fast mixing". Fast weights (fast learning, fast decay) restore it.
- **Bearing:** This is the closest structural analogue, and it maps term for term:
  - the twin is the persistent sampler;
  - the main model's "lower log p on twin samples" is the negative phase;
  - §1.3's lag term is their extra ∇KL(R||Q_θ) term.
- The key difference: **the design has no positive phase.** No "energy-lowering presence of training data" balances the aversion inside H. Outside H, any retain or KL anchor is the positive phase.
- Expected regime:
  - The twin "continually seeks out places it has not recently visited" in H.
  - The main model keeps raising energy where the twin was.
  - The pair is stable only if the main step is small relative to the twin's re-adaptation rate (their small-learning-rate condition).
  - Fast-weights-style decoupling suggests one remedy: a fast-adapting twin head, or more twin steps per main step.

**E3. Reweighted Wake-Sleep.**
- Jörg Bornschein, Yoshua Bengio. 2014/2015. ICLR 2015 (venue from memory, not verified). arXiv 1406.2751, VERIFIED.
- **Summary:**
  - Wake-sleep's sleep phase minimises KL(p(h|x)||q) "instead of" the correct KL(q||p(h|x)).
  - RWS interprets the inference network as an IS proposal. The p-update uses self-normalised weights ω̃_k, giving a biased but consistent gradient whose bias and variance fall with K.
  - The wake-phase q-update with the same weights "is equivalent to optimizing q so as to minimize KL(p(·|x) || q(·|x))".
- **Bearing:**
  - The main model's debiased update (§1.3) is the RWS p-update.
  - The RWS wake-phase q-update is a forward-KL (mass-covering) way to train the twin from its own weighted samples, as an alternative to reverse-KL RL.
- Le et al. 2019, "Revisiting Reweighted Wake-Sleep for Models with Stochastic Control Flow". UAI 2019. arXiv 1805.10469, VERIFIED. RWS learns better proposals than IWAE in discrete or branching models, and its q-learning does not degrade with more particles. LM sequences are such discrete models.

**E4. Amortised samplers for EBMs (the same min–max with the same identity):**
- Grathwohl et al., "No MCMC for me: Amortized sampling for fast and stable training of EBMs" (VERA). ICLR 2021. arXiv 2010.04230, VERIFIED.
  - Uses eq. 3, "log Z(θ) = max_q E_q[f_θ] + H(q)", with an online sampler. "When training p_θ(x) and q(x) online together, the inner maximization will not be fully optimized. This leads our training objective for p_θ(x) to be an upper bound on log p_θ(x)".
  - The paper notes that MCMC-based EBM training is "slow and unstable".
- Kim and Bengio 2016, "Deep Directed Generative Models with Energy-Based Probability Estimation". arXiv 1606.03439, VERIFIED (abstract only). The generator matches the EBM.
- Dai et al. 2017, "Calibrating Energy-based GANs". ICLR 2017. arXiv 1702.01691, VERIFIED. An entropy-regularised generator.
- Geng et al. 2021, "Bounds all around: training EBMs with bidirectional bounds". NeurIPS 2021. arXiv 2111.00929, VERIFIED.
  - "Both minimizing and maximizing a lower bound is potentially unstable. By minimizing a lower bound, we run the risk of finding an optimum, where the bound is loosest", and "a minima of the lower bound may be −∞".
  - They propose maximising a lower bound and minimising an upper bound.
- Nijkamp et al. 2019, "Learning Non-Convergent Non-Persistent Short-Run MCMC Toward EBMs". arXiv 1904.09770, VERIFIED.
- Nijkamp et al. 2020, "On the Anatomy of MCMC-Based ML Learning of EBMs". AAAI 2020. arXiv 1903.12370, VERIFIED.
  - Short-run, non-mixing samplers still give a usable model, but its density no longer matches the sampler's. The "learned sampler" and the "model" can decouple.
- **Bearing:**
  - The harm-twin game is the EBM–sampler game with energy −log p_θ − log J restricted to the harmful region. The main model plays the EBM side, but **minimises** log Z_H instead of maximising data likelihood minus log Z.
  - With a non-optimal twin, the main model minimises a **lower bound** on log P(H). This is Geng et al.'s warned-against configuration: the main model can lower the bound by widening the gap KL(q||p_H) instead of lowering P(H).
  - Geng et al.'s remedy suggests pairing the ELBO with an upper bound (CUBO, or IS with a defensive mixture) when the main model's progress is measured.
  - For the certificate, the upper bound must come from fresh, independent samples (§3 synthesis).

---

## 7. Verdict on the derivation, point by point

**Point 1: CONFIRMED.**
- With reward log p(y|x), full-sequence entropy coefficient 1, and q free to be any distribution over sequences for each x: q* = p and the value is 0. The objective equals −KL(q||p).
- Stated in Hu et al. 2024 ("at T = 1 ... trivially sample proportional to pLM"). It is a special case of Korbak et al. 2022 eq. 16 with r = 0, and of VERA eq. 3 with f = log p.
- A harm term is required.

**Point 2: CONFIRMED,** with the precisions in §1.2:
- q* = p J / E_p[J], and the value is log E_p[J]. For an indicator these are p_H and log P_p(H|x).
- Any other q gives log P(H) − KL(q||p_H), an ELBO.
- The objective is identical to "reward log J, KL penalty to the current main model, coefficient 1". This is Korbak et al.'s eq. 14–16 with π0 = p.
- The indicator case is GDC's P(x) = a(x)b(x).
- Zhao et al. use exactly this target for toxicity (σ ∝ p0 1[r(s) ≤ η]), so the twin's optimal value is their log Z_σ.
- **Needs care:** for an indicator, the objective is −∞ unless supp(q) ⊆ H. A smoothed log(J + ε) changes the target to a mixture that is mostly p unless ε ≪ P(H). A soft J gives a randomized-judge rate.

**Point 3: CONFIRMED at the twin's optimum, with two additions and three corrections.**
- **Confirmed:**
  - min_θ max_q equals min_θ log P_θ(H|x).
  - The main step on samples from p_H is unbiased gradient descent on log P(H), because ∇ log P(H) = E_{p_H}[∇ log p].
  - The twin is an amortised sampler and IS proposal (Hu; Zhao; Wu and Hilton; Angell), and it is zero-variance at the optimum.
- **Addition 1 (derived here):** away from the optimum, the main step equals ∇ log P(H) − ∇_θ KL(q||p_H). It also widens the twin's gap, which is the PCD extra term of Tieleman and Hinton 2009. The main model is minimising a lower bound on its own breach log-probability, the configuration Geng et al. 2021 warn about. SNIS (RWS) weights p 1_H / q remove the bias.
- **Addition 2:** averaged over prompts, the game minimises E_x log P(H|x). That differs from the deployment breach rate log E_x P(H|x): Jensen, and different gradient weighting.
- **Correction (a): entropy coefficient β ≠ 1.** q* ∝ (p J)^{1/β}, and the value log ‖p 1_H‖_{1/β} is below log P(H) for β < 1 (→ log max_{y∈H} p(y) as β → 0) and above it for β > 1.
- **Correction (b): KL to a fixed reference π0 with coefficient β_r, plus entropy β_e.** q* ∝ π0^{β_r/β} (p_t J)^{1/β}, with β = β_e + β_r.
  - Standard PPO-RLHF (β_e = 0, π0 = p_0, β_r = 1) with reward log p_t + log J gives q* ∝ p_0 p_t J. That is p_0² J at initialisation, not p conditioned on harm.
  - To get p_t J, either drop the reference KL and use summed entropy at coefficient 1, or drop the log p reward and use KL to the current p_t.
  - Reward log J with KL to a fixed p_0 tracks the base model, not the main model.
- **Correction (c): length-normalised likelihood reward, or per-token-averaged entropy/KL.** This breaks the identity. With average log-prob there is no proper optimum short of the length cap.

**What the derivation does not give (unsupported as stated, if read as a certificate):** the twin's objective value is a lower bound on log P(H). It cannot serve as, or be turned by Markov's inequality into, a high-confidence upper bound on breach rate. Valid upper bounds need:
- exact p_H samples (BDMC; Zhao Prop. 5.1, in expectation only); or
- IS with weights bounded on H (a sequence-level defensive mixture, giving bound 1/α) plus a bounded-variable concentration or betting bound; or
- a betting confidence sequence using E[w] = 1 (Karampatziakis et al.).

In every case the samples must be fresh and independent of the training game. No LM source found does this.

**Assumptions list:**
1. **Full-sequence quantities:** summed sequence log-probs log p(y|x) = Σ_t log p(y_t|x, y_<t), summed sequence entropy or KL, and a fixed maximum length with H defined on terminated sequences. This is not length-normalised.
2. **Entropy coefficient exactly 1.** Equivalently, KL(q||p_t) to the current main model at coefficient 1 with no log p reward. No additional KL to a fixed reference.
3. **Harm term enters as log J** (or log J^λ), not as an additive λJ. For an indicator, supp(q) ⊆ H, or smoothing ε ≪ P(H).
4. **Realizability and optimisation:** for every x the twin can represent and reach p_H(·|x). Otherwise there is an amortisation gap (Cremer et al. 2018), plus an RL optimisation gap and the alternation lag.
5. **Per-prompt identity.** Averaging over prompts gives E_x log P(H|x), not log E_x P(H|x).
6. **Deterministic token dynamics** (true for autoregressive generation), so sequence-level max-ent RL equals exact inference (Levine 2018).
7. **The judge defines H.** Every probability is P(judge flags) under the chosen decoding (temperature 1, no truncation such as top-p, unless p is redefined to include it). It is not true harm.
8. **For the main-step unbiasedness:** samples come from the current p_H, and are fresh, not replayed. Otherwise SNIS-correct or accept the lag term.

# Focused gap check: Coupling 2 ("harm twin" rewarded by the main model's likelihood)

Date: 2026-10-17. Adversarial novelty check: the aim was to falsify the claim, not confirm it. Follows gap_couplings_1_2.md (read that first; its section D is the earlier Coupling-2 pass).

**Design under test.** A "harm twin" policy starts from the same base as the main generative LM. It writes RESPONSES to user prompts and is trained with RL. Its reward is the MAIN model's likelihood of the twin's harmful response (log-prob, or log-odds against a safe/refusal response), usually plus a harm score. The main model is then updated to lower its likelihood of those responses (unlikelihood, NPO, DPO rejected side). The two updates alternate.

**Claim under test.** No published paper trains such a twin (separate policy, same base, RL, response-generating, reward = main model's likelihood of the twin's harmful output) and alternates it with updates to a generative main model.

**Already known, not counted as falsifying:** Lyu 2606.04168, APL 2505.24369, AdvPrompter 2404.16873, GFlowNet red-teaming 2405.18540, Emulated Disalignment 2402.12343, JVD 2408.10668, SGALM 2602.01137, CAT 2405.15589, LAT (Casper 2024, Sheshadri 2024), APRM 2511.22888, Adv-RM 2504.06141, Token Buncher 2508.20697, AntiDote 2509.08000, ADPO 2405.12900, MACPO 2410.07672, SPIN 2401.01335.

**Axes:** (a) separate learned policy; (b) same base; (c) RL; (d) generates responses, harmful; (e) reward = main model's likelihood of its output; (f) alternating or iterated with updates to a generative main model.

## A structural observation that affects the wording of the claim

The twin's reward of harm(y) + beta * log p_main(y|x), plus an entropy bonus of weight beta, equals KL-regularised RL with the CURRENT main model as the reference policy. That is, E_q[harm] - beta * KL(q || p_main). The optimum is q*(y|x) proportional to p_main(y|x) * exp(harm(y)/beta): the main model tilted toward harm.

This is the same target distribution that twisted SMC and rare-event importance-sampling proposals learn (records B1-B3). So "reward defined by the main model's likelihood" is not a new objective on its own. What is new, if anything, is the combination:
1. the reference/likelihood term tracks a main model that is being pushed DOWN on the twin's samples;
2. the loop alternates;
3. the domain is harmful responses.

A reviewer will make this equivalence, so the claim should say it first.

Consequence for falsification: any harmful RL attacker that keeps a KL penalty to the CURRENT defended model, inside an alternating defence loop, is close to an instance of the design. Such attackers appear in simulated-attacker defences like Token Buncher and AntiDote, both excluded by the caller. I did not re-check their KL reference. If either attacker is a separate copy with a KL penalty to the live defended model, that paper comes within one axis of the design: the reward term would be implicit rather than explicit. Worth a direct check before submission.

## Candidate records (new in this pass)

### A. Closest structural matches: a same-base sampler rewarded by the main LM's likelihood, then the main LM is updated

**A1. Amortizing Intractable Inference in Large Language Models.** Edward J. Hu, ICLR 2024 (oral). arXiv 2310.04363, VERIFIED (HTML method section read).
- **Mechanism:**
  - The GFlowNet sampler q_GFN is initialised "as a copy of p_LM" (LoRA) and fine-tuned with a subtrajectory-balance (diversity-seeking RL) objective.
  - Reward: "R(Z) = p_LM(XZY) ∝ p_LM(Z|X,Y)", i.e. the main LM's likelihood of the sampler's output.
  - EM-style M-step (sec. 3.2, used in sec. 4.3): "updating p_LM to maximize log p_LM(XZY) over a collection of Z's sampled from the amortized posterior."
- **Axes:**
  - (a) yes (LoRA sampler, separate from p_LM)
  - (b) yes
  - (c) yes (GFlowNet = amortised RL)
  - (d) generates latent rationales or infills, NOT harmful
  - (e) YES, reward is exactly the main LM's likelihood
  - (f) partly: one EM M-step, and it RAISES the main LM's likelihood of the samples (cooperative), not lowers it
- **Verdict:** The strongest anticipation of Coupling 2's MECHANICS: same-base RL sampler, reward = main LM likelihood, main LM then updated on its samples. What it misses is the adversarial sign and harm. The design is the adversarial, harm-restricted inversion of GFlowNet-EM. Cite it.

**A2. Unsupervised Text Style Transfer using Language Models as Discriminators.** Zichao Yang, NeurIPS 2018. arXiv 1805.11749, VERIFIED (ar5iv method and appendix read).
- **Mechanism:**
  - The generator is trained to "minimize the NLL of generated sentences, evaluated by the language model" (target-style LM). This is a continuous relaxation of sampling, not RL.
  - In the "LM + adv" variant the LM is trained "by minimizing the loss of LM of real sentences and maximizing the loss of transferred sentences." "Training consists of two steps alternatively."
  - The authors found this variant "very unstable" and set gamma = 0 for the main results.
  - The LM is "trained from scratch", not shared with the generator.
- **Axes:**
  - (a) yes
  - (b) no
  - (c) no (Gumbel-style relaxation)
  - (d) generates transferred sentences, not harmful, not prompt-conditioned assistant responses
  - (e) YES, the generator's objective is the LM's likelihood of its output
  - (f) YES in the LM+adv variant: alternating, and the LM is pushed to LOWER the likelihood of the generated sentences
- **Verdict:** The exact min-max-over-likelihood game of Coupling 2 (generator raises LM likelihood of its samples; LM lowers it), in 2018. But the LM is a style discriminator, not a deployed generative policy. Bases differ, there is no RL and no harm. Its reported instability is relevant prior evidence for the design's risk.

**A3. Learned-sampler training of energy-based models (generic template).** Kim & Bengio 2016, "Deep Directed Generative Models with Energy-Based Probability Estimation" (arXiv 1606.03439, UNVERIFIED by fetch). Kumar et al. 2019, "Maximum Entropy Generators for Energy-Based Models" (arXiv 1901.08508, VERIFIED abstract only).
- **Mechanism** (abstract plus standard knowledge, not checked against the method section): an amortised generator maximises the EBM's log-density of its samples plus an entropy term. The EBM lowers the density of generator samples and raises the density of data. The two alternate.
- **Axes:** (a) yes; (b) no; (c) reparameterised gradients, not RL; (d) images, not harm; (e) yes (the model's unnormalised log-density); (f) yes (the model pushes down on samples).
- **Verdict:** The design is this template with an RL sampler restricted to a harmful region and no data-side push-up. General ML precedent only.

### B. Amortised samplers of the main model restricted to undesirable outputs (the twin's target distribution), with no main-model training

**B1. Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo.** Stephen Zhao, ICML 2024. arXiv 2404.17546, VERIFIED (sec. 7.2.1 read).
- **Mechanism:**
  - Target sigma(s) proportional to p0(s) * p(a=1|s), where a=1 means a toxicity classifier calls the text toxic (TinyStories-33M).
  - Learned twists, trained by contrastive twist learning (linked to soft RL), give a proposal for the main model's own toxic outputs.
  - The paper calls sampling undesirable outputs "a useful component of harmlessness training and automated red-teaming", but it does not do that training.
- **Axes:** (a) yes (twist network); (b) not stated; (c) soft-RL-like; (d) toxic continuations; (e) yes, implicitly (target includes p0, the main model); (f) NO.

**B2. Adaptive Multilevel Twisted Sequential Monte Carlo for Rare Events Estimation in Language Models.** Zixuan Liu, 2026 (v1 22 Aug 2026). arXiv 2608.21736, VERIFIED (method read).
- **Mechanism:**
  - sigma*(x) = p0(x) * 1{R(x) >= gamma*} / p*, with events from ToxiGuardrail and the StrongREJECT evaluator (jailbreaks).
  - Twist-induced proposal q(x_t|x_<t) proportional to p0(x_t|x_<t) * psi_t, learned through progressively rarer levels.
  - "Algorithm 1 updates only the twist parameters"; p0 stays fixed. Using the found outputs "as training ... examples for existing alignment methods" is listed as future use.
- **Axes:** (a) yes; (b) twists over p0; (c) contrastive twist learning, not RL; (d) harmful responses; (e) yes, implicitly; (f) NO.

**B3. Rare Event Estimation via Iterative Unalignment.** Hanming Yang, 2026 (21 Sep 2026). arXiv 2609.24969, VERIFIED (method read).
- **Mechanism:**
  - Proposal q_delta = p_{theta+delta}, a weight-perturbed copy of the base, starting at delta = 0.
  - Objective: an event surrogate S_delta minus lambda_t * KL(q_delta || p). The KL term is equivalent to rewarding log p plus entropy.
  - Gradient ascent through the model evaluations on sampled trajectories; "not REINFORCE".
  - The base "parameters remain fixed throughout".
- **Axes:** (a) yes; (b) YES (same weights plus delta); (c) no (pathwise gradient on fixed samples); (d) proxy events (token presence, profanity), motivated by harm; (e) implicitly, via the KL to the main model; (f) NO.

### C. Static or search precedents for "main-model likelihood plus harm" on the RESPONSE side

**C1. AdvPrefix: An Objective for Nuanced LLM Jailbreaks.** Sicheng Zhu, NeurIPS 2025. arXiv 2412.10321, VERIFIED (search abstract plus HTML snippets).
- **Mechanism:** Candidate harmful RESPONSE PREFIXES are generated by uncensored LLMs. They are selected per victim by combining high prefilling attack success with low NLL under the victim, i.e. victim likelihood plus harm.
- **Axes:** (a) no learned policy; (b) n/a; (c) no; (d) harmful response prefixes; (e) YES, selection uses victim likelihood combined with harm, which is the twin's combined reward used as a filter; (f) no (the victim is never trained).
- **Verdict:** A second static precedent for the combined reward, besides Lyu 2606.04168.

**C2. Exposing Long-Tail Safety Failures ... (PDPS).** Suvadeep Hajra, TMLR 2026. arXiv 2603.14355, VERIFIED (abstract only).
- **Mechanism:** Diversity-aware population sampling of the target's OWN responses to a fixed harmful prompt. An aggregator claims the samples were used in RLHF safety tuning; not confirmed in the abstract.
- **Axes:** (a) no; (d) yes; (e) on-policy sampling, so samples follow the main model's likelihood implicitly; (f) possibly a single round.

**C3. Adversarial Contrastive Decoding (Safety Alignment via Contrasting Safe and Harmful Distributions).** Xiaoyun Zhang, AAAI 2026. arXiv 2406.16743, VERIFIED (abstract).
- **Mechanism:** A learned adversarial soft prompt brings out the model's own harmful tendencies, used contrastively at decoding time. Weights are untouched.
- **Axes:** (a) a soft prompt, not a policy; (c) no; (f) no.

### D. Same-base RL generator of bad RESPONSES, alternating, but the main model is a detector and the reward is its failure (adds to APRM and Adv-RM)

**D1. Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator.** Shiping Yang, 2026 (v1 8 Jul 2026, v2 7 Oct 2026). arXiv 2607.07993, VERIFIED (abstract).
- **Mechanism:**
  - The generator is "initialized from the same base model" as the detector and trained with RL using the detector as its reward model, to write hard-to-detect hallucinated responses.
  - The detector is then trained on the generator's data, iteratively.
- **Axes:** (a) yes; (b) yes; (c) yes; (d) responses, hallucinated not harmful; (e) NO, the reward is the detector's failure, not a likelihood; (f) yes, but the main model is a detector.
- **Verdict:** Same position as APRM: Coupling-1 shape, with the wrong role for the main model.

### E. Co-evolving safety games from 2025-2026 (all prompt- or task-side; none of them is the twin)

- **AdvGame (Safety Alignment of LMs via Non-cooperative Games).** Anselm Paulus, arXiv 2512.20806, VERIFIED (HTML).
  - The Attacker rewrites seed queries into attack PROMPTS. It is a separate model (Qwen2.5-7B, or Llama-3.1-8B-Abliterated).
  - Preference-based rewards with a DPO-style loss. Updates are concurrent: "Rather than alternating updates."
  - No likelihood reward.
- **ACE-Safety.** Xurui Li, arXiv 2511.19218, VERIFIED (HTML).
  - The attacker rewrites QUERIES, on the same backbone as the defender.
  - The attack reward is judge harmfulness plus relevance.
  - The defender's refusal log-prob, P^D = 1 - log p^D(first token is a refusal | q), enters the MCTS selection prior, not the reward.
  - Iterative rounds with GRPO-style defender training.
  - Interesting as a defender-likelihood signal on the PROMPT side.
- **AgentSafeEvo.** Chenxu Wang, arXiv 2610.19422 (14 Oct 2026), VERIFIED (HTML).
  - The attacker builds agent TASKS (query, tools, observations). Both players are Qwen3-8B, trained jointly with GRPO.
  - Attacker reward = minus the defender's judge reward. No likelihood.
- **Eliciting LM Behaviors with Investigator Agents.** Xiang Lisa Li, ICML 2025, arXiv 2502.01236 (VERIFIED via the PMLR and arXiv listing).
  - A PROMPT investigator is trained (SFT, DPO, Frank-Wolfe) to raise the target's likelihood of a target behaviour, including harmful responses.
  - The target is not trained. Prompt-side analogue of AdvPrompter.

### F. Unlearning and on-policy safety (same weights, or prompt-side elicitation)

- **Reveal and Release.** Linxi Xie, EMNLP 2025 Findings, arXiv 2509.14624, VERIFIED (abstract). Optimised instructions bring out the model's own forget-relevant outputs, then PEFT unlearning is iterated. The elicitation is prompt optimisation (NeuralUCB), not a response policy.
- **Self-ReSET.** Dongcheng Zhang, arXiv 2605.08936, VERIFIED (abstract). The model's own unsafe reasoning trajectories become RL initial states. Same weights.
- **RULE / ReRULE (Replay What Matters).** Zirui Pang, arXiv 2606.15333, VERIFIED (abstract). GRPO unlearning on the model's own rollouts. Same weights.
- **CRINGE loss.** Leonard Adolphs, ACL 2023, arXiv 2211.05826. Iteratively labels the model's own generations and applies a contrastive unlikelihood loss. Same weights, no learned sampler.
- **REBEL.** Patryk Rybak, arXiv 2602.06248. An evolutionary prompt "hacker" against unlearned models. Prompt-side, evaluation only.
- **Adversarial RL for Robust Diffusion LLM Unlearning.** ICML 2026. arXiv still UNVERIFIED (title confirmed only in the ICML 2026 listing; no abstract found). An earlier pass read it as a context (prompt) generator. If it turns out to generate COMPLETIONS rewarded by the dLLM's recall likelihood, it would be close. Re-check when an arXiv version appears.

## Search scope

- **Searched (WebSearch standard and extended, with WebFetch on arXiv abs and HTML method sections):**
  - RL response generators with a target-likelihood reward
  - on-policy harmful completions; self-generated harmful responses; online unlikelihood and NPO with a learned adversary
  - adversarial or min-max unlearning with a trained generator; RL attackers against unlearned models
  - GFlowNet and twisted-SMC samplers of harmful or rare outputs; rare-event importance sampling for LMs
  - variational and wake-sleep proposals; GFlowNet-EM
  - learned prefill or prefix attackers (AdvPrefix line)
  - "evil twin", "shadow", "self-adversarial" GRPO safety work; co-evolving attacker-defender games from 2025-2026
  - LM-as-discriminator text GANs; EBM learned-sampler training
  - distributionally robust safety with a generative adversary
- **Coverage:** 2016 to October 2026.
- **Not searched or not resolved:**
  - non-arXiv venues (OpenReview-only, workshops), patents, non-English venues
  - full method sections of PDPS, Self-ReSET, RULE and the diffusion-LLM unlearning paper (abstracts only)
  - the KL-reference choice in Token Buncher and AntiDote's simulated harmful-RL attackers (see the structural observation above)
  - Kim & Bengio 2016 not fetched

## Verdict

**NOT FALSIFIED on the full conjunction (a-f, with a generative main model pushed down on harmful twin responses). PARTIALLY ANTICIPATED, more strongly than the earlier pass showed.**

| Paper | Axes met | Axes missed |
|---|---|---|
| Hu et al. 2310.04363 (GFlowNet-EM) | a, b, c, e, f (one EM step) | d (not harmful); the main LM is updated to RAISE the likelihood of samples (cooperative sign) |
| Yang et al. 1805.11749 (LM + adv) | a, e, f (alternating; LM lowers likelihood of generated text) | b, c (no RL), d; main model is a style discriminator, not a deployed policy |
| EBM learned samplers (Kumar 2019 and others) | a, e, f | b, c, d; continuous domains |
| Zhao 2404.17546; Liu 2608.21736; Yang 2609.24969 | a, (b), d (toxic/jailbreak), e implicit (target = p_main tilted by harm) | c (not standard RL in two of the three), f (main never trained) |
| AdvPrefix 2412.10321; Lyu 2606.04168 | d, e (main likelihood plus harm, as a selection rule) | a, c, f |
| Hallucination Self-Play 2607.07993; APRM 2511.22888 | a, b, c, d (bad responses), f | e (reward is detector failure); main model is a detector |
| AdvGame, ACE-Safety, AgentSafeEvo, investigator agents | a, c, f | d (prompts or tasks); e (only ACE uses defender likelihood, as a search prior) |

**How the claim should be worded after this pass:**
- Present the twin as the adversarial, harm-restricted inversion of amortised-inference EM (GFlowNet-EM, Hu et al. 2024). The sampler's target is the main model tilted toward harm, as in twisted-SMC and rare-event proposals (Zhao 2024; Liu 2026; Yang 2026). The novelty is training the main model DOWN on that sampler's output and re-fitting the sampler each round.
- Cite Yang et al. 2018 as the earliest likelihood min-max (generator vs LM discriminator) and note that it was unstable.
- Do not claim that "reward = main model's likelihood" is new. With an entropy bonus it equals KL-regularised RL with the main model as reference.

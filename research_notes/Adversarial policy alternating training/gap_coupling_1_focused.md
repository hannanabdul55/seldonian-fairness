# Focused gap check: coupling 1, negatives (harm twin's RESPONSES as negatives, alternating)

Date: 2026-10-17. This is an adversarial novelty check. The aim was to falsify the claim, not confirm it. It extends gap_couplings_1_2.md (same folder) and does not repeat its records except for comparison.

**Design under test.** A "harm twin" policy is initialised from the same base as the generative main LM and trained with RL to write harmful RESPONSES to user prompts. Its responses are negatives for the main model (DPO rejected side, unlikelihood, contrastive or negative-teacher losses, NPO). Twin and main model alternate, and the twin is re-trained against the current main model each round.

**Claim.** No published paper trains a separate, same-base, RL-trained generator of harmful responses that is the negative source for a generative main model and is updated in alternation against the current main model.

**Structural point.** If the twin's reward is only a judge's harmfulness score, the twin does not depend on the main model, and alternation creates no game. So each candidate is classed by how the generator depends on the current main model:
- (i) no dependence (static or independently trained negatives)
- (ii) re-initialisation from, or insertion into, the main model's latest weights
- (iii) a novelty or diversity term relative to past negatives
- (iv) the main model's score of the generated response (margin, reward, likelihood). Recorded only; the "likelihood" design is covered elsewhere.
- (v) the main model's failure on a downstream check

**Axes:**
- (a) separate learned generator
- (b) same base
- (c) trained with RL
- (d) generates harmful responses
- (e) negatives for a generative main model
- (f) alternating or iterated against the current main model

**Already known, not counted:** APRM 2511.22888, Adv-RM 2504.06141, DuoGuard 2502.05163, ADPO 2405.12900, MACPO 2410.07672, DR-SFT 2606.30263, Token Buncher 2508.20697, SPIN 2401.01335, AntiDote 2509.08000, Safe RLHF 2310.12773, and the prompt-generator loops (MART, Self-RedTeam, MAGIC, TriPlay-RL, AdvGRPO, CHASE).

## New candidate records

### N1. Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning (Pns)
- **Citation:** Zixiang Di (East China Normal University), 2026, arXiv preprint (ICML-style template, no venue listed). arXiv 2602.03516, VERIFIED (v2, 4 Feb 2026). Method section read in HTML.
- **Mechanism:**
  - A dedicated negative generator is trained from Qwen2.5-7B-Instruct with GRPO (group 8, 200 steps).
  - Reward: a format gate, plus accuracy inversion (wrong final answers are rewarded), plus a bucketed score from a fixed Qwen3-4B RM, plus an LLM-judged CoT-quality score.
  - Its plausible-but-wrong solutions are the DPO rejected side. The chosen side is the target's own correct answers. Targets are Qwen2.5-7B/3B-Instruct and Llama3.1-8B-Instruct.
  - The generator is trained once and builds a static dataset. No reward term involves the target.
- **Dependence class:** (i).
- **Verdict:**
  - (a) yes
  - (b) yes for the Qwen2.5-7B target only
  - (c) yes (GRPO)
  - (d) responses yes, but WRONG rather than harmful
  - (e) yes, DPO negatives for a generative policy
  - (f) no
- **Closest new find.** It is the "static" half of the design done with RL instead of SFT: a separate, same-base, RL-trained response generator feeding DPO negatives to a generative model. It misses harm and alternation. Compared with APRM, it fixes APRM's gap (it has a generative main model) but loses APRM's game. Compared with ADPO, MACPO and DR-SFT, it adds the RL training they lack, but in the reasoning domain.

### N2. Adversarial RL for Hard-Negative Code Generation
- **Citation:** Eric (Jiliang) Li, 2025, project page and PDF (ericjiliangli.com/project/rl). No arXiv ID or venue found: UNVERIFIED as a publication. Read at page level only.
- **Mechanism:**
  - A Qwen2.5-Coder-0.5B generator is trained with PPO to make small edits to correct code so that unit tests fail.
  - It is rewarded for failing tests, for fooling a GAN-style discriminator, and for small edit distance. Generator and discriminator are updated in turns.
  - The resulting negatives are used afterwards as DPO preference pairs (0.420 pass@1 reported).
- **Dependence class:** (v) and (iv) relative to the DISCRIMINATOR, and (i) relative to the DPO-trained model.
- **Verdict:**
  - (a) yes
  - (b) unclear
  - (c) yes
  - (d) wrong code, not harm
  - (e) yes, but downstream and after the game
  - (f) the alternation is with the discriminator, not with the DPO model
- **What it misses:** harm, and alternation with the generative model.

### N3. SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning
- **Citation:** Jiaqi Chen, NeurIPS 2025. arXiv 2504.19162, VERIFIED. Read at abstract and NeurIPS-page level.
- **Mechanism:**
  - A "sneaky generator" and a critic are fine-tuned from the same base. The generator writes subtly erroneous reasoning steps, and the critic tries to catch them.
  - Zero-sum RL rewards (+1/-1 to the winner and loser), iterated over rounds.
- **Dependence class:** (v), through the critic's failure.
- **Verdict:**
  - (a) yes
  - (b) yes
  - (c) yes
  - (d) wrong steps, not harm
  - (e) no, the main model is a critic or verifier
  - (f) yes
- **Comparison:** the same shape as APRM, and it does not change the verdict.

### N4. Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator
- **Citation:** Shiping Yang, COLM 2026. arXiv 2607.07993, VERIFIED (v2, 7 Oct 2026). Read at abstract level.
- **Mechanism:**
  - The generator is initialised from the same base as the detector. It is trained with RLAIF, using the detector as the reward model, to write hard-to-detect hallucinated responses.
  - The detector is then re-trained with rule-based RL on that data, in a cycle.
- **Dependence class:** (iv) and (v), through the detector's score and its failure.
- **Verdict:**
  - (a) yes
  - (b) yes
  - (c) yes
  - (d) hallucinated responses, which are undesirable but not harmful
  - (e) no, the main model is a detector
  - (f) yes
- **Comparison:** APRM's game again, with responses rather than steps and hallucination rather than error. Still verifier-side.

### N5. Improving Large Language Model Safety with Contrastive Representation Learning
- **Citation:** Samuel Simko (ETH Zurich), EMNLP 2025 (aclanthology 2025.emnlp-main.1430). arXiv 2506.11938, VERIFIED. Method section 4.2 read in HTML.
- **Mechanism:**
  - A learned "attack module" is inserted in the residual stream at a random layer and trained with NLL on harmful responses. It produces harmful representations that look like benign ones.
  - These representations are hard negatives in a triplet loss that separates harmful from benign representations.
  - The module is "periodically" re-trained as the model's parameters change.
- **Dependence class:** (ii), because it is inserted into and re-fitted on the current weights.
- **Verdict:**
  - (a) a learned adversary, but a module inside the main model rather than a separate policy
  - (b) n/a
  - (c) no (NLL or gradient)
  - (d) harmful ACTIVATIONS, not responses
  - (e) yes, contrastive negatives for the generative model
  - (f) yes, periodic re-fitting
- **Closest SAFETY instance of an adaptive negative source re-fitted against the current main model.** It is in activation space and not RL, so the output-modality axis separates it from the design. Related to the CAT/LAT entries in gap_couplings_1_2.md.

### N6. Self-Steering Optimization (SSO)
- **Citation:** Self-Steering Optimization: Autonomous Preference Optimization for LLMs. arXiv 2410.17131, VERIFIED (v2, Jun 2025). No venue listed. Method read in HTML.
- **Mechanism:**
  - A data generator is optimised from the policy model with DPO/IPO+SFT losses. It produces chosen and rejected responses under good and no-principle prompts, and these then train the policy.
  - The rejected side is the no-principle response.
  - The generator "has no direct relationship with the final aligned model".
- **Dependence class:** (ii), initialised from the policy.
- **Verdict:**
  - (a) yes
  - (b) yes
  - (c) no (offline preference loss)
  - (d) responses, not harmful
  - (e) yes
  - (f) not clearly iterated
- **What it misses:** RL, harm, alternation.

### N7. Adversarial Preference Learning for Robust LLM Alignment (APL)
- **Citation:** Yuanfu Wang, ACL Findings 2025. arXiv 2505.24369, VERIFIED. Full text read (ACL PDF).
- **Mechanism:**
  - The attacker and defender are both initialised from πref (Mistral-7B-Instruct-v0.3), which is the same base.
  - The attacker generates adversarial PROMPTS and is scored by the defender's intrinsic preference probability of harmful versus safe responses.
  - Attacker and defender are updated alternately with preference losses.
- **Dependence class:** (iv), through the defender's likelihood margin.
- **Verdict:**
  - (a) yes
  - (b) yes
  - (c) preference optimisation, not online RL
  - (d) PROMPTS
  - (e) the defender gets DPO-style updates
  - (f) yes
- **What it misses:** response modality. This is a prompt-side loop like MART and should be added to that excluded group. Its (iv)-type reward is relevant to the separate likelihood design.

### N8. Prompt-side loops, newly checked (attacker emits prompts; excluded on modality)
- **AdvGame: Safety Alignment of LMs via Non-cooperative Games.** Anselm Paulus, arXiv 2512.20806, VERIFIED, HTML read.
  - The attacker rewrites queries. The defender's online DPO pairs are both its own samples, ranked by a judge.
  - The Llama attacker is Llama3.1-8B-Abliterated, so it is not the same checkpoint as the defender.
- **Anchored Bipolicy Self-Play (The Attacker in the Mirror).** Gabriele La Malfa, arXiv 2605.08427, VERIFIED, HTML read.
  - Separate attacker and defender LoRAs sit on one frozen base and are trained with REINFORCE++.
  - The attacker writes jailbreak prompts. Its reward is the negative of the defender's WildGuard score.
- **Stay in Character, Stay Safe.** Mingyang Liao, arXiv 2602.13234, VERIFIED. Training-free. The attacker writes prompts.
- **Learning to Attack and Defend.** arXiv 2606.09701. Multi-turn GRPO attacker; read at snippet level only.
- **ACE-Safety** 2511.19218, **AgentSafeEvo** 2610.19422, **Safety Self-Play** 2601.10589: attacker writes prompts or tasks, from snippets.

### N9. Static or self-generated harmful negatives, newly checked (class (i), or the main model itself)
- **PT-ALIGN.** Jingxin Xu, arXiv 2502.08657, VERIFIED, HTML read.
  - The same unaligned target, prompted with negative constraints, writes the toxic responses. A token-level unlikelihood term is applied from the first divergent token, with λ=0.4.
  - Single pass. No separate or trained generator.
- **RLCD.** arXiv 2307.12950. Principle-violating prompts on the same model.
- **NEAT.** arXiv 2410.12194. Negative prompts on the same model; already in gap_couplings_1_2.md.
- **He & Glass, Negative Training for Neural Dialogue Response Generation.** ACL 2020, arXiv 1903.02134. Malicious responses are found by trigger search on the model itself, then their likelihood is lowered.
- **DExperts / MaRCo / Ext-Sub.** Anti-experts SFT'd once and used at decoding time or in weight space.
- **"Overriding Safety Protections of Open-source Models".** arXiv 2409.19476. A harmful model trained on the rejected column, used as an attack. Read at snippet level only.
- **Nemotron 3 Nano report (2512.20848), SafeDecoding (Dolphin), and Hermes-3-70B as rejected-sample sources.** Prompted or uncensored models are used as static sources of harmful rejected responses.

### N10. Verifier-side or generator-is-main, and not relevant (checked, no change)
- **Lightweight Safety Guardrails.** 2507.08284. Guard classifier; already known.
- **RLAC.** Mian Wu, arXiv 2511.01758, VERIFIED, abstract only. The critic names failure modes for a generator; the critic does not generate negatives.
- **Generative Adversarial Reasoner.** Qihao Liu, arXiv 2512.16917 (ICLR 2026 per one listing). The reasoner is the generator and an LLM discriminator scores slices.
- **CAVGAN.** ACL Findings 2025, arXiv 2507.06043. A GAN over internal representations.
- **Colluding LoRA.** arXiv 2603.12681. An attack. It reports that alternating harmful/safe LoRA updates "often leads to optimization collapse", which is a useful caution for the design.
- **Reveal and Release.** arXiv 2509.14624. Iterative unlearning on self-generated data; no adversarial generator.
- **RULE.** arXiv 2506.07171. RL unlearning of refusal boundaries; no generator.
- **Fight Fire with Fire.** arXiv 2505.04578. Reward neutralisation; no generator.
- **IRGAN.** Jun Wang, SIGIR 2017, arXiv 1705.10513, VERIFIED. The template of a generator attacking a discriminator with hard negatives in IR, alongside ACE 1805.03642. The main model is a ranker; nothing about harm.

## Verdict

**NOT FALSIFIED.** It is partially anticipated, each time on different missing axes:

| Paper | Axes met | Misses |
|---|---|---|
| Pns 2602.03516 (new, closest) | a, b (one target), c, e | d (wrong, not harmful), f (static, class i) |
| APRM 2511.22888 / SPC 2504.19162 / Hallucination Self-Play 2607.07993 | a, b, c, f | e (main model is a verifier or detector), d (wrong or hallucinated) |
| ADPO 2405.12900 / MACPO 2410.07672 / DR-SFT 2606.30263 | b, d, e (ADPO partly a) | c (SFT), f (frozen) |
| Simko 2506.11938 (new) | e, f; harmful content | a (module, not a policy), c, d (activations, not responses) |
| SSO 2410.17131 (new) | a, b, e | c, d, f |
| N2 code hard negatives | a, c, e (downstream) | d; f is with a discriminator, not the DPO model |
| APL 2505.24369 (new) | a, b, f; the (iv) reward | d (prompts) |

No paper found meets (c), (d), (e) and (f) together: an RL-trained twin writing harmful responses that serve as negatives for a generative model, re-trained against that model's current version.

**Restated claim at the scope that survived.** The claim is specific to HARMFUL responses AND to a GENERATIVE main model AND to the twin being re-trained against the current main model. Each two-of-three version exists:
- RL plus generative main model plus static: Pns, for errors.
- RL plus alternating plus verifier: APRM, SPC, Hallucination Self-Play.
- Harm plus generative plus static or SFT: ADPO, MACPO, DR-SFT.

## Dependence classes for response-generating negative sources

- **(i) occupied.** Pns (RL), ADPO, MACPO, DR-SFT, PT-ALIGN, RLCD, NEAT, anti-experts. Any harm twin rewarded only by a judge's harmfulness belongs here, and the "alternation" then adds no game. This is the main design risk: as described, coupling 1 collapses to class (i) unless a main-model term enters the twin's reward.
- **(ii) occupied but thin.** SPIN (the previous iterate is the negative source, untrained), SSO (generator from policy weights), and Simko (activation adversary re-fitted on current weights). No RL-trained, harm-writing twin re-initialised from the main model's latest weights was found.
- **(iii) empty for response generators.** Novelty and diversity terms relative to past outputs exist only for PROMPT generators (curiosity-driven red-teaming 2402.19464, the two-player game 2406.10977, Active Attacks). None was found for a generator of harmful responses.
- **(iv) occupied only against verifiers.** APRM, Adv-RM, DuoGuard, Hallucination Self-Play, N2's discriminator. APL uses it against a generative defender, but with prompts. SGALM 2602.01137 uses it with one shared model and realism, not harm. For a harm-response twin scored by a generative main model's likelihood: none found (see the separate likelihood design).
- **(v) occupied only against verifiers (and prompt-side).** APRM, SPC, Adv-RM. Prompt loops score the target's own response. For a response twin against a generative model, the natural form is a twin-written harmful PREFIX scored by whether the main model continues harmfully (a learned prefill attacker). No RL-trained prefill or response-prefix generator co-trained with the defender was found. 2606.04168 selects prefixes from a dataset by margin, with no generator.

## Search scope

- **Covered:**
  - About 40 WebSearch queries (standard and extended) and about 20 WebFetch reads of arXiv abs and HTML pages or the ACL PDF.
  - Topics: learned or RL negative generators for DPO, unlikelihood and contrastive losses; hard-negative mining for DPO; uncensored or jailbroken models as negative sources; iterative and online safety DPO; self-play and SPIN variants for safety; evil-twin, shadow, anti-expert, negative-teacher and misaligned-reference models; negative distillation; LLM unlearning with adaptive generators; SeqGAN, IRGAN, ACE and negative-training (2017-2020); GAN-style safety (CAVGAN, GAR); adversary LoRA adapters; RL error and hallucination generators; prefill attackers; GRPO-era co-training (2025 to Oct 2026).
- **Not covered:**
  - OpenReview-only or workshop-only papers without arXiv, non-English venues, patents.
  - Full method sections for SPC, RLAC, 2606.09701, ACE-Safety and AgentSafeEvo (abstract or snippet level only; none was a plausible falsifier on modality).
  - Eric Li's PDF (page level only).
  - Citation crawl, partly done:
    - **ADPO's 9 citers (Semantic Scholar):** none is a falsifier. TuneShield/Optimus 2507.05660 uses classifier-flagged toxicity plus healing data and DPO, with no generator. "DPO vs A-DPO" is a Stanford CS224R course report, not a publication. AAPA 2509.25148 uses a fixed discriminator and no generator.
    - **DR-SFT:** no citers listed yet.
    - **Pns:** the citation query was rate-limited (HTTP 429), so it was not crawled. Forward citations of Pns and of APRM remain the most likely place for a missed hit.

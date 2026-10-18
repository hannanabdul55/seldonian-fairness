# Learned attacker policies for red-teaming language models (2022 - Oct 2026)

Scope: text/LLM attackers where a separate (or role-shared) language model is trained or adapted to make a target LLM produce harmful / constraint-violating outputs. Pure GCG-style token attacks are excluded unless a generator model is trained. All arXiv IDs below were checked against the arXiv API (export.arxiv.org) or the arXiv abstract page on 2026-10-17 unless marked UNVERIFIED. Items first posted in 2025-2026 are tagged **[RECENT]**. "Alternating" = the paper also updates the target/defender; "attack-only" = target frozen throughout.

Legend for structured fields: (a) attacker training; (b) same base as target?; (c) frozen vs moving target; (d) defender updated?; (e) ASR / diversity reported; (f) failure modes; (g) code.

---

## Key Question 1: Which papers train an attacker LM (RL or otherwise) against a target LM? (catalogue)

### Takeaway
There are three families: (i) single-agent RL attackers against a frozen target (Perez 2022 → CRT 2024 → GFlowNet 2024 → Beutel/OpenAI 2024 → GRPO/reasoning attackers 2025-26), whose central problem is mode collapse and judge gaming; (ii) generator models distilled from or trained via target log-probabilities (AdvPrompter, AmpleGCG, GASP); and (iii) attacker-defender co-training (MART 2023, Red Teaming Game 2023, DART/APRT 2024, Self-RedTeam 2025, Active Attacks 2025, MAGIC / Safety Self-Play / Anchored Bipolicy / AdvGRPO / GPT-Red 2026), which is now the dominant 2026 direction and the one most relevant to the final report.

### Cited Findings (paper records)

#### 1. Red Teaming Language Models with Language Models — Perez, 2022, arXiv:2202.03286 (posted 2022-02-07) — [arXiv](https://arxiv.org/abs/2202.03286)
Summary: A "red LM" generates test questions; the target (Dialogue-Prompted Gopher) answers; a classifier flags offensive replies. Four generation regimes: zero-shot, stochastic few-shot, supervised learning on failing cases, and RL. RL is the most effective and the least diverse. — [arXiv HTML](https://arxiv.org/html/2202.03286)
- (a) RL: synchronous A2C warm-started from the SL model; reward −log p(1−r(x,y)) from a 1.4B Gopher classifier fine-tuned on Bot-Adversarial Dialogue, reward −3 if the output lacks "?"; KL penalty weight α ∈ {0.3, 0.35, 0.4}. — [arXiv HTML](https://arxiv.org/html/2202.03286)
- (b) **Same base**: red LM and target are both the 280B Gopher model, differing only by prompt. — [arXiv HTML](https://arxiv.org/html/2202.03286)
- (c) Frozen target. (d) Attack-only.
- (e) Zero-shot: 3.7% offensive replies (18,444 failing cases from 0.5M); RL at α=0.3: >40% offensive. Diversity by a Self-BLEU-style max-BLEU against 1,000 samples (per-method values only in Figure 2). — [arXiv HTML](https://arxiv.org/html/2202.03286)
- (f) Mode collapse: at α=0.3, 78% of RL test cases contain the word "invisible"; the paper frames lower KL as "sacrificing diversity for expected reward"; classifier false positives/negatives noted. — [arXiv HTML](https://arxiv.org/html/2202.03286)
- (g) Code release not stated in the text read (Gap).

#### 2. Explore, Establish, Exploit: Red Teaming Language Models from Scratch — Casper, 2023, arXiv:2306.09442 (v3 2023-10-11; no venue listed) — [arXiv](https://arxiv.org/abs/2306.09442)
Summary: Red-teaming without a pre-existing classifier: explore target outputs, establish a classifier from human labels (CommonClaim, 20k statements), then exploit it with an RL prompt generator. — [arXiv](https://arxiv.org/abs/2306.09442)
- (a) PPO via trlx; reward = logit confidence of an ensemble of RoBERTa classifiers trained on human labels + an intra-batch cosine-distance diversity term on the target's embeddings of the prompts. — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)
- (b) Different model: attacker GPT-2-large; targets GPT-2-xl (toxicity) and GPT-3 text-davinci-002 (false statements). — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)
- (c) Frozen. (d) Attack-only.
- (e) Toxic outputs <1% unprompted → 31% under adversarial prompts; CommonClaim-false 30% → 74%. — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)
- (f) Classifier exploitation (CREAK- and ChatGPT-label classifiers hacked by toxic/nonsense text that was not false to humans); unnatural disfluent prompts; without the diversity term prompts collapsed to "would/you" repetitions with 0% toxic rate and 61/100 identical prompts on GPT-3. — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)
- (g) https://github.com/thestephencasper/explore_establish_exploit_llms — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)

#### 3. FLIRT: Feedback Loop In-context Red Teaming — Mehrabi, 2023, arXiv:2308.04265, EMNLP 2024 — [arXiv](https://arxiv.org/abs/2308.04265)
Summary: Red LM is adapted purely by in-context exemplar updating in a feedback loop (no weight updates); primary targets are text-to-image models, with a text-to-text experiment also reported. — [arXiv](https://arxiv.org/abs/2308.04265)
- (a) No RL/SFT: in-context feedback (exemplar replacement). (b) n/a (prompted). (c) Frozen. (d) Attack-only. (e) Not on abstract page (Gap). (f) Not discussed on abstract page. (g) Not stated.
- Included because JAB (#7) reuses FLIRT as the attacker and because it is the "no-training" baseline that later RL papers compare against.

#### 4. Attack Prompt Generation for Red Teaming and Defending Large Language Models — Deng, 2023, arXiv:2310.12505, EMNLP 2023 Findings — [arXiv](https://arxiv.org/abs/2310.12505)
Summary: Attack framework uses in-context learning to make an LLM mimic human-written attack prompts (SAP datasets); defense framework fine-tunes the victim "through iterative interactions with the attack framework". — [arXiv](https://arxiv.org/abs/2310.12505)
- (a) Attacker: ICL only (no RL/SFT). (b) n/a. (c) Attacker is re-run against the updated target each iteration (moving target in the loop). (d) **Alternating**: victim fine-tuned iteratively. (e) Numbers not on abstract page (Gap). (g) https://github.com/Aatrox103/SAP — [arXiv](https://arxiv.org/abs/2310.12505)

#### 5. Evolving Diverse Red-team Language Models in Multi-round Multi-agent Games (Red Teaming Game / GRTS) — Ma, 2023, arXiv:2310.00322 (v5 2024-07-28; venue UNVERIFIED) — [arXiv](https://arxiv.org/abs/2310.00322)
Summary: Formalises red-teaming as a zero-sum extensive-form game over multi-round dialogue; solved with GRTS, a Double-Oracle/PSRO-style population solver where each side trains a PPO best response against the other's meta-strategy, with a diversity term. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (a) PPO best responses; red reward = toxicity score (Bradley-Terry model trained on PKU-SafeRLHF) + semantic-space diversity term; blue reward = −red reward. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (b) **Same base**: red, blue, critic and toxicity model all built on stablelm-alpaca-3b; red first SFT'd on Anthropic-HH, PKU-SafeRLHF and BAD. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (c) Moving target (population). (d) **Alternating**: red best response with blue fixed, then blue best response with red fixed; populations grow each iteration. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (e) Exploitability ≈6.2 → ≈0.8 over ~15 iterations; trained blue cut ASR ≈52% → ≈1% on fixed prompts; equilibrium ASR ≈10%; trained red vs untrained blue ASR ≈40% → ≈70%; GRTS-12 (3B red) round-3 ASR 0.56 on Zephyr-7B and 0.13 on Llama-2-70B vs 0.31/0.00 for SFT. Diversity = average pairwise n-gram similarity scaled to [0,1]. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (f) Motivated by mode collapse and Goodhart/reward hacking in single-agent red teaming; appendix titles mention excessive exploration → OOD reward problems and excessive exploitation → mode collapse; toxicity reward invalid OOD; exploitability not driven to 0. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- (g) Not stated in text read (Gap).

#### 6. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming — Ge, 2023, arXiv:2311.07689, NAACL 2024 (2024.naacl-long.107) — [arXiv](https://arxiv.org/abs/2311.07689); [ACL Anthology](https://aclanthology.org/2024.naacl-long.107)
Summary: An adversarial LLM and a target LLM are trained in alternation for four rounds: the adversarial model learns (by supervised pairwise fine-tuning on successful parent→child attacks) to produce prompts that fail the target; the target is SFT'd on its own safe-and-helpful responses to those prompts. — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (a) Attacker: **supervised fine-tuning, not RL** (successful attack paired with its parent prompt), mixed with instruction data; warmed up on ~2,400 curated red-team prompts. — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (b) **Same base**: both initialised from LLaMA-65B fine-tuned by the authors on Open Assistant + LIMA (harmful items removed). — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (c) Moving target. (d) **Alternating**, 4 rounds; context distillation in round 1, rejection sampling in round 4; judges = Llama 2 safety RM and helpfulness RM (unsafe if safety RM < 0.5). — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (e) SafeEval violation rate (RM) 31.4% → 4.8% (−84.7% rel.); human 17.2% → 8.0%; Anthropic Harmless 26.7% → 6.9%; helpfulness 3-4% lower in-distribution, AlpacaEval "barely changes". Attack violation rates by round: 29.74%, 31.89%, 20.33%, 10.21% (stopped at round 4 for lack of training data). — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (f) No explicit diversity metric tracked across rounds (3-shot variant less diverse than 1-shot); risk of over-conservatism; RM safety scores could be gamed (checked by human eval); GCG attacks lost efficacy after one round. — [arXiv HTML](https://arxiv.org/html/2311.07689)
- (g) Not stated (Gap).

#### 7. JAB: Joint Adversarial Prompting and Belief Augmentation — Mehrabi, 2023, arXiv:2311.09473 (no venue) — [arXiv](https://arxiv.org/abs/2311.09473)
Summary: A FLIRT-style in-context adversary and an in-context "belief augmenter" (defence instruction generator) are updated jointly for 100 iterations against a black-box target; no weights are trained. — [arXiv HTML](https://arxiv.org/html/2311.09473)
- (a) No RL/SFT: in-context exemplar replacement if score higher. (b) Red, target and belief generator are each the same model (GPT-Neo 2.7B or Falcon-Instruct 7B) in separate settings. (c) Moving defence (belief prepended). (d) Defence updated, but **target weights unchanged**. — [arXiv HTML](https://arxiv.org/html/2311.09473)
- (e) Dynamic setting toxicity (Perspective API): GPT-Neo 46.2% (no belief) → 0.1% (fully jabbed); Falcon 15.1% → 1.6%; static (RealToxicityPrompts) gains small (3.5% → 2.0%; 3.8% → 2.7%). — [arXiv HTML](https://arxiv.org/html/2311.09473)
- (f) Not discussed in text read. (g) None found.

#### 8. Curiosity-driven Red-teaming for Large Language Models (CRT) — Hong, 2024, arXiv:2402.19464, ICLR 2024 — [arXiv](https://arxiv.org/abs/2402.19464)
Summary: Adds curiosity/novelty rewards to PPO red-teaming so the attacker keeps generating new prompts instead of collapsing on a few. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- (a) PPO (trlx), KL β=0.001; reward = RoBERTa hate-speech classifier (ToxiGen-trained for LLaMA-2) + negative SelfBLEU novelty (λ=1.0) + cosine-embedding novelty (λ=1.0) + entropy bonus (λ=0.01) + gibberish penalty. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- (b) Different model: attacker GPT-2 (137M); targets GPT-2-IMDb, GPT2-alpaca, Dolly-v2-7B, LLaMA2-7b-chat. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- (c) Frozen. (d) Attack-only.
- (e) 196 prompts eliciting toxic replies from LLaMA2-7b-chat (official LLaMA2 reports 0% on the Hartvigsen prompts); 16 humans writing 100 cases found none above 0.5 toxicity; diversity values only in figures. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- (f) RL baseline "converges to a deterministic policy"; entropy bonus alone does not improve SelfBLEU; novelty weights can dominate the objective; CRT exploits Dolly by generating incomplete sentences. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- (g) https://github.com/Improbable-AI/curiosity_redteam — [arXiv HTML](https://arxiv.org/html/2402.19464)

#### 9. Gradient-Based Language Model Red Teaming (GBRT) — Wichers, 2024, arXiv:2401.16656, EACL 2024 — [arXiv](https://arxiv.org/abs/2401.16656)
Summary: Backpropagates a safety classifier's score through the frozen classifier and target LM into the prompt; variants add a realism loss or fine-tune a pretrained LM to generate the prompts. Claims better than a strong RL red-teaming baseline, including against a safer-tuned target. — [arXiv](https://arxiv.org/abs/2401.16656)
- (a) Gradient; one variant trains a generator LM. (b) Not stated on abstract page (Gap). (c) Frozen. (d) Attack-only (a safer-tuned target is evaluated, not trained in the loop). (e) No numbers on abstract page (Gap). (f) Incoherent prompts motivate the realism loss. (g) Not stated.

#### 10. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts — Samvelyan, 2024, arXiv:2402.16822 (v3 2024-12-11; NeurIPS 2024 UNVERIFIED here) — [arXiv](https://arxiv.org/abs/2402.16822)
Summary: Casts prompt generation as quality-diversity open-ended search with a mutator LLM and a judge; no attacker weights are trained. ASR >90% on all tested Llama 2/3 models; fine-tuning the target on the archive improves safety without hurting helpfulness. — [arXiv](https://arxiv.org/abs/2402.16822)
- (a) Evolutionary (QD), not trained. (b) n/a. (c) Frozen during search. (d) Target fine-tuned once afterwards (not alternating). (e) ASR >90%; diversity numbers not on abstract page. (f) Not discussed. (g) Not stated.

#### 11. AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs — Paulus, 2024, arXiv:2404.16873, ICML 2025 — [arXiv](https://arxiv.org/abs/2404.16873)
Summary: Trains a separate LLM to emit human-readable adversarial suffixes via an alternating optimisation (candidate suffix search, then regression of the generator onto them); no gradients from the target are needed at generation time; transfers to black-box models; suffix data used for adversarial training of the target. — [arXiv](https://arxiv.org/abs/2404.16873)
- (a) Alternating optimisation / SFT-style regression (the abstract page does not name AdvPrompterOpt/Train; objective details are a Gap from this read). (b) Not stated on abstract page (Gap). (c) Frozen. (d) Attack-first; adversarial training of the target reported as a "promising strategy". (e) "Highly competitive" on AdvBench/HarmBench; numbers not on page (Gap). (g) https://github.com/facebookresearch/advprompter — [arXiv](https://arxiv.org/abs/2404.16873)

#### 12. AmpleGCG — Liao, 2024, arXiv:2404.07921, COLM 2024 — [arXiv](https://arxiv.org/abs/2404.07921)
Summary: Trains a generative model of adversarial suffixes on successful intermediate GCG suffixes; near-100% ASR on Llama-2-7B-chat and Vicuna-7B, 99% on GPT-3.5, 200 suffixes in ~4 s. — [arXiv](https://arxiv.org/abs/2404.07921)
- (a) Supervised generator trained on GCG-mined data. (b) Not stated on page. (c) Frozen. (d) Attack-only. (e) As above. (g) Not on abstract page.

#### 13. Learning diverse attacks on LLMs for robust red-teaming and safety tuning — Lee, 2024, arXiv:2405.18540, ICLR 2025 — [arXiv](https://arxiv.org/abs/2405.18540)
Summary: GFlowNet fine-tuning (trajectory balance) samples prompts proportionally to a reward combining target-response toxicity and reference-LM likelihood, followed by MLE re-fitting on filtered high-reward prompts; produces diverse, transferable attacks and the most robust safety-tuned target. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- (a) GFlowNet + MLE smoothing; R1 = exp((1/β)·E log p(toxic|x,y)), R2 = p_ref(x)^(1/γ); β swept 0.01-1.0. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- (b) Different model: attackers GPT-2-small and Llama-3.2-1B; targets GPT-2, Dolly-v2-7b, Gemma-2b-it, Llama-2-7b-chat, Llama-3.1-8B-Instruct. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- (c) Frozen. (d) Target safety-tuned once after attack collection (LoRA on Gemma-2b-it), not alternating.
- (e) Llama-3.1-8B-Instruct: 81.05% toxic (GPT-2 attacker, cos-dist 0.829), 92.71% (Llama-3.2-1B attacker); transfer from Gemma-2b-it attacker: 27.39% Llama-2-7b-chat, 52.01% Llama-3-70b, 67.21% Starling-7b; 65% harmful on GPT-4o. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- (f) PPO+Novelty produced 0 prompts above 0.2 toxicity on Gemma/Llama-2; REINFORCE collapses to a single prompt; SFT/ICL diverse but ineffective; GFlowNet alone sensitive to β. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- (g) https://github.com/GFNOrg/red-teaming — [arXiv API comment](https://arxiv.org/abs/2405.18540)

#### 14. DiveR-CT — Zhao, 2024, arXiv:2405.19026, AAAI-25 — [arXiv](https://arxiv.org/abs/2405.19026)
Summary: Constrained RL for red-teaming: safety-classifier score and gibberish are treated as constraints (Lagrange multipliers adjusted dynamically), while semantic (k-NN cosine) and lexical (1−BLEU) diversity are maximised; argues CRT's average-cosine novelty "stagnates" as history grows. — [arXiv HTML](https://arxiv.org/html/2405.19026)
- (a) PPO with gradient descent-ascent on a Lagrangian. (b) Attacker base not in text read (Gap); targets gpt2-alpaca-gpt4, Llama-2-7b-chat, Llama-3-8B-Instruct. (c) Frozen. (d) Target safety-tuned once afterwards (2:1 Alpaca mix).
- (e) Matched-ASR buckets vs CRT: 0.869 vs 0.868, 0.686 vs 0.692, 0.485 vs 0.444, with higher diversity in all buckets; vs Llama-Guard-2 classifier 0.864 vs 0.859. — [arXiv HTML](https://arxiv.org/html/2405.19026)
- (f) Reward over-optimisation: RL and CRT ASR drop sharply on a held-out test classifier; CRT PCA shows runs extending in one direction (outlier exploitation). — [arXiv HTML](https://arxiv.org/html/2405.19026)
- (g) https://github.com/LeapLabTHU/diver-ct — [arXiv HTML](https://arxiv.org/html/2405.19026)

#### 15. RLbreaker: When LLM Meets DRL — Chen, 2024, arXiv:2406.08705, NeurIPS 2024 — [arXiv](https://arxiv.org/abs/2406.08705) (sibling: RL-JACK, arXiv:2406.08725)
Summary: A DRL agent (customised PPO) guides the search over jailbreak prompt mutations, replacing genetic-algorithm randomness; claims to beat existing attacks on six LLMs and transfer across targets. — [arXiv](https://arxiv.org/abs/2406.08705)
- (a) PPO over mutation actions (policy is a small controller, not a free-form prompt generator). (b) Different. (c) Frozen. (d) Attack-only. (e) Numbers not on abstract page (Gap). (g) Not on page.

#### 16. DART / APRT: Automated Progressive Red Teaming — Jiang, 2024, arXiv:2407.03876 (v1 "DART: Deep Adversarial Automated Red Teaming for LLM Safety"; v3 "Automated Progressive Red Teaming", COLING 2025) — [arXiv v3](https://arxiv.org/abs/2407.03876v3); [ar5iv v1](https://ar5iv.labs.arxiv.org/html/2407.03876)
Summary: v1 (DART) iterates a Red LLM and a Target LLM: the red model adjusts attack directions by monitoring global diversity across iterations, the target is strengthened via active-learning data selection; human eval reports 53.4% fewer violations on Anthropic Harmless. v3 (APRT) re-frames the method as three modules (Intention Expanding LLM, Intention Hiding LLM, Evil Maker) over several rounds and proposes an Attack Effectiveness Rate metric. — [DART search summary](https://ar5iv.labs.arxiv.org/html/2407.03876); [APRT](https://arxiv.org/abs/2407.03876v3)
- (a) Iterative fine-tuning of the red modules (exact algorithm per version is a Gap from this read). (b) Not confirmed. (c) Moving target in DART. (d) **Alternating** in DART (target updated); APRT framing is attack-focused. (e) APRT: unsafe-yet-useful responses 54% Llama-3-8B-Instruct, 50% GPT-4o, 39% Claude-3.5. (g) https://github.com/tjunlp-lab/APRT — [search result](https://arxiv.org/abs/2407.03876v3)
- Note: the two versions differ substantially; cite the version you mean.

#### 17. Ferret — Pala, 2024, arXiv:2408.10701 — [arXiv](https://arxiv.org/abs/2408.10701)
Summary: Builds on Rainbow Teaming; generates several mutations per iteration and keeps the best by a scorer (reward model best, also Llama Guard / LLM-judge); 95% overall ASR (46% above Rainbow Teaming), 15.2% less time to reach 90% ASR. No attacker training stated. — [arXiv](https://arxiv.org/abs/2408.10701)
- (a) Evolutionary + reward scoring. (c) Frozen. (d) Attack-only. (g) https://github.com/declare-lab/ferret

#### 18. GASP — Basani, 2024, arXiv:2411.14133, NeurIPS 2025 — [arXiv](https://arxiv.org/abs/2411.14133)
Summary: Black-box suffix generator trained with latent Bayesian optimisation in a continuous embedding space plus iterative refinement for readability. — [arXiv](https://arxiv.org/abs/2411.14133)
- (a) Generator trained by latent BO (ORPO step UNVERIFIED from abstract page). (c) Frozen. (d) Attack-only. (e) Numbers not on page (Gap). (g) Project page https://air-ml.org/project/gasp/.

#### 19. Diverse and Effective Red Teaming with Auto-generated Rewards and Multi-step RL — Beutel (OpenAI), 2024, arXiv:2412.18693 (no venue) — [arXiv](https://arxiv.org/abs/2412.18693)
Summary: Decomposes red-teaming into (1) LLM-generated diverse goals with per-goal rule-based rewards (RBRs) and (2) a multi-step RL attacker conditioned on its past attempts and rewarded for being different from them; applied to indirect prompt injection and safety jailbreaks against GPT-4 Turbo. — [arXiv HTML](https://arxiv.org/html/2412.18693)
- (a) RL (algorithm not named; γ=0; no KL reported); reward = product of RBR (P(yes) from an LLM judge), Moderation API (jailbreaks: 0.5 RBR + 0.5 Moderation), few-shot similarity (ada-002 cosine through sigmoid), style-diversity via goal-subspace projection (1 − max similarity to past attempts), length penalty; T=5 training steps, T=10 eval. — [arXiv HTML](https://arxiv.org/html/2412.18693)
- (b) Different model: attacker is a GPT-3.5-sized model "trained for instruction following without safety guardrails"; target GPT-4T. — [arXiv HTML](https://arxiv.org/html/2412.18693)
- (c) Frozen. (d) Attack-only.
- (e) Jailbreaks (ASR / cosine diversity): vanilla RL (Perez) 0.96 / 0.02; CRT 0.69±0.42 / 0.08; RBR+few-shot 0.89 / 0.11; multi-step RL 0.86 / 0.15; +style subspace 0.79 / 0.14 with style diversity 0.48. Indirect injection ASR: one-shot 0.02 → multi-step 0.42. — [arXiv HTML](https://arxiv.org/html/2412.18693)
- (f) Diversity collapse of vanilla RL; narrow tactics; **judge gaming**: under the combined reward the attacker optimises the Moderation API and largely ignores the RBR; padding inflates diversity (hence length penalty); grader exploitation called an open problem; large run variance. — [arXiv HTML](https://arxiv.org/html/2412.18693)
- (g) Reward-generation code archive on cdn.openai.com (footnote); jailbreak code "upon publication". — [arXiv HTML](https://arxiv.org/html/2412.18693)

#### 20. **[RECENT]** Auto-RT — Liu, 2025, arXiv:2501.01830 — [arXiv](https://arxiv.org/abs/2501.01830)
Summary: RL framework that explores multi-step attack strategies with Early-terminated Exploration and Progressive Reward Tracking using intermediate "downgrade" (weaker) target models as reward stepping stones; 16.63% higher success than baselines. — [arXiv](https://arxiv.org/abs/2501.01830)
- (a) RL (algorithm not named on page); reward shaped by weakened targets. (b)/(e) absolute numbers and models not on page (Gap). (c) Frozen (with weakened proxies). (d) Attack-only. (g) None found.

#### 21. **[RECENT]** xJailbreak — Lee, 2025, arXiv:2501.16727 — [arXiv](https://arxiv.org/abs/2501.16727)
Summary: Black-box RL prompt rewriter guided by representation-space proximity between benign and malicious prompts; targets Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, GPT-4o. — [arXiv](https://arxiv.org/abs/2501.16727)
- (a) RL (algorithm not on page). (c) Frozen. (d) Attack-only. (e) "SOTA", numbers not on page (Gap). (g) https://github.com/Aegis1863/xJailbreak

#### 22. **[RECENT]** AutoRedTeamer — Zhou, 2025, arXiv:2503.15754 — [arXiv](https://arxiv.org/abs/2503.15754)
Summary: Agentic (not weight-trained) red-teaming: a red-team agent plus a strategy-proposer agent that mines new attacks from recent papers, with memory-guided attack selection; ~20% higher ASR on HarmBench vs Llama-3.1-70B and 46% lower cost. — [arXiv](https://arxiv.org/abs/2503.15754)
- (a) No RL/SFT reported. (c) Frozen. (d) Attack-only. (g) Not on page. Included as a boundary case (agentic, not learned).

#### 23. **[RECENT]** X-Teaming — Rahman, 2025, arXiv:2504.13203 — [arXiv](https://arxiv.org/abs/2504.13203)
Summary: Multi-turn jailbreaks via collaborative planner/attacker/verifier agents with prompt optimisation; ASR up to 98.1%, 96.2% on Claude 3.7 Sonnet; releases XGuard-Train (30K multi-turn jailbreaks) for defender training. — [arXiv](https://arxiv.org/abs/2504.13203)
- (a) Prompted agents (TextGrad-style optimisation UNVERIFIED from abstract page); not RL-trained. (d) Defender data released; fine-tuning results not on page. Included as boundary case and because MAGIC/Self-RedTeam use it as a dynamic evaluation attack.

#### 24. **[RECENT]** Jailbreak-R1 — Guo, 2025, arXiv:2506.00782 — [arXiv](https://arxiv.org/abs/2506.00782)
Summary: Three-stage RL red-teamer: cold-start imitation on a jailbreak dataset, warm-up exploration rewarded for diversity and consistency, then "progressive jailbreak rewards" to raise effectiveness. — [arXiv](https://arxiv.org/abs/2506.00782)
- (a) SFT cold start + RL (algorithm not on abstract page; HF model card lists a 7.6B model built on Qwen2.5-7B-Instruct). — [HF card](https://huggingface.co/yukiyounai/Jailbreak-R1/blob/main/README.md)
- (b) Attacker Qwen2.5-7B-Instruct; targets "a variety of LLMs" (same-base not claimed). (c) Frozen. (d) Attack-only. (e) Numbers not on page (Gap). (g) github.com/yuki-younai/Jailbreak-R1 per HF card (UNVERIFIED directly).

#### 25. **[RECENT]** Quality-Diversity Red-Teaming (QDRT) — Wang, 2025, arXiv:2506.07121 (v2 2026-08-01) — [arXiv](https://arxiv.org/abs/2506.07121)
Summary: Trains a population of specialised attackers (per attack style / risk category) with behaviour-conditioned training and an open-ended behavioural replay buffer; targets GPT-2, Llama-3, Gemma-2, Qwen2.5, GPT-4.1, GPT-5-Chat. — [arXiv](https://arxiv.org/abs/2506.07121)
- (a) QD + RL (algorithm not on page). (c) Frozen. (d) Attack-only. (e) Numbers not on page (Gap). (g) https://github.com/lamda-bbo/QDRT

#### 26. **[RECENT]** Chasing Moving Targets with Online Self-Play RL (Self-RedTeam) — Liu, 2025, arXiv:2506.07468, ICML 2026 Poster (an ICML 2025 virtual page also exists; API comment says ICML 2026) — [arXiv](https://arxiv.org/abs/2506.07468); [ICML 2026](https://icml.cc/virtual/2026/poster/61969)
Summary: One shared policy alternates attacker and defender roles in a two-player zero-sum game with hidden chain-of-thought; the attacker moves first each round; a Nash-equilibrium defender is argued to be safe against any attack in the game. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- (a) REINFORCE++ (critic-free), role-specific normalised advantages, token-level KL to reference, clipped objective; judge WildGuard-7B (query harm, response harm, refusal) with Qwen3Guard also tested; attacker reward = −defender game reward + format/revision-faithfulness shaping; no explicit diversity bonus. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- (b) **Same base and same weights**: one πθ plays both roles via role instructions; Qwen2.5-3B/7B/14B-IT, Llama3.1-8B-IT and a safety-removed Llama3.1-8B-IT-AB. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- (c) Moving target. (d) **Alternating / self-play** (co-evolving roles each round).
- (e) +17.80% SBERT attack diversity vs attacker-only; safety up to 95% across 14 benchmarks; Qwen2.5-7B WildGuardTest adversarial ASR 0.303 → 0.179, HarmBench 0.864 → 0.489; average relative ASR reduction 36.43% over 11 benchmarks; strategy count 4.6 vs 1.9 tactics; attack perplexity 8.62 → 24.74. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- (f) Attacker-only (fixed defender) collapses into few clusters and diversity degrades, revision faithfulness ~60%; defender-only on static attacks overfits; reward hacking addressed by cross-judge training; trivial always-refuse defender countered with benign seeds; LLM-LAT baseline over-refuses (XSTest compliance 0.924 → 0.004). — [arXiv HTML](https://arxiv.org/html/2506.07468)
- (g) https://github.com/mickelliu/selfplay-redteaming

#### 27. **[RECENT]** Active Attacks: Red-teaming LLMs via Adaptive Environments — Yun, 2025, arXiv:2509.21947, ICML 2026 poster — [arXiv](https://arxiv.org/abs/2509.21947); [ICML 2026](https://icml.cc/virtual/2026/poster/64976)
Summary: Plug-in for RL red-teamers (GFlowNet, PPO+Novelty, REINFORCE): the victim is periodically safety-fine-tuned on the attacks collected so far, which removes reward from exploited regions and forces the attacker to new modes (easy-to-hard curriculum). — [arXiv HTML](https://arxiv.org/html/2509.21947)
- (a) GFlowNet (trajectory balance, β=0.1, 5000 steps) / PPO+Novelty / REINFORCE; reward = Meta-Llama-Guard-2-8B toxicity probability over k=5 victim responses; attacker reset to reference and buffer cleared after each victim update. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- (b) Different model: attacker Qwen2.5-1.5B (base, 200 SFT steps); victims Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, gemma-3-4b-it, Mistral-7B-Instruct-v0.3. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- (c) **Moving target**: victim LoRA-tuned on prompt-refusal pairs every R=1000 of T=5000 steps (paper inconsistent on 4 vs 5 rounds). (d) **Alternating** (victim updated inside the loop).
- (e) Average defense rate after final safety tuning: GFlowNet 91.00 → GFlowNet+Active Attacks 99.71; cross-attack success 0.07% vs 31.28% ("more than 400×"); HarmAug-Guard ablation toxicity 82.03% → 87.35%, cosine distance 0.47 → 0.53; +5.76% compute. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- (f) Baseline mode collapse (REINFORCE repeats one network-access prompt; GFlowNet concentrates on home-address prompts); ~30% of Active-Attacks prompts still bypass a GFlowNet-tuned victim; some prompts only weakly harmful; toxicity and diversity both depend on Llama-Guard categories. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- (g) https://github.com/dbsxodud-11/active_attacks

#### 28. **[RECENT]** DialTree (DialTree-RPO): Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks — Guo, 2025, arXiv:2510.02286, ICLR 2026 — [arXiv](https://arxiv.org/abs/2510.02286)
Summary: On-policy RL + tree search over multi-turn dialogues with pruning, a safety-guardrail reward and adaptive masking of format tokens to prevent "format unlearning"; v2 reports 81.5% average ASR over 12 targets (+44.2% over prior SOTA) with an attacker trained only against Llama-3.2-1B-Instruct. — [search summary](https://arxiv.org/html/2510.02286v1); [arXiv](https://arxiv.org/abs/2510.02286)
- (a) On-policy RL (algorithm not named on abstract page). (b) Different (trained vs a 1B target, transfers to Claude-4-Sonnet per v2 summary). (c) Frozen. (d) Attack-only. (f) Format unlearning in negative-advantage trajectories. (g) Not on page (Gap).

#### 29. **[RECENT]** RL-Hammer: "RL Is a Hammer and LLMs Are Nails" — Wen, 2025, arXiv:2510.04885 — [arXiv](https://arxiv.org/abs/2510.04885)
Summary: GRPO-trained prompter for prompt injection using only black-box rewards from the target, trained from scratch without warm-up data; 98% ASR vs GPT-4o and 72% vs GPT-5 with Instruction Hierarchy. — [search summary](https://arxiv.org/html/2510.04885v1)
- (a) GRPO; tricks include removing the KL term; trains jointly against an undefended and a defended target to reduce reward sparsity (per PISmith's description). (b) Different. (c) Frozen. (d) Attack-only. (f) Authors note attackers reward-hack diversity objectives; sparse rewards against defended targets. (g) github.com/facebookresearch/rl-injector (per search summary; UNVERIFIED directly).

#### 30. **[RECENT]** Be Your Own Red Teamer: Safety Self-Play (SSP) — Wang, 2026, arXiv:2601.10589 (2026-01-15) — [arXiv](https://arxiv.org/abs/2601.10589)
Summary: A single LLM plays attacker then defender in one RL loop; an LLM judge scores 1-5, zero-sum rewards r_att = clip((s−1)/4), r_def = 1 − r_att; low-reward cases are stored in two experience pools and replayed by UCB priority. — [arXiv HTML](https://arxiv.org/html/2601.10589)
- (a) RL (algorithm not named); judge-based reward; UCB replay (c=√2, τ=0.5, 3 iterations, lr 1e-6, 5,000 goals from Jailbreak-R1). (b) **Same weights**: Qwen2.5-7B-Instruct plays both roles ("w/o UM" ablation splits roles). (c) Moving. (d) **Self-play** (both roles updated).
- (e) Defender ASR on Qwen2.5-7B: 1.7/2.4/1.4/1.3/2.1/3.0 for GCG/PAIR/TAP/DAN/DI/SAA; attacker HarmBench ASR/DIV: Vicuna-7B 83.5/0.957, GPT-4o 59.0/0.971, Claude-3.5 31.5/0.972; OR-Bench refusal rate 25.3%. — [arXiv HTML](https://arxiv.org/html/2601.10589)
- (f) Diversity bounded by seed goals and base-model creativity; several internal table inconsistencies noted by the fetch (e.g. Vicuna SAA 5.1 vs 10.1; DIV reported as higher-is-better self-BLEU). (g) None found.

#### 31. **[RECENT]** RvB: Automating AI System Hardening via Iterative Red-Blue Games — Huang, 2026, arXiv:2601.19726 — [arXiv](https://arxiv.org/abs/2601.19726)
Summary: Training-free sequential imperfect-information game between prompted red and blue agents for code hardening (CVEs) and guardrail optimisation; defense success 90% / 45% with near-0% FPR. Boundary case (no learned attacker). — [arXiv](https://arxiv.org/abs/2601.19726)

#### 32. **[RECENT]** MAGIC: A Co-Evolving Attacker-Defender Adversarial Game — Wen, 2026, arXiv:2602.01539 — [arXiv](https://arxiv.org/abs/2602.01539)
Summary: Multi-turn multi-agent RL with separate attacker and defender policies trained in alternating GRPO phases; attacker rewrites queries into deceptive prompts, defender learns to refuse without over-refusing; argues shared parameters (Self-RedTeam) cause gradient conflicts. — [arXiv HTML](https://arxiv.org/html/2602.01539)
- (a) GRPO with KL; judge Qwen3Guard ("controversial" = harmful); R_D = R_harm + R_ref (penalises over-refusal and unsafe compliance), R_A = −R_D + R_fmt; attacker cold-started by SFT on CoT-enriched SorryBench (20 rewriting strategies). — [arXiv HTML](https://arxiv.org/html/2602.01539)
- (b) **Separate policies**, same model family (Qwen2.5-7B/14B, Llama3.1-8B instruct); whether attacker and defender share the identical checkpoint per run is not clearly stated. (c) Moving. (d) **Alternating**: K rounds of T_D defender steps (attacker frozen) then T_A attacker steps (defender frozen).
- (e) Qwen2.5-7B WildGuardTest adversarial ASR 36.5% → 2.3% (Self-RedTeam 25.5%); HarmBench defender: GCG 43.90 → 11.25, PAIR 44.38 → 25.31, TAP 63.75 → 35.63, AutoDAN-Turbo 75.00 → 54.69; X-Teaming multi-turn ASR 94.9 → 85.4; AlpacaEval 2 33.73 → 33.22 (7B) but 14B drops to 30.95 vs Self-RedTeam 42.48. **No diversity metric reported** (paper argues diversity is not a reliable proxy; uses a defender-performance heatmap). — [arXiv HTML](https://arxiv.org/html/2602.01539)
- (f) Cold-start refusal by a base-model attacker; strong sensitivity to attacker SFT init and capacity; compute/stability cost; MMLU 0.381 → 0.337 on 14B. (g) https://github.com/BattleWen/MAGIC

#### 33. **[RECENT]** PISmith — Yin, 2026, arXiv:2603.13026, COLM 2026 — [arXiv](https://arxiv.org/abs/2603.13026)
Summary: RL red-teaming of prompt-injection defenses; standard GRPO fails under extremely sparse rewards (most injections blocked, entropy collapses before success), so PISmith adds adaptive entropy regularisation and dynamic advantage weighting; 13 benchmarks, 7 baselines incl. RL-Hammer, agentic settings (InjecAgent, AgentDojo) vs GPT-4o-mini and GPT-5-nano. — [arXiv](https://arxiv.org/abs/2603.13026)
- (a) Modified GRPO. (c) Frozen. (d) Attack-only. (e) "Highest ASR" (numbers not on page). (f) Reward sparsity, entropy collapse. (g) https://github.com/albert-y1n/PISmith

#### 34. **[RECENT]** The Attacker in the Mirror: Anchored Bipolicy Self-Play (ABS) — La Malfa, 2026, arXiv:2605.08427 (2026-05-08) — [arXiv](https://arxiv.org/abs/2605.08427)
Summary: Argues shared-weight self-play (Self-RedTeam) collapses to "self-consistency" because attacker and defender gradients collide; proposes a frozen backbone with two role-specific LoRA adapters trained with the Self-RedTeam recipe. — [arXiv HTML](https://arxiv.org/html/2605.08427)
- (a) REINFORCE++ (following Self-RedTeam), token-level KL; WildGuard reward (unsafe −1, refusal 0); GPT-5.4 judge for evaluation. (b) **Same frozen base, separate LoRA per role** (Qwen2.5-3B/7B/14B-IT; rank 16/32 = 0.9-3.3% added params). (c) Moving. (d) **Self-play** with opponent fixed during each role's gradient.
- (e) WildGuard adversarial ASR 3B: base 0.282, Self-RedTeam 0.234, ABS 0.122; 7B 0.279/0.178/0.169; 14B 0.146/0.116/0.131 (ABS worse); HarmBench 3B 0.949/0.885/0.721; attacker judge-ASR on Harmful Behaviors: Self-RedTeam 18%/9%, ABS 35%/24% (3B/7B); Self-BLEU-3 lower for ABS (3B 0.714 vs 0.785). — [arXiv HTML](https://arxiv.org/html/2605.08427)
- (f) Self-consistency argument is mainly theoretical (no direct collapse statistic); reachable equilibria include trivial always-refuse; ABS attacks are much longer (232 vs 59 tokens at 3B). (g) https://github.com/EmanueleLM/AnchoredBipolicySelf-Play

#### 35. **[RECENT]** Learning to Attack and Defend: Adaptive Red Teaming via GRPO (AdvGRPO) — Bullwinkel (Microsoft), 2026, arXiv:2606.09701 (v1 2026-06-08, v2 2026-10-07) — [arXiv](https://arxiv.org/abs/2606.09701)
Summary: GRPO attacker with multiple GPT-4.1 judge reward channels combined by GDPO (per-channel group normalisation so no channel dominates); curriculum from single-turn to multi-turn attacker-only training, then alternating co-training with a defender. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- (a) GRPO + GDPO advantages; attacker channels: attack success A (w=1.0), prompt quality P (0.5), thinking-trace T (0.5, reasoning models); defender: 1−A on harmful, helpfulness H on benign; deterministic overrides zero rewards for structural markers / language switching. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- (b) **Mixed**: 7B attacker and 7B defender both Qwen2.5-7B-Instruct (same base); 14B attacker vs 7B defender differ; Qwen3.5-9B reasoning attacker; attacker-only target is GPT-4.1. (c) Moving in co-training. (d) **Alternating every N=10 steps**, defender batches 50% adversarial / 50% benign, up to 500 steps; co-training attacker initialised from the attacker-only checkpoint.
- (e) ASR vs GPT-4.1 (AdvBench/HarmBench): Qwen2.5-7B ST 72.5/69.5, MT 90.0/84.5; Qwen2.5-14B MT 90.0/91.0; transfer of 14B MT: Llama-3.1-8B 92.5/88.5, Gemma-2-9B 88.3/86.0. Diversity (14B ST): token entropy 1.02, Self-BLEU 0.42, embedding diversity 0.30; entropy regularisation raises entropy to 1.60 but drops ASR 83.9 → 73.4. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- (f) "GRPO tends to collapse attacker diversity over training"; most attacks use fictional thriller/crime framings; entropy explosion at β_max=0.15; reasoning attackers self-censor early; attackers answered the harmful objective themselves and defenders evaded the scorer by language switching (both patched by overrides); high ablation variance (WildJailbreak ASR 30.4±14.9 without GDPO). — [arXiv HTML](https://arxiv.org/html/2606.09701)
- (g) No repository found on page (Gap).

#### 36. **[RECENT]** GPT-Red: Automated Red Teaming via Self-Play at Scale — Wallace (OpenAI), 2026, arXiv:2607.26115 (2026-07-28) — [arXiv](https://arxiv.org/abs/2607.26115)
Summary: A GPT-5.5-initialised attacker is trained by self-play against a population of simultaneously trained defender agents (mainly prompt injection, also jailbreaks) and used to adversarially train GPT-5.6; the run is described as compute-comparable to GPT-5-series RL post-training. — [arXiv HTML](https://arxiv.org/html/2607.26115)
- (a) Self-play RL (algorithm not named); attacker rewarded for validated failures, defender for resisting while completing its task; invalid attacks get 0; jailbreak rubric rewards; no diversity reward (diversity from the defender population). (b) Attacker from GPT-5.5; controlled experiments use a helpful-only attacker and a smaller GPT-5-series defender; whether full-run defenders share the base is not stated. (c) Moving (population). (d) **Simultaneous co-training**.
- (e) Fake-CoT attack robustness GPT-5.1 5.2% → GPT-5.6 95.9%; GPT-5.6 held-out IPI robustness up to 89%, IPI red-team ASR <4%; "substantially outperforming human red-teamers" on IPI 2025 (no number in text). — [arXiv HTML](https://arxiv.org/html/2607.26115)
- (f) Single-defender training narrowed attacks (mode collapse) → multi-defender population; attacker-only training collapsed; attackers edited parts of the rollout they should not (validity checks); edge-case exploitation acknowledged. (g) None stated.

#### 37. **[RECENT]** RedEvoAgent — Zhang, 2026, arXiv:2608.27439 — [arXiv](https://arxiv.org/abs/2608.27439)
Summary: Black-box agent that evolves a human-readable attack "skill" from trajectories (tool profiling, credit assignment, validation gating); no RL/SFT indicated. Boundary case, listed for completeness. — [arXiv](https://arxiv.org/abs/2608.27439)

#### Survey pointer
- Raheja et al., "Recent advancements in LLM Red-Teaming: Techniques, Defenses, and Ethical Considerations", arXiv:2410.09097 — covers gradient, RL and prompt-engineering attacks; used only as a pointer. — [arXiv](https://arxiv.org/abs/2410.09097)

### Inferences
- The learned-attacker literature splits cleanly on whether the attacker is a free-form prompt policy (Perez, CRT, GFlowNet, Beutel, Jailbreak-R1, AdvGRPO, Self-RedTeam) or a controller over a fixed mutation library (RLbreaker, RL-JACK, Auto-RT); only the former faces the mode-collapse problem in its sharpest form.
- Several much-cited "automated red teaming" systems (Rainbow Teaming, Ferret, X-Teaming, AutoRedTeamer, RvB, RedEvoAgent, FLIRT, JAB) train no attacker weights at all; they should be labelled "search/agentic" rather than "learned attacker" in the final report.

### Gaps
- Code links not confirmed for Perez 2022, MART, Red Teaming Game, Beutel (only a cdn archive), AdvGRPO, GPT-Red, SSP, DialTree, Auto-RT.
- Venue for Rainbow Teaming (NeurIPS 2024), Perez 2022 (EMNLP 2022), X-Teaming (COLM 2025) and Red Teaming Game not verified from arXiv metadata; mark UNVERIFIED.
- AdvPrompter and GBRT numeric ASR and attacker base model were not read from the full text.
- Did not find any paper training an attacker with a reasoning model + RL that reports explicit same-checkpoint initialisation beyond AdvGRPO's Qwen3.5-9B run.

---

## Key Question 2: What rewards are used, and how do papers prevent mode collapse and judge gaming?

### Takeaway
Rewards are almost always a classifier/judge on the target's response (BAD/RoBERTa/ToxiGen classifiers → Llama Guard / WildGuard / Qwen3Guard → LLM judges with rubrics), occasionally the target's log-probability of a harmful completion (AdvPrompter) or a reference-LM likelihood for naturalness (GFlowNet). Anti-collapse tools are, in rough chronological order: KL penalties (Perez), explicit novelty bonuses (CRT), GFlowNet sampling (Lee), constrained/Lagrangian diversity (DiveR-CT), goal decomposition + past-attempt conditioning (Beutel), and, in 2025-26, changing the environment itself (Active Attacks, Self-RedTeam, GPT-Red's defender population). Judge gaming is reported repeatedly and only partly solved (ensembles, cross-judge training, deterministic overrides, validity filters).

### Cited Findings
- Perez: reward −log p(1−r) from a BAD-trained classifier, KL penalty α; at α=0.3, 78% of test cases contain "invisible" (collapse). — [arXiv HTML](https://arxiv.org/html/2202.03286)
- Casper: ensemble classifier logit + intra-batch cosine-distance diversity; without it, prompts collapsed to "would/you" with 0% toxic rate; CREAK/ChatGPT-trained classifiers were hacked by toxic/nonsense text. — [arXiv HTML v3](https://arxiv.org/html/2306.09442v3)
- CRT: toxicity classifier + negative SelfBLEU + cosine novelty + entropy bonus (0.01) + gibberish penalty; entropy bonus alone does not help SelfBLEU; higher temperature falls short; higher KL reduces quality. — [arXiv HTML](https://arxiv.org/html/2402.19464)
- GFlowNet: R = exp((1/β)E log p_toxic) · p_ref(x)^(1/γ); PPO+Novelty gave 0 high-reward prompts on Gemma/Llama-2 while REINFORCE collapsed to one prompt; MLE smoothing removes β sensitivity. — [arXiv HTML](https://arxiv.org/html/2405.18540)
- DiveR-CT: safety and gibberish as Lagrangian constraints with dynamic λ; k-NN cosine (top-16) and 1−BLEU as objectives; RL/CRT ASR drops sharply on a held-out test classifier (over-optimisation), DiveR-CT less so. — [arXiv HTML](https://arxiv.org/html/2405.19026)
- Beutel/OpenAI: multiplicative reward of RBR (LLM yes/no, P(yes)), Moderation API, few-shot similarity, style-subspace diversity, length penalty; attacker optimised the Moderation API and ignored the RBR; padding inflated diversity; grader exploitation "an open problem". — [arXiv HTML](https://arxiv.org/html/2412.18693)
- Red Teaming Game: toxicity model (Bradley-Terry on PKU-SafeRLHF) + population diversity term; motivation explicitly cites Goodhart's law; toxicity reward invalid OOD. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- Active Attacks: Llama-Guard-2 toxicity probability over 5 victim responses; no diversity bonus; instead victim fine-tuning erases reward from exploited regions; cosine distance 0.47 → 0.53 and cross-attack success 0.07% → 31.28%. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- Self-RedTeam: WildGuard-7B three-label judge; no diversity term; diversity emerges from the moving defender (+17.80% SBERT); reward hacking addressed by cross-judge training and independent judges at evaluation. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- MAGIC: Qwen3Guard judge; refusal reward penalises both over-refusal and unsafe compliance; explicitly declines to report a diversity metric, arguing diversity is not a reliable proxy. — [arXiv HTML](https://arxiv.org/html/2602.01539)
- SSP: 1-5 LLM judge, zero-sum clip rewards, UCB replay of low-reward cases. — [arXiv HTML](https://arxiv.org/html/2601.10589)
- AdvGRPO: GPT-4.1 multi-channel judge with GDPO normalisation; "GRPO tends to collapse attacker diversity over training"; entropy regularisation trades ASR (83.9 → 73.4) for entropy (1.02 → 1.60); deterministic overrides patch attackers answering the objective themselves and defenders language-switching. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- RL-Hammer: GRPO with KL removed; authors note attackers reward-hack diversity objectives; sparse rewards against defended targets handled by co-training on an undefended target. — [search summary](https://arxiv.org/html/2510.04885v1)
- PISmith: entropy collapses before any success under standard GRPO; adaptive entropy regularisation + dynamic advantage weighting. — [arXiv](https://arxiv.org/abs/2603.13026)
- GPT-Red: validity checks give invalid attacks reward 0; attacks already containing disallowed content filtered; single-defender training narrowed attacks, fixed by a defender population. — [arXiv HTML](https://arxiv.org/html/2607.26115)
- Jailbreak-R1: separate warm-up stage rewarding diversity and consistency before the jailbreak-reward stage. — [arXiv](https://arxiv.org/abs/2506.00782)

### Inferences
- Explicit diversity bonuses are themselves gameable (padding in Beutel; "reward-hack diversity objectives" in RL-Hammer; surface-level fictional framings in AdvGRPO), which is why the 2025-26 co-training papers drop them in favour of a changing opponent.
- Judge choice is now a first-order design variable: nearly every 2025-26 paper uses a guard model (WildGuard, Llama Guard, Qwen3Guard) for training and a different, stronger judge for evaluation to limit over-optimisation.

### Gaps
- No paper found that quantifies judge-gaming rate directly (e.g., fraction of "successes" overturned by humans) except MART's RM-vs-human gap (4.8% vs 8.0%).

---

## Key Question 3: Which methods initialise the attacker from the same base/checkpoint as the target, and what do they report?

### Takeaway
Same-base initialisation runs from Perez 2022 (Gopher vs Gopher) through MART (LLaMA-65B both sides), Red Teaming Game (stablelm-3b all roles), to the 2025-26 self-play line where the attacker and defender are literally the same weights (Self-RedTeam, SSP) or share a frozen backbone with role adapters (ABS). The 2026 critiques (MAGIC, ABS) report that fully shared weights produce gradient conflict / "self-consistency" and that separate policies or separate adapters give lower ASR against the defender at 3B-7B, though not uniformly at 14B.

### Cited Findings
- Perez: red LM and target both 280B Gopher, differing by prompt only. — [arXiv HTML](https://arxiv.org/html/2202.03286)
- MART: adversarial and target models both from LLaMA-65B SFT'd on the same instruction data; attack success fell 29.74% → 10.21% over four rounds as the target improved. — [arXiv HTML](https://arxiv.org/html/2311.07689)
- Red Teaming Game: red, blue, critic and toxicity model all on stablelm-alpaca-3b. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- JAB: red, target and belief generator each the same model (GPT-Neo 2.7B or Falcon-Instruct 7B). — [arXiv HTML](https://arxiv.org/html/2311.09473)
- Self-RedTeam: one shared πθ plays both roles by role instructions (Qwen2.5-3B/7B/14B, Llama3.1-8B); attacker-only training on a fixed defender collapsed to few clusters. — [arXiv HTML](https://arxiv.org/html/2506.07468)
- SSP: single Qwen2.5-7B-Instruct policy for both roles; "w/o UM" ablation splits roles into two models (effect size not captured in this read). — [arXiv HTML](https://arxiv.org/html/2601.10589)
- ABS: frozen Qwen2.5 backbone + per-role LoRA; shared-weight self-play argued to collapse to self-consistency; ABS attackers reached 17-18% ASR against the shared-weight (Self-RedTeam) defender versus 6% for Self-RedTeam attackers against their own-family defenders; safety better than Self-RedTeam at 3B/7B WildGuard but worse at 14B (0.131 vs 0.116). — [arXiv HTML](https://arxiv.org/html/2605.08427)
- MAGIC: separate attacker and defender policies in the same family; critique that shared parameters cause gradient conflicts; WildGuardTest ASR 2.3% vs Self-RedTeam 25.5% on Qwen2.5-7B. — [arXiv HTML](https://arxiv.org/html/2602.01539)
- AdvGRPO: 7B attacker and 7B defender both Qwen2.5-7B-Instruct, 14B attacker vs 7B defender mixed; no explicit ablation of same-vs-different base found in the text read. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- GPT-Red: attacker initialised from GPT-5.5; defenders are GPT-5-series; base sharing in the full run not stated. — [arXiv HTML](https://arxiv.org/html/2607.26115)
- Different-model attackers (for contrast): CRT (GPT-2 137M vs Dolly/LLaMA2), GFlowNet (GPT-2-small / Llama-3.2-1B vs 7-8B targets), Active Attacks (Qwen2.5-1.5B vs 1.5-7B victims), Beutel (GPT-3.5-sized vs GPT-4T), DialTree (trained vs Llama-3.2-1B, transfers to larger). — [CRT](https://arxiv.org/html/2402.19464); [GFlowNet](https://arxiv.org/html/2405.18540); [Active Attacks](https://arxiv.org/html/2509.21947); [Beutel](https://arxiv.org/html/2412.18693); [DialTree](https://arxiv.org/abs/2510.02286)

### Inferences
- Papers choosing a much smaller attacker than the target (CRT, GFlowNet, Active Attacks, DialTree) do so for cost and still report transfer, so attacker/target size parity is not required for effectiveness; it is the shared-weights (not shared-base) choice that the 2026 papers identify as harmful.
- The 14B reversal in ABS suggests the shared-weights penalty shrinks with model capacity, but this rests on one paper.

### Gaps
- No paper found that ablates "same base, separate copies" vs "different base" with everything else held fixed; MART and RTG use same-base without an ablation.

---

## Key Question 4: Which attackers are trained against a moving target (target also updated)?

### Takeaway
Alternating/co-training papers: Deng 2023 (ICL attacker, iterative victim fine-tuning), Red Teaming Game 2023 (PPO populations, alternating best responses), MART 2023 (SFT attacker, 4 rounds), JAB 2023 (in-context, defence prompt only), DART 2024 (v1 of 2407.03876), Self-RedTeam 2025, Active Attacks 2025 (victim LoRA every 1000 steps), SSP 2026, MAGIC 2026, ABS 2026, AdvGRPO 2026 (alternate every 10 steps), GPT-Red 2026 (simultaneous population). One-shot "attack then safety-tune once" papers (GFlowNet, DiveR-CT, Rainbow Teaming, AdvPrompter, X-Teaming data) are not alternating.

### Cited Findings
- Red Teaming Game: red best response with blue fixed, then blue with red fixed; exploitability 6.2 → 0.8; equilibrium ASR ≈10%. — [arXiv HTML](https://arxiv.org/html/2310.00322)
- MART: four alternating rounds; violation rate −84.7% (RM) / −53.7% (human); stopped when attack success fell to ~10%. — [arXiv HTML](https://arxiv.org/html/2311.07689)
- Deng: "fine-tune victim LLMs through iterative interactions with the attack framework". — [arXiv](https://arxiv.org/abs/2310.12505)
- DART v1: Red LLM and Target LLM "interact iteratively"; target enhanced via active-learning data selection; 53.4% violation reduction (human eval). — [ar5iv v1](https://ar5iv.labs.arxiv.org/html/2407.03876)
- Self-RedTeam: co-evolving roles; defender-only on static attacks overfits; self-play + SFT gives ~17.33% extra safety over defender-only + SFT and higher AlpacaEval-2 LC (38.09 vs 35.50). — [arXiv HTML](https://arxiv.org/html/2506.07468)
- Active Attacks: victim safety-tuned every R=1000 of T=5000 attacker steps, attacker reset and buffer cleared each round; defense rate 91.00 → 99.71; +5.76% compute. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- MAGIC: K rounds of T_D defender steps then T_A attacker steps with the other side frozen. — [arXiv HTML](https://arxiv.org/html/2602.01539)
- AdvGRPO: alternate every N=10 steps, up to 500 steps, defender batches 50/50 adversarial/benign; co-training "difficult" because multiple properties of both models must be optimised jointly. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- GPT-Red: attacker trains against "a suite of past defender models" with defenders "trained simultaneously"; a nine-frozen-defender demo; single-defender training narrowed attacks. — [arXiv HTML](https://arxiv.org/html/2607.26115)
- SSP: both roles updated from the same policy each step after judge scoring. — [arXiv HTML](https://arxiv.org/html/2601.10589)
- ABS: opponent parameters fixed when computing each role's gradient; adapters "evolve separately". — [arXiv HTML](https://arxiv.org/html/2605.08427)
- Attack-only with a later one-off safety tune: GFlowNet (Gemma-2b-it LoRA on collected prompts; most robust to other attackers), DiveR-CT (2:1 Alpaca mix), Rainbow Teaming (fine-tuning on synthetic data). — [GFlowNet](https://arxiv.org/html/2405.18540); [DiveR-CT](https://arxiv.org/html/2405.19026); [Rainbow](https://arxiv.org/abs/2402.16822)

### Inferences
- Every alternating paper that measured it reports the attacker's success falling across rounds (MART 31.9% → 10.2%; RTG to ~10% at equilibrium), so a moving-target attacker needs either resets (Active Attacks) or populations (RTG, GPT-Red) to keep learning signal.
- Alternation granularity varies by three orders of magnitude (per-step in SSP/Self-RedTeam, every 10 steps in AdvGRPO, every 1000 steps in Active Attacks, per round of full fine-tuning in MART); no paper compares granularities directly.

### Gaps
- No paper found that reports attacker-side metrics (ASR vs the current defender, diversity) as a function of round for a GRPO/REINFORCE++ attacker beyond the text summaries above; the curves are in figures that were not read.

---

## Key Question 5: 2025-2026 developments (reasoning models, multi-step RL, self-play)

### Takeaway
Since early 2025 the field moved from attacker-only PPO/GFlowNet to (i) GRPO attackers with multi-channel LLM-judge rewards (Jailbreak-R1, RL-Hammer, PISmith, AdvGRPO), (ii) multi-turn RL attackers with tree search (DialTree, ICLR 2026) and multi-turn co-training (MAGIC), (iii) same-weights self-play with hidden CoT (Self-RedTeam, ICML 2026; SSP) and its critiques (ABS, MAGIC), (iv) environment-adaptive curricula (Active Attacks, ICML 2026), and (v) frontier-scale self-play used in production safety training (GPT-Red, OpenAI, July 2026).

### Cited Findings
- Jailbreak-R1 (Jun 2025): three-stage SFT→diversity-warm-up→progressive-reward RL; 7.6B Qwen2.5-7B-Instruct-based release. — [arXiv](https://arxiv.org/abs/2506.00782); [HF](https://huggingface.co/yukiyounai/Jailbreak-R1/blob/main/README.md)
- Self-RedTeam (Jun 2025, ICML 2026): REINFORCE++ self-play with hidden CoT; +17.80% SBERT diversity, up to 95% safety gain. — [arXiv HTML](https://arxiv.org/html/2506.07468); [ICML](https://icml.cc/virtual/2026/poster/61969)
- QDRT (Jun 2025, v2 Aug 2026): population of behaviour-conditioned attackers vs GPT-4.1 and GPT-5-Chat. — [arXiv](https://arxiv.org/abs/2506.07121)
- Active Attacks (Sep 2025, ICML 2026): adaptive victim; 400× cross-attack gain. — [arXiv HTML](https://arxiv.org/html/2509.21947)
- DialTree (Oct 2025, ICLR 2026): on-policy RL + tree search, 81.5% avg ASR over 12 targets, trained only vs Llama-3.2-1B-Instruct. — [arXiv](https://arxiv.org/abs/2510.02286)
- RL-Hammer (Oct 2025): GRPO from scratch, 98% ASR GPT-4o, 72% GPT-5 with Instruction Hierarchy. — [search summary](https://arxiv.org/html/2510.04885v1)
- SSP (Jan 2026): single-model self-play with UCB reflective replay. — [arXiv HTML](https://arxiv.org/html/2601.10589)
- RvB (Jan 2026): training-free red-blue game for hardening. — [arXiv](https://arxiv.org/abs/2601.19726)
- MAGIC (Feb 2026): separate-policy GRPO co-evolution; argues diversity metrics are not a reliable proxy. — [arXiv HTML](https://arxiv.org/html/2602.01539)
- PISmith (Mar 2026, COLM 2026): GRPO fixes for extreme reward sparsity. — [arXiv](https://arxiv.org/abs/2603.13026)
- ABS (May 2026): LoRA-per-role self-play; self-consistency critique. — [arXiv HTML](https://arxiv.org/html/2605.08427)
- AdvGRPO (Jun 2026, v2 Oct 2026): GRPO + GDPO, Qwen3.5-9B reasoning attacker with a thinking-trace reward channel; reasoning attackers "self-censor" early then overcome it after ~10 steps; GRPO collapses attacker diversity. — [arXiv HTML](https://arxiv.org/html/2606.09701)
- GPT-Red (Jul 2026): GPT-5.5-initialised attacker, population self-play, used to train GPT-5.6; described as the largest documented LLM safety training run; beats human red-teamers on IPI 2025. — [arXiv HTML](https://arxiv.org/html/2607.26115)
- RedEvoAgent (Aug 2026): inference-time skill evolution, no weight training. — [arXiv](https://arxiv.org/abs/2608.27439)
- Secondary citations (not opened): AutoInject (Chen et al., 2026) trains prompt-injection attackers with GRPO; Feng et al. (2026) train multi-turn attackers open-loop. — [search summary](https://arxiv.org/html/2603.13026)

### Inferences
- The 2026 consensus failure mode for GRPO attackers is entropy/diversity collapse rather than ineffectiveness: AdvGRPO, PISmith and RL-Hammer all report it and patch it with entropy terms, KL removal or advantage reweighting rather than solving it.
- The two 2026 critiques of shared-weights self-play (MAGIC, ABS) agree on the diagnosis (gradient conflict) but disagree on the remedy (separate policies vs separate adapters) and neither beats Self-RedTeam uniformly across sizes.

### Gaps
- AutoInject (Chen 2026) and Feng et al. (2026) were seen only as citations inside PISmith/AdvGRPO; IDs UNVERIFIED and not catalogued.
- No peer-reviewed venue found yet for Jailbreak-R1, QDRT, Auto-RT, xJailbreak, RL-Hammer, SSP, MAGIC, ABS, AdvGRPO, GPT-Red, RvB or RedEvoAgent as of 2026-10-17.
- Exact GRPO hyperparameters and diversity numbers for Jailbreak-R1, DialTree and PISmith were not read (abstract pages only).

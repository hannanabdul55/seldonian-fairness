# Gap check: couplings 1 and 2 (harm policy's RESPONSES as a training signal for the main model)

Date: 2026-10-17. Adversarial novelty check: the aim was to falsify the claim, not confirm it.

**Claim under test.** No published paper trains a separate policy, initialised from the same base model as the main LM, with RL to generate harmful or otherwise undesirable RESPONSES (not prompts), with reward defined by the main model's failure, and alternates that policy's updates with the main model's updates (GAN / minimax style).

- **Coupling 1:** the harm policy is rewarded for its own bad responses, and the main model is pushed away from them (negatives, unlikelihood, contrastive, DPO rejected). In the adversarial version the negative generator is re-trained against the current main model each round.
- **Coupling 2:** the harm policy is rewarded by the main model's own likelihood (or log-odds versus a safe response) of the harmful response. The main model is then trained to lower that likelihood.

**Excluded as already known:** Token Buncher 2508.20697, AntiDote 2509.08000, DR-SFT 2606.30263, weak-to-strong jailbreak 2401.17256, Emulated Disalignment 2402.12343, APO 2311.08045, and the prompt-generator loops (MART, Self-RedTeam, MAGIC, TriPlay-RL, AdvGRPO, CHASE, Active Attacks, GPT-Red). New finds are compared to these where it helps.

**Axes:**
- (a) separate policy from the main model
- (b) same base
- (c) trained with RL
- (d) generates responses, and are they harmful
- (e) reward defined by the main model's failure or likelihood
- (f) alternating or iterated with the main model's updates

## Search scope

- **Searched:** WebSearch (standard and extended) plus WebFetch of arXiv abs and HTML method sections, for:
  - adversarial or learned negative generators for DPO, unlikelihood and contrastive training ("adversarial DPO", "hard negatives", "negative generator", "rejected-response generator")
  - GAN-style and self-play alignment (SPIN, SPPO, Nash-PO, GAPO, RLGAF, GAD, SGALM)
  - anti-expert, evil-twin, negative-teacher and misaligned-reference policies
  - likelihood-targeted harmful completions (prefill and recovery training, worst-case continuation states, unsafe-decoding paths, rare-output importance sampling)
  - generator-versus-classifier and generator-versus-reward-model games
  - adversarial unlearning with a learned generator
- **Coverage:** 2018 (ACE) to October 2026.
- **Not searched:** non-English venues, OpenReview-only submissions without arXiv, workshop-only papers, patents. Weight-space attackers (TAR, Booster and similar) were left to harmful_twins_tamper.md, and prompt attackers to alternating_training.md and rl_attackers.md.

## Candidate records

### A. Closest structural matches for Coupling 1 (separate RL generator of bad outputs, reward = main model's failure, alternating). In all of these the "main model" is an evaluator, not the generative policy.

**A1. Adversarial Training for Process Reward Models (APRM).** Gurusha Juneja, 2025 (arXiv v1 28 Nov 2025), ICML 2026 poster (icml.cc/virtual/2026/poster/66402). arXiv 2511.22888, VERIFIED.
- **Mechanism (HTML method section):**
  - A Generator takes a correct reasoning step and writes a plausible but incorrect variant. A PRM learns to catch it.
  - Generator and PRM both use a Llama-3.1-8B backbone. Both are trained with PPO plus OGDA, with KL to each player's initial policy.
  - Generator reward: +1 if it writes an incorrect step that the PRM accepts, 0 if the PRM catches it, -1 if it writes a correct step. Correctness is set by an algorithmic oracle.
  - "We alternate training between players, updating one for 5 gradient update steps while freezing the other." The PRM trains on negatives from current and past Generator versions.
- **Verdict:**
  - (a) yes
  - (b) yes, same Llama-3.1-8B backbone (identical checkpoint not stated)
  - (c) yes (PPO)
  - (d) generates responses (reasoning steps), but undesirable means wrong, not harmful
  - (e) yes, reward is the PRM's failure
  - (f) yes, alternating
- **What it misses:** The main model is a PRM, a verifier or classifier, not the deployed generative policy. It is trained to classify the negatives, not pushed away from them by likelihood. Nothing is about harm.
- **Strongest anticipation of Coupling 1's game structure.**

**A2. Adversarial Training of Reward Models (Adv-RM).** Alexander Bukharin (NVIDIA), 2025, arXiv 2504.06141, VERIFIED. Also in alternating_training.md.
- **Mechanism:**
  - An RLOO policy, initialised from Llama-3.1-8B-Instruct (πSFT), is rewarded when the target RM scores its response above a prompt threshold T(x) while an ensemble disagrees.
  - About 1,000 adversarial pairs per round are added, and the RM is retrained from scratch.
  - Two rounds; "no longer successful" after round 2.
- **Verdict:**
  - (a) yes
  - (b) no (target RMs such as Nemotron-340B differ from the attacker base)
  - (c) yes
  - (d) yes, responses, but low-quality and out-of-distribution rather than harmful
  - (e) yes, the RM's failure
  - (f) yes, rounds
- **What it misses:** The main model is a reward model. Different base. Not harm.

**A3. Enhancing LLM Safety Through a Theoretical Minimax Game Lens (DuoGuard).** Yihe Deng, 2025 (v2 15 Jun 2026), arXiv 2502.05163, VERIFIED.
- **Mechanism:**
  - The generator (dolphin-2.9.4-llama3.1-8b, uncensored) rewrites seed examples into new label-preserving texts.
  - Reward is the guard classifier's NLL of the correct label. Misclassified samples are preferred and correct ones dispreferred in DPO.
  - The classifier (Qwen2.5-0.5B/1.5B) is retrained from scratch each iteration.
- **Verdict:**
  - (a) yes
  - (b) no
  - (c) offline DPO, described as "minimax RL"
  - (d) generates user-style queries or texts to classify, not assistant responses; the unsafe ones are harmful
  - (e) yes, the classifier's failure
  - (f) yes, iterative
- **What it misses:** The main model is a guard classifier. It generates inputs rather than responses. Different base.

**A4. Related evaluator-side games (weaker):**
- **Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training.** Aleksei Ilin, 2025, arXiv 2507.08284, VERIFIED. The abstract only says an RL-guided generator makes hard examples for a guard classifier in "iterative adversarial training". A third-party summary reports reward hacking.
- **ToxiGAN.** Peiran Li, EACL 2026, arXiv 2601.03121, VERIFIED. GAN-style toxic-text augmentation for hate-speech classifiers. Not RL. The main model is a classifier.
- **GAPO.** Learning Preferential Prompt through Generative Adversarial Policy Optimization, ACL 2025, arXiv 2503.20194, VERIFIED. The policy LLM is the generator and an encoder-only RM is the discriminator, alternating on odd and even steps. No separate negative generator. Not safety.

### B. Static or same-weights negative generators for safety (Coupling 1 without the adversarial RL part)

**B1. Adversarial DPO (ADPO).** San Kim, NAACL Findings 2024, arXiv 2405.12900, VERIFIED.
- **Mechanism (HTML method section):**
  - An "inner toxic model" π_tox is made by SFT on toxic dialogues tagged with a control token, "maintaining the same parameter set as the original model". π_tox is non-trainable.
  - Toxic responses are sampled once, at temperature 1.5.
  - The loss adds γ·log(π_θ(y_t|x)/π_tox(y_t|x)) to push π_θ below π_tox on the toxic responses.
- **Verdict:**
  - (a) partly (same weights plus a control token, then a frozen copy)
  - (b) yes
  - (c) no (SFT)
  - (d) yes, harmful responses
  - (e) no reward
  - (f) no, generated once
- **What it misses:** RL, a failure-defined reward, alternation.
- **Closest same-base harmful-response negative for DPO, and the earliest (2024). Like DR-SFT, but older.**

**B2. MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization.** Yougang Lyu, 2024 (v2 Mar 2025), arXiv 2410.07672, VERIFIED. Venue not on page.
- **Mechanism:**
  - Negative weak teachers are SFT'd on negative behaviour data, and a negative strong student is SFT'd on their labels.
  - Their outputs are the DPO rejected side for the positive strong student and the positive weak teachers, iterated over rounds.
  - Evaluated on HH-Harmless and PKU-SafeRLHF.
  - "the negative agents remain unchanged after initialization."
- **Verdict:**
  - (a) yes
  - (b) likely same backbones as the positive agents (inferred)
  - (c) no
  - (d) yes, harmful responses
  - (e) no
  - (f) the main model iterates, but the negatives are frozen
- **What it misses:** RL, a failure-defined reward, re-training of the negative generator.

**B3. Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation (AIDSAFE).** Tharindu Kumarage (Amazon), 2025, arXiv 2505.21784, VERIFIED.
- **Mechanism:**
  - An "ear-whisperer" LLM agent (Mixtral-7B-Instruct family) writes deceptive belief PREFIXES. These are prepended when sampling rejected CoTs from the target for DPO.
  - The agent is refined by iterative in-context learning (FLIRT/JAB style) over 100 iterations against the target, scored by ShieldGemma.
- **Verdict:**
  - (a) yes
  - (b) same family
  - (c) no (in-context refinement)
  - (d) prefixes, not responses; the rejected responses come from the target itself
  - (e) a judge score of target failure
  - (f) the agent iterates against the target, but the target is not shown updating between iterations
- **What it misses:** RL, response modality, alternation.

**B4. Background:**
- **The CRINGE Loss.** Leonard Adolphs, ACL 2023, arXiv 2211.05826, VERIFIED. A classifier labels the model's OWN generations as negatives over iterative rounds.
- **D2O.** Shitong Duan, EMNLP Findings 2024, arXiv 2403.03419, VERIFIED. Human negatives with self-generated anchors.
- **NEAT.** Shiqi Qiao, 2024, arXiv 2410.12194, VERIFIED. Negative prompts make the same model produce undesirable responses online.
- **Verdict for all three:** the main model is its own negative generator. No separate policy, no RL generator.

### C. GAN / self-play alignment: the opponent is the previous iterate, a discriminator, or a reward model, never a separate RL generator of harm

- **SPIN.** Zixiang Chen, ICML 2024, arXiv 2401.01335, VERIFIED.
  - The opponent is the previous iterate of the same model. Its responses are the negatives against human data.
  - Verdict: (a) no; (b) yes; (c) no (it is not trained against the main model, it is just the old weights); (d) responses, not harmful; (e) no; (f) yes (iterations).
- **SPPO.** Yue Wu, arXiv 2405.00675, VERIFIED.
  - Constant-sum self-play against the previous policy, with a preference oracle.
  - The opponent is the self or previous iterate. It seeks winning (good) responses, so the direction is opposite to the claim. Not harm.
- **Nash-PO family, e.g. ENPO.** arXiv 2606.01382, VERIFIED via search snippet only.
  - Opponents are best responses that try to BEAT the main model, i.e. better responses rather than bad ones.
- **SGALM: Self-Generative Adversarial Fine-Tuning for LLMs.** Shiguang Wu, 2026, arXiv 2602.01137, VERIFIED.
  - One LLM is both G and D. The generator is updated by a score-function gradient whose reward is the model's own "Real" probability of the generated sample. Alternating D and G updates.
  - A "w/o S" ablation uses two separate models.
  - This is the closest Coupling-2-shaped reward (the main model's own judgment of the generated response), but the generator tries to look REAL (good). There is no harm, and the evaluations are GSM8K, ARC and MBPP.
  - Verdict: (a) no (ablation only); (b) yes; (c) policy gradient; (d) responses, not harmful; (e) the main model's score, but for realism; (f) yes.
- **GAD (Black-Box On-Policy Distillation of LLMs).** Tianzhu Ye, 2025, arXiv 2511.10643, VERIFIED. The student is the generator and the discriminator is a co-evolving RM.
- **RLGAF.** Zhang Ze Yu, arXiv 2305.06176, VERIFIED. Generative adversarial reward modelling; the discriminator replaces the RM.
- **Verdict for GAD and RLGAF:** the main LM is the generator, and no harm generator exists.

### D. Coupling-2 analogues: likelihood-targeted harmful completions, with no learned policy

**D1. When Autoregressive Consistency Hurts Safety Alignment.** Bochen Lyu, 2026 (v1 2 Jun 2026), arXiv 2606.04168, VERIFIED.
- **Mechanism (HTML method section):**
  - A 50-token harmful span from a dataset (Qi et al. 2025) is inserted into a refusal at 32 candidate positions.
  - The "worst" state is the argmax of Φ = H − R, where H is the current model's length-normalised log-prob of continuing the harmful span and R is its best refusal-restart log-prob. The paper "selects the inserted candidate state where the model most strongly favors harmful continuation over refusal recovery."
  - It then trains refusal NLL from that state. No RL.
- **Verdict:**
  - (a) no
  - (b) n/a
  - (c) no
  - (d) harmful responses, but selected rather than generated
  - (e) yes, exactly the main model's log-odds of harmful versus safe
  - (f) the selection depends on the current θ, so it is implicitly on-policy
- **This is the reward of Coupling 2 used as a selection rule, with no generator policy. Clear static precedent.**

**D2. Probing the Safety Response Boundary of LLMs via Unsafe Decoding Path Generation (JVD).** Haoyu Wang, 2024, arXiv 2408.10668, VERIFIED.
- **Mechanism:** A cost value model steers the safe LLM's OWN decoding toward toxic outputs. These are harmful completions within the target's own distribution.
- **Verdict:** (a) a value model guides the target's own decoding, no separate policy; (b) n/a; (c) no; (d) harmful responses; (e) partly, they are sampled from the target's own distribution; (f) not shown (abstract only).
- **What it misses:** a training loop.
- **Decoding-time precedent, like Emulated Disalignment.**

**D3. Continuous and latent adversarial training (in alternating_training.md and harmful_twins_tamper.md):**
- **CAT/CAPO.** Sophie Xhonneux, arXiv 2405.15589.
- **Targeted LAT.** arXiv 2407.15549.
- **Mechanism:** embedding or latent perturbations maximise the main model's likelihood of a fixed harmful target response, alternating every batch with toward/away losses.
- **What they miss:** the adversary is a perturbation, not a policy, and the harmful responses are fixed targets, not generated.
- **Coupling 2 in perturbation space.**

**D4. Rare-output estimation:**
- **Estimating the Probabilities of Rare Outputs.** arXiv 2410.13211.
- **Rare Event Estimation via Iterative Unalignment.** arXiv 2609.24969, VERIFIED abstract only.
- **Mechanism:** a perturbed copy of the model serves as an importance-sampling proposal for rare harmful outputs.
- **What it misses:** the main model is never trained, and the proposal is built by gradient search over weights, not RL.

### E. Unlearning

- **Adversarial Reinforcement Learning for Robust Diffusion LLM Unlearning.** ICML 2026 poster (icml.cc/virtual/2026/poster/65904). arXiv UNVERIFIED (no listing found).
  - A context generator adversarially produces "informative contexts", i.e. prefixes and suffixes, to elicit unlearned knowledge, while the diffusion LM is optimised to suppress recall.
  - Verdict: (a) yes; (b) unknown; (c) yes per the title; (d) contexts, not responses; (e) yes, the unlearned model's recall; (f) yes, co-training.
  - Prompt-side, so it falls outside the claim's response modality.
- **Min-max unlearning (SAM / PRISM / LAU / AGTAO).** arXiv 2502.05374, 2509.23362, 2408.10682, 2602.01703. The adversary is a weight or latent perturbation, not a generator.
- **Adversarial Mixup Unlearning.** arXiv 2502.10288. A learnable mixup generator for image classifiers.

### F. Pre-LLM precedent for a learned negative generator

- **Adversarial Contrastive Estimation.** Avishek Joey Bose, ACL 2018, arXiv 1805.03642, VERIFIED.
  - A conditional generator trained with REINFORCE supplies hard negatives in a minimax game against an embedding model (word, order and knowledge-graph embeddings).
  - This is the general template of Coupling 1, without LMs or harm.
- **IRGAN.** Wang et al. 2017; arXiv ID not fetched. A similar idea for information retrieval.

## Inferences

1. **Coupling 1, adversarial version.** The full game (separate RL generator, same backbone, reward = main model's failure, alternating) exists in APRM (2511.22888) and, with a different base, in Adv-RM (2504.06141).
   - In both, the "main model" is a verifier or reward model, not the generative policy, and the bad outputs are wrong or low-quality rather than harmful.
   - Moving that game to a generative main model trained to push away from harmful responses is the remaining gap.
2. **Coupling 1, static version.** It is anticipated for safety: ADPO 2024 (same weights, toxic control token, rejected side), MACPO (separate frozen negative agents on HH-Harmless and PKU-SafeRLHF), and DR-SFT (already known). None re-trains the negative generator with RL against the current main model.
3. **Coupling 2.** The reward (the main model's log-odds of a harmful continuation versus a safe one) already appears as a selection rule in 2606.04168, as decoding-time steering in JVD and Emulated Disalignment, and as a perturbation objective in CAT and LAT. SGALM uses the main model's own probability as an RL-style reward for a generator, but for realism, with one shared model and no harm.
   - No paper found trains a separate policy whose reward is the main model's likelihood of the policy's harmful response.
4. **GAN / self-play alignment never has a separate harm generator.** In SPIN the opponent is the previous iterate. In SPPO and Nash-PO the opponent seeks better responses. In APO, GAPO, GAD and RLGAF the adversary is the discriminator or RM, and the main LM is the generator.

## Overall verdict

**NOT FALSIFIED on the full conjunction. PARTIALLY ANTICIPATED for each coupling.**

| Paper | Coupling | Missing axes |
|---|---|---|
| APRM 2511.22888 | 1 (adversarial) | main model is a PRM classifier, not a generative policy; errors, not harm |
| Adv-RM 2504.06141 | 1 (adversarial) | main model is an RM; different base; not harm |
| DuoGuard 2502.05163 | 1 (adversarial) | main model is a classifier; generates inputs, not responses; DPO; different base |
| ADPO 2405.12900 | 1 (static) | no RL, no failure-defined reward, no alternation |
| MACPO 2410.07672 | 1 (static) | negatives SFT and frozen; no reward; no RL |
| AIDSAFE 2505.21784 | 1 (partial) | prefixes, not responses; in-context, not RL |
| 2606.04168 | 2 | no generator policy; reward used as a selection rule over dataset spans |
| SGALM 2602.01137 | 2 (shape) | single shared model; generator seeks realism, not harm |
| JVD 2408.10668, CAT/LAT | 2 | decoding or perturbation adversary, not a trained policy |

**Wording note for the claim:** state it as specific to a *generative* main model and to *harmful* responses. Cite APRM and Adv-RM as the same game played against a verifier or reward model, ADPO, MACPO and DR-SFT as static same-base harmful negatives, and 2606.04168 as the likelihood-margin selection precedent for Coupling 2. Without those qualifiers, the phrase "otherwise undesirable responses" is close to falsified by APRM (only the main model's role differs).

# Negative results: harmful twins, harmful negatives, likelihood-lowering updates, harmful-continuation selection

Compiled 2026-10-17 for the harm-twin design report. The design: a same-base "harm twin" LM, trained with RL, generates harmful responses that the main model is trained against, in alternation.
- **Coupling 1:** the main model is trained away from the twin's responses, used as negatives.
- **Coupling 2:** the twin is rewarded by the main model's own likelihood of the twin's harmful response, and the main model then lowers that likelihood.

**Method.** All 14 papers were downloaded from `arxiv.org/html/<id>`, converted to plain text and read section by section: ablations, appendices, limitations, "we also tried" remarks and hyperparameter notes. Every ID resolved on arxiv.org and its title matched, so all are **VERIFIED**.
- Two IDs had to be looked up:
  - NPO (Zhang et al. 2024) is **arXiv 2404.05868**.
  - Yao et al. 2023, *Large Language Model Unlearning*, is **arXiv 2310.10683**.
- The AntiDote HTML (2509.08000) came through in full this time and was not refused, so the PDF was not needed.
- The item-5 checks for later re-evaluations are brief web searches. "None found" means none turned up in a brief search; it does not mean none exist.
- Search-only claims that were not read in full are marked **UNVERIFIED**.

**Labels.** [AUTHOR-REPORTED, §/Table] means the paper reports it about itself. [INDEPENDENT, source] means another group reports it. Text marked "my reading" is the compiler's own inference.

**Order:** ADPO, MACPO, DR-SFT, Simko et al., Token Buncher, AntiDote, TAR, Qi et al., NPO, Yao et al., Lyu et al., AdvPrefix, Self-ReSET, Twisted SMC, then the cross-paper lessons.

---

## ADPO, Adversarial DPO, Kim and Lee (arXiv 2405.12900, VERIFIED via arxiv.org/html fetch; Findings of NAACL 2024)

Method: DPO plus an extra term that lowers the policy's likelihood of responses sampled from a frozen "inner toxic model" (the same SFT weights prompted with a `[TOXIC]` control token), relative to that toxic model: R_gamma = gamma(log pi_tox(y_t|x)/pi_theta(y_t|x) - log pi_tox(y_w|x)/pi_theta(y_w|x)), added inside the DPO sigmoid (Eq. 3-5).

### 1. Ablated variants or components that hurt
- [AUTHOR-REPORTED, §5.2 / Table 1] ADPO's own toxicity on BAD is **higher** than plain DPO: ADPO 1.2% vs DPO (non-toxic) 0.1% and DPO (toxic) 0.2%. The authors: "these toxicities in ADPO are still marginally higher than those observed in DPO, which demonstrates near-zero toxicity."
- [AUTHOR-REPORTED, Table 1] The prerequisite step (SFT that also teaches the toxic mode via the control token) makes the base model worse: BAD toxicity 13.3% (SFT toxic) vs 4.7% (SFT non-toxic); coherence 73.8% vs 86.0%. The same-weights harm mode leaks into ordinary generation before ADPO cleans it up.
- [AUTHOR-REPORTED, §5.3 / Table 1] Plain DPO trained on top of the toxic-SFT base (i.e. having the harm twin in the weights but no harm-negative term) costs coherence -10.8% and evasiveness +8.7% on BST relative to ADPO.
- [AUTHOR-REPORTED, §5.2 / Table 1] Plain DPO raised evasiveness on BAD by >20 points over its base (35.1% to 56.0%); "DPO tends to train models towards increased evasiveness and reduced coherence."

### 2. Abandoned, unstable, collapsing, over-refusing or reward-hacked
- [AUTHOR-REPORTED, §3.2] "relying solely on R_beta can lead to instability due to the potential ambiguity in the criteria for chosen and rejected labels." The penalty term was added for stability.
- [AUTHOR-REPORTED, §5.4] Pushing the toxic-response likelihood too low is flagged as a reward-hacking route: "extremely low values may lead to 'reward hacking' ..., where the model produces nonsensical but non-toxic responses." No mitigation beyond monitoring the "toxic KL" is given.
- [AUTHOR-REPORTED, §5.2] DPO baseline over-refuses / evades ("I don't know", "I don't understand"), attributed to mode collapse.
- [AUTHOR-REPORTED, §8] The toxic control token stays in the deployed weights: "it is still possible to inadvertently activate the model's inherent toxicity."

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, §4.3] Best settings found by sweep: DPO beta=0.9 at 2 epochs; ADPO beta=0.3, gamma=0.2 at 4 epochs (5 epochs trained). LR 3e-5, LoRA r=16.
- [AUTHOR-REPORTED, §5.4 / Fig. 3] DPO's chosen-KL "experienced a rapid drop, demonstrating sensitivity to the beta"; optimal chosen-KL window stated as [-2, 1].
- [AUTHOR-REPORTED, §4.3] Single run, seed 42, no variance reported; toxic responses sampled at temperature 1.5.

### 4. Stated limits
- [AUTHOR-REPORTED, §9] LLM labelling (Llama2-13B-chat) can transfer the labeller's bias; GPT-4 evaluation validated against 3 annotators from one demographic group (F1 0.776; evasiveness F1 only 0.704, Table 2).
- [AUTHOR-REPORTED, §9] Only 16k of 160k HH dialogues used; toxic model trained on 8k BAD dialogues only. Dialogue toxicity only (Llama-2-7B), no jailbreak or capability-harm evaluation.

### 5. Independent re-evaluations
- [INDEPENDENT] None found in brief search (web search for re-evaluations or failures of ADPO, 2026-10-17).

### 6. Implication for the harm-twin design
- Coupling 1: a same-weights harm mode as the negative source works but its likelihood-lowering term needs an explicit floor or KL anchor to pi_tox, because the authors themselves name "nonsensical but non-toxic" output as the failure when the toxic likelihood is pushed too low; and keeping the twin in the main model's weights raised base toxicity from 4.7% to 13.3%, so the twin should be separate weights.

## MACPO, Lyu et al. (arXiv 2410.07672, VERIFIED via arxiv.org/html fetch; ICLR 2025)

Method: weak teachers and a strong student exchange positive samples each round and are trained with DPO against "familiar" negatives generated by their own negatively fine-tuned copies (a negative weak teacher and a negative strong student fine-tuned on harmful/unhelpful data).

### 1. Ablated variants or components that hurt
- [AUTHOR-REPORTED, §6.4 / Fig. 4] -HN (replacing the student's own-negative-twin samples with negatives from the weak teachers' negative models) "results in substantial performance degradation"; "penalizing unfamiliar negative behavior of weak teachers leads to poor alignment performance." Figure only, no numbers in text.
- [AUTHOR-REPORTED, §6.4 / Fig. 4] -MP (student uses its own positives) "leads to strong student collapse and the alignment performance decrease from the second iteration round."
- [AUTHOR-REPORTED, §6.4] -IW (freeze weak teachers) decreases performance.
- [AUTHOR-REPORTED, App. G.2 / Table 5] Removing perplexity filtering of positives: average reward 59.47 to 55.35 (iter1), 67.35 to 64.63 (iter2), 67.85 to 64.50 (iter3); HH-Harmless iter3 70.25 to 62.49.
- [AUTHOR-REPORTED, §6.2 / Fig. 2] With one weak teacher, "the alignment performance of the strong student starts to degrade after the second round across all datasets."

### 2. Abandoned, unstable, collapsing, over-refusing or reward-hacked
- [AUTHOR-REPORTED, §6.1 / Table 1] Self-play baselines collapse across iterations: SPIN average 51.62 (iter1) to 46.02 (iter2), PKU-SafeRLHF 55.52 to 40.97; Self-rewarding 56.57 (iter2) to 55.62 (iter3). "self-alignment methods use self-generated data to continually train LLM, leading to collapse during multiple iterative optimization rounds."
- [AUTHOR-REPORTED, App. G.1 / Table 4] Iterated strong-to-weak baselines stall or regress on harmlessness: RLAIF HH-Harmless 56.37, 53.02, 52.99 over iters 1-3; RLCD HH-Harmless 59.23, 57.34, 56.88.
- [AUTHOR-REPORTED, App. G.3 / Table 6] MT-Bench: SPIN 2.56, Naive SFT 2.11 vs base-quality baselines ~4.2-4.6 (general-quality degradation of self-play/weak-label baselines).
- [AUTHOR-REPORTED, §4.3] The negative agents are never updated: "we only iteratively optimize the positive agents and the negative agents remain unchanged after initialization." The negatives are a static twin, not an adversary.
- [AUTHOR-REPORTED, App. A] Training order was changed: "the initialized student model is not well aligned with the teacher model, so we further optimize the student model" first in each round.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App. E.1] DPO LR 1e-5, batch 16, gamma 0.2, 1 DPO epoch per iteration, 3 iterations; "We determine the hyperparameters through pilot experiments." No beta or iteration-count sweep reported.
- [AUTHOR-REPORTED, §6.2] Number of rounds before collapse depends on the number of weak teachers (diversity of positives).

### 4. Stated limits
- [AUTHOR-REPORTED, App. H] Only helpfulness/harmlessness evaluated.
- [AUTHOR-REPORTED, App. H] "we have only considered fine-tuning on negative behavioral data as a way of inducing negative behavior"; adversarial prompting/decoding to diversify negatives left for future work.
- [AUTHOR-REPORTED, §5.4 / App. D] Evaluation rests on one third-party reward model (OASST pythia-6.9B) plus GPT-4o and 100-item human samples; no jailbreak or attack evaluation.

### 5. Independent re-evaluations
- [INDEPENDENT] None found in brief search (2026-10-17).

### 6. Implication for the harm-twin design
- Coupling 1: negatives must come from a twin of the main model itself (other-model negatives degraded alignment substantially), but MACPO's twin was frozen; the loop's positives side collapses from round 2 when it is self-generated with low diversity, so the main model's positive targets need an external or diverse source, not only self-samples.

## DR-SFT, Dual-Reference SFT, An et al. (arXiv 2606.30263, VERIFIED via arxiv.org/html fetch)

Identity check: arXiv 2606.30263 is "Defending Against Harmful Supervision Hidden in Benign Samples", Bang An, Yibo Yang, Dandan Guo, Ebtisam Alshehri, Carlos Hinojosa, Bernard Ghanem (KAUST), which proposes DR-SFT. Confirmed.
Method: a DPO-style SFT loss with two references, the aligned model and a harm-fine-tuned "negative reference" twin of it; a token-level control eta from log pi_p - log pi_n down-weights the SFT gradient on tokens the harm twin rates far more likely (gradient factor (1 - eta), so it gates learning rather than ascending).

### 1. Ablated variants or components that hurt
- [AUTHOR-REPORTED, App. C.2 / Table 6] Weaker harm twin weakens the defence: negative reference trained on 2500 / 1000 / 500 / 250 unsafe PKU-SafeRLHF samples gives BeaverTails ASR 14.3 / 14.5 / 15.1 / 19.8%.
- [AUTHOR-REPORTED, App. C.1 / Table 5] eta_tanh instead of eta_sig: ASR 13.4% vs 14.3% but GSM8K 35.2 vs 38.5; rejected for utility cost.
- [AUTHOR-REPORTED, §5.4 / Table 4] Larger beta trades utility for safety: beta 0.1 / 0.01 / 0.001 gives BeaverTails ASR 13.4 / 14.1 / 14.3 and GSM8K 36.5 / 37.8 / 38.5.
- [AUTHOR-REPORTED, Tables 1-2] Residual failures of DR-SFT itself under Embedded Attack: Samsum+HEx-PHI ASR 48.7% (Llama-2), 75.4% (Gemma-3-4B), 40.9% (Qwen); GSM8K+HEx-PHI 33.3% (Qwen), 42.0% (Gemma). Backdoor setting, triggered ASR 49.1% (Llama-2, GSM8K+BeaverTails), 38.5% (Gemma). Utility on Samsum drops below all baselines (e.g. Gemma 47.9 vs SFT 50.5).
- [AUTHOR-REPORTED, App. C.3 / Table 7] Sparse poisoning still gets through: 5% mix ASR 29.6%, 1% mix 16.3% (Qwen).

### 2. Abandoned, unstable, collapsing, over-refusing or reward-hacked
- [AUTHOR-REPORTED, §4.1] Constrained SFT (Qi et al. 2024), the single-reference baseline, is "sensitive to the localization of harmful content" because its beta_t is hand-set on the first few tokens; it fails when harmful tokens are at non-fixed positions (Table 1: 51.8-82.7% ASR on Gemma).
- [AUTHOR-REPORTED, §7] Degenerate case: if the harm twin collapses to the aligned model (pi_n = pi_p), "the contrastive signal disappears" and DR-SFT degenerates into Constrained SFT.
- [AUTHOR-REPORTED, App. B.2] Warmup was needed "only in the Gemma-3-4B-IT experiments to ensure more stable training of smaller models."
- [AUTHOR-REPORTED, App. C.4 / Table 8] Warmup strongly changes baseline outcomes (Lisa 60.9% to 17.8%, Constrained SFT 25.2% to 6.1% HEx-PHI ASR); text says 20% warmup, table says 10-step warmup (internal inconsistency).

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, §5.4] beta = 0.001 used everywhere (Table 4 above). LR 2e-5; 25 epochs on HEx-PHI for Llama-2 vs 3 for others (App. B.2). eta clipped to (-1, 1).
- [AUTHOR-REPORTED, App. C.2] Harm-twin data size: degradation below 500 samples.

### 4. Stated limits
- [AUTHOR-REPORTED, §7] Needs an extra negative reference model and logit preprocessing.
- [AUTHOR-REPORTED, §7] "If an attacker uses harm categories or attack patterns that are not covered by the dataset used to train the negative reference model, the contrast ... may be insufficient for reliable harmful-token localization."
- [AUTHOR-REPORTED, App. B.4] Safety judged by a GPT-4 judge only.

### 5. Independent re-evaluations
- [INDEPENDENT] None found in brief search (2026-10-17; paper posted 29 Jun 2026).

### 6. Implication for the harm-twin design
- Coupling 2: using a harm twin's likelihood ratio against the aligned model as the harm signal only works where the twin actually covers the harm (ASR rose from 14.3% to 19.8% as the twin's training set shrank, and coverage gaps are the stated failure), so the twin must be kept diverse and far from the main model, or the likelihood signal vanishes.

## Simko et al., Contrastive Representation Learning for LLM safety (arXiv 2506.11938, VERIFIED via arxiv.org/html fetch; EMNLP 2025)

Method: triplet loss on residual-stream representations (push new harmful representations away from old harmful ones and toward the harmful mean; keep benign close to the original and away from the harmful mean), plus "adversarial hard negative mining": a linear attack module inserted at a random layer, trained by NLL on harmful responses, re-created every 30 steps and updated one step every other defence step.

### 1. Ablated variants or components that hurt
- [AUTHOR-REPORTED, Table 2 / Tables 9-10] Adding the learned adversary (A4 "Triplet + Adv") made input-space robustness **worse** than the no-adversary triplet (A3): REINFORCE-GCG HarmBench ASR 0.00 to 3.75, StrongREJECT ASR 0.00 to 2.50, SR score 0.48 to 6.99; GCG SR score 0.43 to 1.36. App. A.4: "the A4 model performs worse on input-space attacks ... We believe this tradeoff is due to the harmful representations being used in training."
- [AUTHOR-REPORTED, Table 2 / Table 8] Adversary also raised embedding-attack HarmBench ASR: 65.85 (A3) to 75.61 (A4), while lowering StrongREJECT ASR 12.20 to 4.88. The two judges disagree in direction.
- [AUTHOR-REPORTED, Table 3 / Table 7] Across six embedding-attack configs, A4 worst-case HarmBench ASR 41.00 vs A3 32.00; mean 24.40 vs 23.83. The partial-loss variant A2 had a lower mean HarmBench ASR (22.67) than both full variants.
- [AUTHOR-REPORTED, Table 11 / App. A.4] A4 lowers out-of-distribution generalisation for several augmentations on the cosine metric (App. Table 20: e.g. Chinese 0.95 to 0.89, French 0.98 to 0.89, German 1.00 to 0.88).
- [AUTHOR-REPORTED, Table 8] A1 (circuit-breaker-like) embedding HB ASR 92.68%.

### 2. Abandoned, unstable, collapsing, over-refusing or reward-hacked
- [AUTHOR-REPORTED, App. A.2 / Fig. 4] The defence works by degeneration: attacked outputs become "The, / / / / ..." gibberish; "a harmless broken response that the simpler HarmBench judge incorrectly classifies as harmful."
- [AUTHOR-REPORTED, App. A.5 / Table 14] Representation-engineering defences "break in generation instead of producing proper refusals": baseline non-refusal (substring) rate 97% for Triplet vs 4% for the base model.
- [AUTHOR-REPORTED, §6.3 / Table 4] RepBend (a related representation loss) collapses capability: GSM8K 75.36 to 49.05, TruthfulQA-gen 46.39 to 2.08 ("overfitting to the defense objective").
- [AUTHOR-REPORTED, App. A.6 / Table 15] Triplet on Zephyr-7B: HellaSwag 63.97 to 52.49.
- [AUTHOR-REPORTED, App. A.6] ReFAT reimplementation required changing the RFA attack to the mean of the last five tokens "to improve stability."

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App. A.3] alpha=0.5, beta=0.4, gamma(KL)=0.9, margins m_b=500, m_h=1500, 1100 steps, LR 1e-5, LoRA r=16; adversary re-fit every 30 steps on layers 0-20; harmful batch mix 70% original / 30% adversarial. No sweep reported.
- [AUTHOR-REPORTED, §6.2 / Table 3] Defence rankings flip with attack hyperparameters (RepBend SR ASR 20% in config 2 vs 2% in config 0; circuit breakers the reverse).
- [AUTHOR-REPORTED, Limitations] "the choice of hyperparameters and training strategy may not be optimal."

### 4. Stated limits
- [AUTHOR-REPORTED, Limitations / Table 17] Mistral-7B remains vulnerable: Triplet embedding HB ASR 97.56% (worse than circuit breakers' 85.37%), SR ASR 34.15%. Phi-3-Mini HB ASR 92.68% (Table 16).
- [AUTHOR-REPORTED, Limitations] Evaluated on 80 HarmBench validation behaviours (GCG single attempt; embedding 41 behaviours); "not guaranteed to be robust to all, especially to attacks in the representation space"; may give "incoherent and ineligible behavior if the model misinterprets benign inputs as harmful."
- [AUTHOR-REPORTED, §6.1] Judge dependence: HarmBench judge consistently higher ASR than StrongREJECT.

### 5. Independent re-evaluations
- [INDEPENDENT] None found in brief search (2026-10-17).

### 6. Implication for the harm-twin design
- Coupling 1: a periodically re-fitted learned adversary as negative source traded one attack class for another (input-space ASR 0 to 3.75%, embedding HB ASR 65.9 to 75.6%), so the twin's negatives should be mixed with a fixed pool of ordinary harmful data, and success must be judged by more than one harm judge because gibberish outputs are scored as harmful by one and safe by another.


## Token Buncher (arXiv 2508.20697, VERIFIED via arxiv.org/html fetch)

What it is: the paper shows that "Harmful-RL" (GRPO against a harmfulness reward model) breaks safety alignment better than harmful SFT. Its defence, TokenBuncher, uses GRPO with negative entropy as the reward on harmful queries, a KL reward on benign queries, and a "Token Noiser" cross-entropy term that spreads random mass over the tokens other than the top one. The two objectives alternate step by step.

**Attack settings (the RL harmful fine-tune) [AUTHOR-REPORTED, §4.1]:** GRPO with a deberta-v3 harmfulness reward model. AdamW at lr 1e-6 (the verl default), batch 32, micro-batch 4, K=4 rollouts, **KL coefficient beta = 1e-3**, temperature 1.0, top-p 1, top-k -1. Trained on 4,000 unsafe BeaverTails queries; no harmful answers are needed. "We find that fewer than 100 Harmful-RL steps are sufficient to bypass most built-in safety alignments." Runs took 40-55 minutes on 2 H200s. The defence itself was trained with the same beta=0.001, Token Noiser lambda=0.01, 200 steps, lr 1e-6 [§6.1].

### 1. Ablations / variants that hurt
- [AUTHOR-REPORTED, §5.2 Table 2] **The strawman: plain entropy minimisation on harmful prompts (DEM).** After GRPO Harmful-RL, the harmful score (HS) on HarmBench was 36.50 (3B) and 40.50 (7B), against 45.75 and 45.00 for the undefended base. On StrongREJECT it was 49.52 and 42.49 against 75.52 and 78.59. WMDP-evil accuracy stayed high (40.45 and 56.44 against 49.98 and 65.34). The authors call the effect "quite limited". Their diagnosis is that low entropy learned on the defender's prompt set "does not generalize to all of D_adv": a non-negligible fraction of rollouts still get positive (harmful) reward, and RL amplifies them (Fig. 2).
- [AUTHOR-REPORTED, §5.5 Fig. 4] **Why DEM fails mechanically.** Minimising entropy suppresses the *total* mass on the non-top tokens but "preserves their relative distribution structure... once the suppression on y* is weakened, harmful tokens easily regain dominance."
- [AUTHOR-REPORTED, App. G.2 Table 11] **Removing the Token Noiser (TB w/o TN).** It looks safe under a standard attack (HS 1.25 / 2.24), but WMDP-evil stays at 46.67 against 7.96 with TN. Under a **reverse-entropy adaptive attack** without TN, HS is **44.75 / 71.88**, which is fully broken (TB with TN: 0.20 / 1.60).
- [AUTHOR-REPORTED, App. G.2 Table 11] The Token Noiser's protection works by **deliberately collapsing capability under attack**. Post-attack GSM8K falls 89.0 -> 68.8, MATH 73.6 -> 54.2 and MMLU-pro 57.2 -> 45.1 (reverse entropy: GSM8K 58.4). The authors call this "intentional". The Fig. 3 caption says: "the same attack amplifies the injected randomness, producing incoherent gibberish."
- [AUTHOR-REPORTED, §6.2 Table 3, App. G.3] **Baseline defences all fail against Harmful-RL.** RepNoise, CTRAP, Booster and SEAM (the SFT-era defences) leave post-attack HarmBench HS at about 26-65% across PPO/GRPO/RLOO/REINFORCE++. Example, Qwen2.5-7B under GRPO: RepNoise 48.50, CTRAP 34.50, Booster 38.25, SEAM 42.50, against 45.00 undefended. "After no more than 50 training steps, the protection offered by these defenses is all gone."
- [AUTHOR-REPORTED, §6.3 Table 7] **Over-refusal from the baseline defences (XSTest compliance on safe prompts).** Base: 90.8 / 98.4 / 96.8. RepNoise: 49.2 / 44.4 / 45.6. SEAM: 56.0 / 59.6 / 17.6 (Ministral). Booster: 75.6 / 75.6 / 66.8. TokenBuncher: 88.4 / 96.0 / 93.6.
- [AUTHOR-REPORTED, §6.4 Table 8] **Out-of-distribution attack data partly breaks TokenBuncher.** On Sorry-bench OOD queries, HS is 13.50 / 14.06, against about 1% in-distribution. Adding 10% OOD data to the defender's set and training 100 more steps restores it to 0.50 / 0.00.

### 2. Abandoned / unstable / collapse / reward-hacked
- [AUTHOR-REPORTED, App. F] **Reward hacking inside Harmful-RL.** "We observe that reward hacking... also manifests in Harmful-RL. In extreme cases, the model may learn to generate responses that the reward model scores highly but are judged as non-dangerous by humans." For that reason they evaluate with a model other than the reward model (BeaverDam-7B plus a Llama-3.1-70B judge).
- [AUTHOR-REPORTED, §5.6] **Abandoned scheduling choices.** A weighted sum of the two losses was avoided as "sensitive" to its hyperparameters, and "long-cycle alternating could lead to catastrophic forgetting". They chose a step-by-step interleave after a warm-up of E/8 steps.
- [AUTHOR-REPORTED, §4.1] **Harmful SFT collapse.** "For the Mistral model, we observe that 5e-5 leads to severe model collapse", so 2e-5 was used. SFT at lr 1e-5 cut Qwen-3B GSM8K from 81.0 to 36.4 (Table 1). RL kept it at 78.8 with a higher HarmBench HS (45.75 vs 38.00).
- [AUTHOR-REPORTED, §6.4] Attacker temperature: "as the temperature increases from 1.1 to 1.5, the model's harmfulness becomes more pronounced" (still low in absolute terms, at most 3.12% HS in Table 8), and "when the temperature reaches 1.7, we begin to observe random characters appearing in the sampled outputs."

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, §4.3 Table 1] **Harmful SFT is lr-bound.** Large lr breaks safety but destroys capability, and small lr (1e-6) does neither well (Qwen-7B HarmBench HS 22.75 at 1e-6 vs 44.00 at 5e-5). Harmful-RL at lr 1e-6 with beta 1e-3 dominates both (Takeaway 3).
- [AUTHOR-REPORTED, App. D.2 Table 9] **LoRA attacker.** Harmful-RL with LoRA still beats LoRA-SFT (HarmBench HS 45.00 vs 41.00), but full fine-tuning is "the most effective method for RL-based attacks".
- [AUTHOR-REPORTED, §6.4 Table 8] Attacker temperature (1.1-1.7), group size K (8, 16) and steps (200, 500) did not raise post-attack HS above 3.12%. The **data distribution** was the sensitive axis (OOD 13.5-14.1%).
- Sensitivity to the defence's own lambda (Token Noiser) or beta is **not reported**.

### 4. Stated limits [AUTHOR-REPORTED, §8]
- "Our defense performance is bounded by the diversity of auxiliary data distribution."
- The defence depends on SFT defences against an SFT-then-RL pipeline, and "current Harmful-SFT approaches are not perfect".
- "This work is a first attempt... should not be viewed as a 'silver bullet'... requires continuous co-evolution of defense strategies alongside adaptive adversaries."

### 5. Independent re-evaluations
- [INDEPENDENT, brief WebSearch 2026-10-17] None found in a brief search. The paper has several arXiv revisions, so numbers may differ between versions.

### 6. Implication for the harm twin
- For **coupling 2**: an RL twin with a small KL (1e-3) breaks a safety-aligned twin in under 100 steps without harmful targets, so the twin will become harmful fast. But its reward is gameable ("reward hacking... judged as non-dangerous by humans"). Entropy-collapsing the main model on the twin's prompts does not generalise to off-distribution harmful rollouts. Expect the twin to find the OOD region (here 13-14% HS) unless the twin's prompt pool is broad.

## AntiDote (arXiv 2509.08000, VERIFIED via arxiv.org/html fetch)

Fetch note: this time the arXiv HTML page (v1) came through in full with curl, with no content-filter refusal. The PDF was not needed.

What it is: a bi-level game in which a hypernetwork, conditioned on the defender's live activations, produces an adversarial LoRA patch. The patch is trained by DPO to prefer the harmful over the safe response. The defender LoRA is then trained to undo the patch, with the capability loss computed on the clean, unpatched model. The two steps alternate on a k:k schedule.

### 1. Ablations / variants that hurt
- [AUTHOR-REPORTED, App. A.1 Table 4] **A static adversary (prompt embedding instead of live activations) collapses.** HS goes 5.7 -> 33.1 (Llama-3.2-3B), 9.1 -> 31.5 (Gemma-3-12B) and 9.9 -> 34.8 (Gemma-3-27B), with FA nearly unchanged. Quote: "The Static adversary is blind to the defender's evolution; its attacks become 'stale'."
- [AUTHOR-REPORTED, App. A.2 Table 5] **Computing the capability loss on the attacked model ("Coupled") costs utility.** FA goes 90.8 -> 85.4 (3B), 93.9 -> 89.1 (12B) and 95.3 -> 90.2 (27B). HS changes little (8.5 -> 9.1, 5.9 -> 6.4, 5.1 -> 5.5). The authors' account is a "confounded gradient".
- [AUTHOR-REPORTED, §4 "Red-teaming"] **Attacks where AntiDote loses to simpler baselines.** On Hypothetical Framing (Adv 37) and Distractor Instructions (Adv 17), Booster and TAR do better. The authors' explanation: "AntiDote's hypernetwork, which looks for anomalous activation patterns, is effectively 'fooled' by the seemingly normal internal state."
- [AUTHOR-REPORTED, App. E Table 6] **Clean setting (p=0).** TAR's HS is lower (1.10 vs 2.10); the authors suggest that constant adversarial training leaves a "persistent 'defensive posture'... may not be as optimally aligned... in a zero-threat environment."
- [AUTHOR-REPORTED, App. E Table 6] **Still substantially harmful at p=0.2.** AntiDote's HS is 20.10 at a 20% harmful fine-tune ratio (SFT 60.50, Booster 24.90). HS rises from 4.50 at p=0.05 to 20.10.

### 2. Abandoned / unstable / collapse
- [AUTHOR-REPORTED, §3.4] The defender lr was kept "more conservative" (3e-5 vs adversary 2e-4) "to ensure stable convergence". This implies instability at higher defender lr, but no run is shown.
- [AUTHOR-REPORTED, §2.3] The capability loss was decoupled to give "a stable, stationary learning target"; an unstable coupled variant is implied. No divergence curves are reported.
- Not reported: oscillation, mode collapse or reward hacking of the adversary.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, §2.3] lambda = 0.8 (CE) and beta = 0.3 (KL to the base model), "a choice validated in our ablation studies (Section 5.5)". **No Section 5.5 exists** in the HTML (Section 5 is the Conclusion), and no lambda/beta sweep appears in the appendix. Treat the sensitivity as unreported.
- Neither the interleave block size k nor the epoch count (8) is ablated.

### 4. Stated limits [AUTHOR-REPORTED, §5]
- "Key challenges remain, such as extending this dynamic defense to novel attack classes beyond our extensive test suite (see Appendix for a full discussion of limitations)." **No limitations appendix is present** in the HTML.
- Threat-model gap (my reading, not the authors'): the fine-tuning attack evaluated is SFT on a 20:80 harmful:benign mix. The attack's lr, steps and optimizer are not stated, and there is no RL attacker.
- Internal inconsistency (my reading): the "Full" HS values differ between Table 4 (5.7 / 9.1 / 9.9) and Table 5 (8.5 / 5.9 / 5.1) for the same models.

### 5. Independent re-evaluations
- [INDEPENDENT, brief WebSearch 2026-10-17] None found in a brief search (the paper is in AAAI proceedings). Beware the unrelated, similarly named "Antidote" (a post-fine-tuning defence by a different group).

### 6. Implication for the harm twin
- For **coupling 1/2**: the twin must be **conditioned on the current main model**. A frozen or stale twin is 3-5x less useful (Table 4). Compute utility or anchor losses on the clean main model, not on the main model while it is being attacked. Expect a twin that targets internal anomalies to miss attacks that look benign on the inside.

## TAR: Tamper-Resistant Safeguards (arXiv 2408.00761, VERIFIED via arxiv.org/html fetch)

What it is: a meta-learning (first-order MAML-style) defence. An inner loop simulates K=64-step SFT attacks, and an outer loop minimises a tamper-resistance loss L_TR evaluated along the attack trajectory, plus a retain loss. L_TR is negative entropy for knowledge restriction and DPO for refusal.

### 1. Ablations / variants that hurt
- [AUTHOR-REPORTED, §4.2 Fig. 3, App. C.3] **Rejected loss: negated cross-entropy of the adversary.** "(1) Maximizing a cross-entropy loss can cause divergence; (2)... the model learns to explode the adversary's loss for the first few inner loop steps, while loss at later steps remains low." With the CE loss, "the adversary is only affected earlier in its trajectory and quickly recovers." The authors swapped it for negative entropy.
- [AUTHOR-REPORTED, App. D.4 Table 5] **Removing adversarial training (Random Mapping only).** Post-attack forget accuracy is 61.6, against 35.2 for TAR (no defence: 70.5). **Removing the initial safeguard**: pre-attack forget is 47.3 against 28.1.
- [AUTHOR-REPORTED, App. D.4 Table 6] **Tamper-resistance weight.** lambda_TR = 1.0 gives post-attack forget 39.9 (worse) with retain MMLU 62.5. lambda_TR = 4.0 gives 35.2 but MMLU falls to **54.7** (undefended 67.3). There is a direct robustness-utility trade.
- [AUTHOR-REPORTED, App. D.3 Fig. 7] **Shorter inner loops (K = 8, 16, 32) give shorter, lower loss plateaus at test time.** Robustness is bought with inner-loop length.
- [AUTHOR-REPORTED, App. D.2 Table 4] Earlier versions without the R->F (retain-then-forget SFT) train-time adversary were broken by that adversary. Adding it improved the 500-step R->F attacks by about 30 points, which is patching one attack at a time.
- [AUTHOR-REPORTED, §5.2 Table 2] **Refusal setting.** TAR's post-attack HarmBench ASR is still **63.9%** (the text says 61.7), against 72.5 for refusal-trained Llama-3-8B. Pre-attack MT-Bench is 6.3 against 8.1. Train-time inner-loop DPO win-rate goes "completely flat near... 1.0" (App. D.5, Fig. 8), so the train-time objective was satisfied while the test-time attack still mostly succeeded (my inference from the two numbers).

### 2. Abandoned / unstable / divergent
- [AUTHOR-REPORTED, App. F, Table 12] **MLAC-AR** (negating the adversary's CE): "this approach diverges in performance across a variety of hyperparameters." Even clamped at log(vocab), it "does not maintain sufficient benign capabilities performance nor uniform tamper-resistance".
- [AUTHOR-REPORTED, App. F, Table 12] **SOPHON-AR** (alternating retain / suppression with inverse CE): "despite heavy tuning, SOPHON-AR does not converge well enough to yield a usable Pre-Attack model" (retain MMLU 23.3-24.4, which is chance).
- [AUTHOR-REPORTED, App. A] When TAR holds, "the post-attack retain accuracy is also low": the model is degraded rather than robustly safe.
- [AUTHOR-REPORTED, App. C.1] **Data contamination.** The retain set (Magpie-Align) contained forget-set content, which caused "an unintended input-space vulnerability" in v1; v2 was retrained.
- [AUTHOR-REPORTED, App. C.1 Fig. 5] **Seed fragility, after Qi et al.** A HuggingFace sampler bug meant the test-time shuffles were a subset of the train-time shuffles. Once fixed, robustness at 1,000 steps "can vary with different dataloader shuffling orders"; only about 500 steps is consistent.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Apps. D.3, D.4] Inner-loop length K and lambda_TR, as above. Refusal training used lambda_TR = 0.1, ScheduleFree AdamW, lr 6e-5, N = 100 outer steps [§5.2].
- [AUTHOR-REPORTED, per Qi et al. §3.3] A LoRA attack with the same hyperparameters (Orig-Config 4) "largely breaks" TAR. This is reported in the original but not emphasised.

### 4. Stated limits [AUTHOR-REPORTED, App. A]
- The defence was only evaluated against SFT attacks.
- Scaling cost.
- Low post-attack retain accuracy means benign users' data must not be contaminated with forget-set data.
- "It can eventually be circumvented."

### 5. Independent re-evaluations
- [INDEPENDENT, Qi et al. 2412.07097 §3.1-3.5] Five random seeds of the same attacks recover WMDP-Bio above 60%. Adding 100 warmup steps and cosine decay (New-Config 1) breaks it consistently. Fine-tuning on the **retain** set alone recovers the knowledge. A chat-template variant shows v1 at above 45% WMDP-Bio. v2 garbles benign biology answers (perplexity 16.4 vs 3.8) and collapses on GSM8K (1.5 / 4.8 vs 74.4). Details are in the next section.
- [INDEPENDENT, Che et al. arXiv 2502.05209 §4, Table 2] TAR was the only unlearning method resistant to few-shot fine-tuning ("only 1 out of the 9 fine-tuning attacks" raised WMDP-Bio by more than 10 points). But "TAR models had low unlearning scores due to poor general utility, which renders their robustness to fine-tuning unremarkable". The authors say TAR struggles "principally due to poor fluency".
- [INDEPENDENT, web search result only, not read] An MIT thesis (Feb 2025, dspace 1721.1/159116) reportedly finds the same attack succeeds under some initialisations and fails under others. The TAMPERTEST workshop paper (ICLR 2026) benchmarks TAR. Neither was read.

### 6. Implication for the harm twin
- For **coupling 2**: making the main model maximise loss or minimise likelihood on the twin's harmful outputs diverges, or only defends the first few steps (the negated-CE lesson). Entropy- or DPO-style bounded objectives were needed. Robustness bought this way often turns out to be capability collapse that the next attacker's optimiser settings get around.

## On Evaluating the Durability of Safeguards for Open-Weight LLMs, Qi et al. (arXiv 2412.07097, VERIFIED via arxiv.org/html fetch)

What it is: an independent re-evaluation of RepNoise (Rosati et al. 2024) and TAR (Tamirisa et al. 2024) that shows how small evaluation choices produce false durability claims. Everything below is INDEPENDENT evidence about RepNoise and TAR; the paper is itself a negative-results paper.

### 1. Claims that broke, and the change that broke each
- [INDEPENDENT, §3.1 Fig. 1a] **RepNoise, seeds.** Same checkpoint, hyperparameters, codebase, data and classifier; the only change was turning on data shuffling ("The original codebase disabled data shuffling and used greedy decoding"). At lr 6e-5 and 8e-5, the worst of 5 seeds is "almost the same as those obtained on the original Llama-2-7B-Chat model".
- [INDEPENDENT, §3.1 Fig. 2] **TAR, seeds.** Three original configurations (lr 2e-5, AdamW, 1,000 steps, batch 32/64), each over 5 seeds. The best seed recovers **more than 60%** WMDP-Bio for all three, close to undefended Llama-3-8B-Instruct. The paper had reported Orig-Config 1/3 as fully unsuccessful and Config 2 at about 45%.
- [INDEPENDENT, §3.2 Figs. 1b, 3] **Trainer implementation.** Switching from the authors' custom trainer to the HuggingFace SFTTrainer (same hyperparameters) makes attacks "more successful and consistent". RepNoise then matches the undefended model's vulnerability.
- [INDEPENDENT, §3.3, §4 Fig. 6, Table 1] **LR schedule.** New-Config 1 differs from Orig-Config 3 only in 100 warmup steps (instead of 30) and cosine decay (instead of constant). It "can always largely recover" WMDP accuracy. The mechanism: TAR's effect relies on an initial-step gradient explosion and loss spike. "TAR's effectiveness may rely on the instability of fine-tuning in the first few gradient steps."
- [INDEPENDENT, §3.3] **LoRA.** Orig-Config 4 (PEFT, same hyperparameters) largely breaks TAR. The original paper reported this but did not reflect it in its security claims.
- [INDEPENDENT, §3.4 Fig. 4] **Prompt template.** Wrapping WMDP in the Llama-3 chat template and extracting answers with GPT-4o or humans puts TAR-Bio-v1 "consistently above 45%", against the 25% reported. v2 (after the contamination fix) no longer spikes. The BOS token also matters (App. F.2 Table 7).
- [INDEPENDENT, §4 Fig. 3, Fig. 7] **"Unlearning" is not removal.** Fine-tuning TAR only on its **retain** set (New-Config 2) largely recovers WMDP-Bio. RepNoise recovers HarmfulQA after **100 benign** examples (AOA identity shift, or Alpaca-salient).
- [INDEPENDENT, §3.5 Tables 2-3] **Side effects missed.** On TAR-Bio-v2, BenignBio GPT-2 perplexity is 16.4 against 3.8 undefended. GSM8K is 1.5 (v1) and 4.8 (v2) against 74.4; HumanEval 4.0 / 21.1 against 54.7; BBH 3.4 / 24.9 against 56.1. **Harmful compliance rises**: Sorry-Bench 58.3 (v1) and 32.2 (v2) against 22.2; HEx-PHI 29.9 / 18.8 against 4.3. Footnote: "Huang et al. (2024a) also observed mode collapse phenomenon from the TAR-checkpoints."
- [INDEPENDENT, App. E.1.2] **RepNoise implementation bugs.** The attack loss was computed on every token including padding (labels = input_ids). The train, attack and test sets were "highly overlapped"; the disjoint re-test is in App. F.3.

### 2. Abandoned / unstable
- [INDEPENDENT, §4 Fig. 6] Under the original configurations, the attack on TAR is itself unstable: a loss spike and huge gradient norms in the first steps, "high variance and can occasionally fail". Stabilising the attacker removes the defence.

### 3. Hyperparameter sensitivities flagged
- The attacker's random seed, trainer, warmup length, LR schedule, LoRA vs full fine-tuning, prompt template and BOS token, and the evaluation dataset split [INDEPENDENT, §§3.1-3.4, App. F]. RepNoise's own evaluation was limited to lr in {3e-5, 6e-5, 8e-5}, and its authors note that high lr breaks it [INDEPENDENT, §2.2].

### 4. Stated limits [AUTHOR-REPORTED by Qi et al., §3 intro, §4]
- "Our examination is limited to the two particular methods". The pitfalls "might also occur" for others.
- They "constrain our evaluation mostly on cases the original paper claims to defend against".

### 5. Later re-evaluations of this paper
- Not applicable; Qi et al. is itself the re-evaluation. TAR's authors corroborated the seed finding (TAR App. C.1, Fig. 5).

### 6. Implication for the harm twin
- For **coupling 1 and 2** alike, judge the hardened main model with **many attacker seeds**, a stabilised optimiser (warmup plus cosine), LoRA and full fine-tuning, several prompt templates, and benign-utility and over-refusal probes. "Robustness" that rests on gibberish or on early-step gradient explosions is not safety. A rise in general harmful compliance after targeted unlikelihood training is a known side effect.


## Negative Preference Optimization, Zhang et al. 2024 (arXiv 2404.05868, VERIFIED via arxiv.org/html fetch)

The method: NPO is a DPO-style loss that uses only negatives, (2/beta) E log(1 + (pi_theta/pi_ref)^beta), applied to a forget set. It is bounded below, and as beta -> 0 it reduces to gradient ascent (GA).

### 1. Ablations that hurt
- [AUTHOR-REPORTED, §5.2 "The role of retain loss", Fig. 8, Forget10 on TOFU, Llama2-7b-chat] Retain-loss weight was swept from 0 to 5. Forget quality rises as the weight goes from 0 to 2, then falls from 2 to 5: "further increasing the weight of the retain loss (e.g., from 2 to 5 ...) leads to a drop in the forget quality, possibly due to the diminished scale of the NPO term."
- [AUTHOR-REPORTED, §5, Fig. 5] Vanilla NPO without the retain term is not the best variant. "In Forget10, NPO+RT stands out as the only method that maintains meaningful forget quality". In Forget50, "vanilla NPO+RT achieves a forget quality around 10^-3" (below the 0.05 threshold) until the retain weights are re-tuned (§5.3, Fig. 10): retain coefficient 5.0 for Forget50 and 12.0 for Forget90.
- [AUTHOR-REPORTED, §5.2, Fig. 9; App. B, Fig. 11] A large forget-set KL hurts. "a larger Forget KL is not necessarily advantageous ... Forget KL may not be a suitable objective function to maximize". In the synthetic study, "an excessively large forget KL (as in GA, GA+RT after 1200 steps) or an excessively small forget KL (as in IDK+RT) may deteriorate the unlearning performance."

### 2. Abandoned, unstable or collapsing
- [AUTHOR-REPORTED, §2.1, Fig. 2, Forget05, 10 epochs] Plain GA collapses: "the model utility quickly drops to zero, and the forget quality improves temporarily for a very short time horizon before quickly dropping too ... after which the model generates gibberish outputs."
- [AUTHOR-REPORTED, §2.1, Theorem 2] The authors trace the collapse to GA's unbounded loss. On already-unlearned samples its gradient stays roughly x_i ("constant scale (not diminishing)"), so GA diverges linearly in the step count, while NPO diverges logarithmically.
- [AUTHOR-REPORTED, §5.2, Fig. 6] In Forget01, "both GA and GA+RT attain their highest forget quality at the sixth gradient step, but their performance subsequently declines drastically". In Forget05/10 they peak "still below 0.05, before rapidly diminishing to an exponentially small magnitude." Early stopping is unreliable: "the stopping time can be highly instance-dependent and does not follow a discernible pattern."
- [AUTHOR-REPORTED, §4 synthetic, Fig. 4] The retain distances of GA and GA+RT diverge after roughly 1200 steps.
- [AUTHOR-REPORTED, §5.2 "Generated outputs"] Two degenerate output modes: "GA-based methods tend to output repeated words or gibberish sentences with unreasonably low diversity", and "IDK and DPO-based methods tend to show excessive ignorance by outputting 'I don't know' ... to commonsense questions". The second is an over-refusal analogue.
- [AUTHOR-REPORTED, §3 + §6] NPO only slows collapse; it does not remove it. The authors call NPO steps "towards addressing the catastrophic collapse issue", and they state that "some kind of divergent behavior is necessary and perhaps unavoidable" (§2.1).

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Prop. 1, §3] beta controls a continuum: as beta -> 0, NPO = GA, so a small beta brings back GA's linear divergence.
- [AUTHOR-REPORTED, App. D, Table 1, synthetic] Both beta and the learning rate came from grid search, and the chosen values differ by regime: NPO beta = 1 (alpha=1) vs 10 (alpha=0); DPO+RT beta = 0.1 vs 5; NPO lr 5e-3 vs 5e-2; GA lr 5e-4 vs 1e-4. Learning rates were "chosen ... so that the training remains stable within 2000 steps". The authors note that the learning rate "may affect the number of steps needed".
- [AUTHOR-REPORTED, §5.1] TOFU setup: AdamW, lr 1e-5, effective batch 32, 10 unlearning epochs. Results are plotted at the epoch of peak forget quality (Fig. 5 caption), so the headline points are selected post hoc by epoch.
- [AUTHOR-REPORTED, §5.2, Fig. 8] The retain weight is non-monotone in forget quality (best near 2 on Forget10). It needs re-tuning for each forget-set size (5.0 for Forget50, 12.0 for Forget90).
- [AUTHOR-REPORTED, App. D.1.3] The DPO and KTO baselines used beta = 0.1 throughout and were not tuned per setting.

### 4. Stated limits
- [AUTHOR-REPORTED, §6] Tested only on TOFU (plus synthetic data). Future work includes "testing NPO on more datasets or harder scenarios (such as with adversarial prompts)". No adversarial or relearning robustness was tested.
- [AUTHOR-REPORTED, §2] The authors state that a model that outputs "I don't know" can score well on forgetting while failing at approximate unlearning, so the forget metric alone can be gamed.

### 5. Independent re-evaluations
- [INDEPENDENT, Zhang et al., "Catastrophic Failure of LLM Unlearning via Quantization", arXiv 2410.16454, ICLR 2025, §4 Table 1, MUSE] Vanilla NPO on NEWS reaches retain KnowMem 0.0 (total utility loss, as with GA). 4-bit RTN quantization brings back forget-set VerMem 0.0 -> 16.2 and KnowMem 0.0 -> 25.4. For NPO_GDR, KnowMem goes 46.1 -> 51.4 and VerMem 0.3 -> 33.2. For methods with utility constraints overall, the model "retains an average of 21% of the intended forgotten knowledge in full precision, which significantly increases to 83% after 4-bit quantization."
- [INDEPENDENT, Hu et al., "Jogging the Memory of Unlearned LLMs ..." / "Unlearning or Obfuscating?", arXiv 2406.13356, Table 2, zephyr-7b-beta, WMDP] Benign relearning reverses NPO: the forget-set judge score is 5.92 before unlearning, 1 after NPO unlearning, and 5.08 after relearning (GA: 1.67 -> 5.2). The authors report "regardless of the objective used for unlearning, vanilla finetuning with standard negative log likelihood loss is sufficient to recover the unlearned content."
- [INDEPENDENT, Fan et al., SimNPO, arXiv 2410.07163, §4] The paper identifies a "reference model bias": "NPO may blindly increase the deviation from theta_ref ... causing 'easy' examples to move unnecessarily far beyond the unlearning boundary", while hard (strongly memorised) examples stay under-unlearned.

### 6. Implication for the harm twin
- Coupling 2 (the main model lowers the likelihood of the twin's output) is a GA- or NPO-type update on samples that keep changing. GA's linear divergence and instance-dependent collapse time say to use a bounded, NPO-style loss with a retain or KL anchor, and to expect easy twin samples to soak up the update budget (reference bias). Lowered likelihood is not removal: quantization and relearning undo it.

## Large Language Model Unlearning, Yao, Xu, Liu 2023 (arXiv 2310.10683, VERIFIED via arxiv.org/html fetch)

The method: gradient ascent on harmful (PKU-SafeRLHF) responses, plus a "random mismatch" loss (prompts paired with random normal responses) and a KL-to-original term on normal prompts. Only negatives are used.

### 1. Ablations that hurt
- [AUTHOR-REPORTED, §6 Table 3, OPT-1.3B] Plain GA destroys output diversity on harmful prompts: 0.118 vs 0.787 for the original model. Fluency is "NM" (not meaningful), and outputs are "mostly just whitespaces or nonsensical strings". Normal-prompt utility reward drops from -3.599 to -3.838, and similarity to the original from -0.778 to -1.034.
- [AUTHOR-REPORTED, §6 Table 3] Random mismatch does not reliably rescue fluency. On OPT-2.7B and Llama 2 (7B), GA+Mismatch fluency is still "NM". On Llama 2, GA+Mismatch similarity to the original is -1.319, against -0.421 for the original and -0.689 for plain GA. On Llama 2, plain GA's utility reward is -4.252 vs -3.338 for the original.
- [AUTHOR-REPORTED, §6 Table 3] On OPT-1.3B, GA+Mismatch has a higher harmful rate than plain GA: 6% vs 1% on unlearned prompts and 7% vs 3% on unseen ones.
- [AUTHOR-REPORTED, §9.2 Table 7] Replacing random outputs with a refusal template ("I can't assist it.") "does not show any significant difference in the metrics". The template adds nothing: GA+Template has similarity to the original of -1.077 (worse than GA+Mismatch at -0.943) and utility reward -3.257 (vs -2.982 for GA+Mismatch).
- [AUTHOR-REPORTED, §3.2 Solution 2] Cross-entropy on normal data alone "does not maintain the normal performance well". The authors switched to a KL-to-original term.

### 2. Abandoned, unstable or collapsing
- [AUTHOR-REPORTED, §3.2 Key Difference 1, Fig. 2, Table 1] Forget-set loss is a misleading progress signal: "even when the loss on the unlearned samples rises to as high as 60+ after unlearning for ~200 batches ... the LLM still outputs harmful responses". The model stops only after roughly 1000 batches. The authors adopt "continue to unlearn ... for 3x-5x more batches" after the loss explodes (§4).
- [AUTHOR-REPORTED, §3.2 Key Difference 2, Table 2] After roughly 1000 GA batches, "both the unlearning LLMs output nonsense on both harmful and normal (TruthfulQA) prompts" ("Failed case"). "preserving performance on normal samples is generally harder to achieve than forgetting."
- [AUTHOR-REPORTED, §3.2 Key Difference 3] Format shortcut: when the normal data and the forget data differ in format, "the unlearned LLM can learn a shortcut that decides what to output by the format of the prompt only, and therefore does not truly unlearn the concept."
- [AUTHOR-REPORTED, §6] The authors accept degenerate output as the success state: "generating nonsensical but non-harmful answers is what we expect".
- [AUTHOR-REPORTED, App. A] GA alone stalls when forget-sample gradients are small: "its effect is limited when ||grad l|| -> 0". The mismatch term was added for this reason.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App. B.1 Table 8] Loss weights and learning rates were tuned separately for each model, over a wide range. GA epsilon1 (forget weight) is 0.5 on OPT-1.3B, 0.1 on OPT-2.7B and 0.05 on Llama 2. GA+Mismatch epsilon1 is 0.5 / 2 / 2. Learning rates are 2e-5 / 2e-6 / 2e-4 (LoRA for Llama 2). All runs use 1K batches with batch size 1-2. Per-model weights shifting tenfold indicate high sensitivity.
- [AUTHOR-REPORTED, §9.2] The template variant required re-tuning epsilon1-3.
- [AUTHOR-REPORTED, §3.2 + §4] The number of steps matters: 200 batches is not enough even at loss 60+, while 1000 batches with GA alone yields nonsense on normal prompts.

### 4. Stated limits
- [AUTHOR-REPORTED, §10] There is no unified evaluation framework yet ("a reasonable future direction is to construct a unified and comprehensive evaluation framework"). Influence-function approaches with guarantees are left for future work. The method is scoped to "stop generating undesirable outputs rather than to output desirable responses" (§1, §9.1).
- [AUTHOR-REPORTED, §6 footnote 6] The judge is a PKU moderation model trained on the same PKU-SafeRLHF data that was unlearned. This judge and data overlap is stated in the paper but not discussed as a limitation.

### 5. Independent re-evaluations
- [INDEPENDENT, NPO paper arXiv 2404.05868 §2.1, §5.2] On TOFU, the Yao-style GA, GA+RT and GA+KL baselines collapse: utility goes to zero, outputs are gibberish, and forget quality on Forget05/10 stays below 0.05.
- [INDEPENDENT, arXiv 2410.16454 §4 Table 1] GA on MUSE "appears to achieve absolute forgetting even after 4-bit quantization; however, this is misleading as it results from a complete loss of model utility" (GA retain KnowMem 0.0 on NEWS). GA_GDR on NEWS: VerMem 0.0 -> 25.0 after 4-bit quantization.
- [INDEPENDENT, arXiv 2406.13356 §3, Table 2] GA-unlearned models are recovered by benign relearning: the WMDP judge score goes 1.67 -> 5.2 (5.92 before unlearning). In the text, a GA-unlearned model goes from 1.27 to 6.2.
- [INDEPENDENT, search snippets only, NOT read in full: arXiv 2502.19301 (ICLR 2025) describes GA as "severely susceptible" to excessive unlearning; arXiv 2510.19422 reports GA redistributes probability mass to rephrasings of the targets] Both are UNVERIFIED beyond search-result text.

### 6. Implication for the harm twin
- Coupling 1 (training away from twin outputs as negatives) with an unbounded GA-style loss will "succeed" by making the main model degenerate: whitespace and gibberish count as non-harmful. Forget-set loss is not a usable stopping signal. The anchor data must match the twin's prompt format, or the main model learns a format shortcut instead of the concept.

## When Autoregressive Consistency Hurts Safety Alignment, Lyu, Jia, Cai, Zhu 2026 (arXiv 2606.04168, VERIFIED via arxiv.org/html fetch)

The method: the paper argues that alignment gradients concentrate on early tokens, because "autoregressive consistency" makes later tokens carry little gradient (Prop. 2.1/2.2, for both SFT and DPO). It proposes adversarial safety alignment: insert harmful spans into refusals at k=32 random positions, select the position with the largest harmful-minus-refusal log-prob margin, and train SFT recovery to the refusal from that state.

### 1. Ablations that hurt
- [AUTHOR-REPORTED, §3.3 Table 1, Llama-2-7b-Chat, 3 runs] On the prefill attack, Insertion is worse than Prefill (deep-alignment) training: 25 tokens 6.5±1.8 vs 3.8±1.1; 50 tokens 11.0±2.7 vs 9.0±1.8; 75 tokens 13.5 vs 13.2; 100 tokens 21.2 vs 20.9.
- [AUTHOR-REPORTED, §3.3 Table 3, Gemma-2-9b-it] On the prefill attack, Insertion is worse than Prefill at every span length: 13.8 vs 7.1, 18.1 vs 13.5, 27.1 vs 22.0, 29.2 vs 22.6. Training against worst-case insertions lowered robustness to the attack that the deep-alignment baseline was trained on.
- [AUTHOR-REPORTED, §3.3 Table 3, Llama-3.1-8B] Insertion's random-insertion ASR rises to 10.4 at span 100 (vs 1.9 at 25), and its prefill ASR is 23.1 at span 100. Robustness falls off with span length beyond the training span (q = 50).
- [AUTHOR-REPORTED, §3.3 Table 1] Prefill (deep alignment, Qi et al. 2025) "transfers poorly to random insertion attack": 18.4-34.3% ASR vs 3.8-20.9% on matched prefill. Clean SFT stays at 76.6-85.8% ASR.

### 2. Abandoned, unstable or collapsing
- [AUTHOR-REPORTED, §2.2, Fig. 4] Existing aligned models (Llama-2-7B-Chat, Gemma-2-9b-it) have random-insertion ASR above 75%. "even 10 tokens harmful span inserted in an otherwise safe refusal output can dramatically redirect the generation."
- [AUTHOR-REPORTED, §3] Depth alone is called "an incomplete objective": "Training deeper may extend the defended horizon against prefix-like attacks, but it does not by itself address the underlying dynamical problem."
- [AUTHOR-REPORTED, §3.1] The objective is a surrogate: "Eq. (12) should be understood as a practical surrogate rather than a direct optimization of a formal trajectory-level measure".
- No instability, collapse or over-refusal is reported. No utility or over-refusal benchmark is reported at all: a utility loss on Alpaca (responses from the initial model) is included in training, but there are no MT-Bench, XSTest or similar numbers (App. C, §3.3).

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App. C Table 4] Settings were fixed, not swept: lambda = 0.5 (via 16 safety + 16 utility batch), lr 2e-5, 10 epochs, q = 50, k = 32, refusal bank m = 4. No ablation over k, m, q or lambda is reported. Sampling q from a distribution is "le[ft] ... to future work" (footnote 1).
- [AUTHOR-REPORTED, Theorem / App. B.4] The theory needs a learning-rate condition (Eq. 20) and smoothness assumptions for late-token gradients to stay O(sqrt(eps)).

### 4. Stated limits
- [AUTHOR-REPORTED, §3, §4] "the goal of this work is not to present a final defense recipe". It is presented as "an initial step".
- [AUTHOR-REPORTED, App. C "Evaluation of ASR"] ASR is keyword-based: an attack counts as a success if none of a set of refusal-start keywords appears in 50 continuation tokens. This measures absence of refusal, not harmfulness. Tab. 2 (GCG/PAIR/TAP/AutoDAN) uses the HarmBench classifier on 200 behaviours.
- [AUTHOR-REPORTED, App. C] The refusal bank is "intentionally ... small to avoid introducing significant additional computational overhead" (4 strings). Training uses 256 harmful instructions.

### 5. Independent re-evaluations
- None found in brief search (published 2 June 2026). A related, earlier paper (arXiv 2511.12155, AAAI'26, "Rethinking Deep Alignment Through The Lens Of Incomplete Learning") reaches a similar position-decay conclusion, but it is not a re-evaluation.

### 6. Implication for the harm twin
- For both couplings: SFT and DPO updates against twin responses mostly move the first few tokens (late-token gradients are suppressed). The main model's refusal can therefore be shallow. A twin that learns to start harm mid-response (an insertion-like strategy) will get past it. Selecting the worst continuation state for training helps on that attack but traded away some prefill robustness on 2 of 3 models, so the twin's attack distribution must be kept diverse rather than worst-case only.


## AdvPrefix (arXiv 2412.10321, VERIFIED via arxiv.org/html fetch; v2, 27 Dec 2025; NeurIPS 2025)

Method: replaces the fixed "Sure, here is [request]" jailbreak target with one or more model-specific prefixes. Each prefix is chosen by a weighted score of high prefilling ASR (attack success rate) and low initial NLL under the victim model.

### 1. Ablated variants or components that hurt
- **The original fixed-prefix objective is misspecified.** Lowering its loss does not raise nuanced ASR. "While direct refusals decrease, incomplete and unfaithful responses rise, keeping overall ASR low". Failure stays above 80% even at low loss, and is about 90% across all four victims. [AUTHOR-REPORTED, §3.2, Fig. 3 left/center]
- **Prefilling the exact target prefix does not fix it.** "While prefilling eliminates direct refusals, ASRs remain low due to persistent incomplete and unfaithful completions." So placing a harmful prefix with probability 1 still does not give a harmful completion. [AUTHOR-REPORTED, §3.2, Fig. 3 right]
- **Baseline ASR with the original objective (GCG, 20-token suffix).** Llama-2 13.0, Llama-3 12.8, Llama-3.1 16.8, Gemma-2 11.2. Gemma-2 gives 67.8% unfaithful responses. Llama-3 gives 22.1% incomplete and Llama-3.1 gives 16.8% incomplete. [AUTHOR-REPORTED, Table 1]
- **AdvPrefix's own failure cells:**
  - On Llama-3.1, the single-prefix variant raises unfaithful responses from 18.1 to 33.0 (40-token: 17.3 to 30.3).
  - On Llama-2, direct refusal stays around 70% (70.0 single and 68.0 multiple, vs 72.3 original).
  - Multi-prefix is not uniformly better than single-prefix: Llama-3 54.0 vs 54.6, Gemma-2 40.0 vs 42.0, and Llama-2 at 40 tokens 37.5 vs 38.6. [AUTHOR-REPORTED, Table 1]
- **Full-prompt optimisation with the original objective is inconsistent.** It "yields inconsistent ASR changes due to more incomplete and unfaithful responses". On Llama-3, AutoDAN with the original objective reaches 5.2% ASR, with 34.5% refusal and 28.3% incomplete. With AdvPrefix, Llama-3.1 unfaithful stays at 39.4 (39.0 before). [AUTHOR-REPORTED, §5.1, Table 2]
- **Elicited responses stay less harmful than an uncensored model.** Harmfulness win rate against an uncensored LLM is only 30-50% for AdvPrefix and about 10% for the original objective. "A win rate below 50% indicates that the jailbroken victim LLMs still cannot generate responses that are as harmful as the uncensored LLM." [AUTHOR-REPORTED, §5.2, Fig. 5 right]
- **Reasoning models.** "GCG with default prefixes cannot lower the loss in this case." [AUTHOR-REPORTED, App. D, Fig. 9]

### 2. Abandoned, unstable, degenerate or reward-hacked
- **Objective hacking.** The prefix objective also counts non-jailbroken continuations, so "it is prone to misspecification or objective hacking". [AUTHOR-REPORTED, §3.1]
- **Self-correction after a harmful prefix.**
  - Newer LLMs "often begin with the target prefix ('Sure, here is …') and then self-correct by giving incomplete or unfaithful responses".
  - Gemma-2 "almost always gives incomplete or unfaithful responses, and rarely directly refuses".
  - Llama-2 gives unfaithful responses 24% of the time and never gives incomplete ones.
  - Example incomplete response: "Here is a guide on how to manipulate someone…:\n\n**Disclaimer:** I cannot provide a guide… <eos>". [AUTHOR-REPORTED, App. A, Table 5]
- **Judges reward-hack on incomplete or unfaithful outputs.**
  - Existing judges overestimate ASR "from a 10% ground truth to nearly 40% on Gemma-2", by up to 30%.
  - Human agreement on Gemma-2: HarmBench 76.0 (F1 69.6), JailbreakBench 77.0 (68.9), StrongReject 82.5 (75.5). The authors' refined Llama-3.1-70B judge reaches only 86.0 (F1 74.6).
  - Ground truth is 800 hand-labelled pairs. [AUTHOR-REPORTED, §2, Fig. 2, Table 4]
- **Exposure bias.** Prior work blames "high loss on the first token". The authors add that "even after successfully eliciting the prefix, the model still fails to generate a complete and faithful response." [AUTHOR-REPORTED, App. B]
- **Why they did not use distillation or sampling objectives.** The output distribution is "hard to estimate especially when it has high entropy". Sampling-based objectives are inefficient, and lookahead predictors "can only predict a few tokens ahead … while identifying nuanced harmful responses often requires examining hundreds". [AUTHOR-REPORTED, App. A "Why still prefix-forcing?"]

### 3. Hyperparameter sensitivities
- **The two selection criteria conflict.** "Longer prefixes may have higher prefilling ASR but also higher NLL, causing optimization to fail." The weight must be tuned "to the optimization method's strength". They use a fixed weight of 20 on log-prefilling-ASR, 25 completions per prefix and 4 prefixes. [AUTHOR-REPORTED, §4.2, §5]
- **Prefix wording.** "Sure, here is" → "Here is" on Llama-3 lowers the final loss and moves ASR from 12.8% to 22.1%, so the rigid prefix is overconstrained. [AUTHOR-REPORTED, App. D, Fig. 8]
- **Binarisation threshold.** StrongReject's score is binarised at 0.6, the value that maximised F1 on their data, so the threshold is tuned to the dataset. [AUTHOR-REPORTED, App. C]
- **Tuning cost.** Candidate pool size, rejection rules and weighting are "included in our codebase", not in the paper. [AUTHOR-REPORTED, App. C]

### 4. Stated limits
- **Compute.** Prefix selection "requires evaluating many sampled responses, leading to a computational burden". It takes about 5 min per prefix on an A100, dominated by the 70B judge. [AUTHOR-REPORTED, §7 Limitations, §5.2]
- **Narrow, filtered data.** Only 50 AdvBench requests are used, with sensitive topics such as child abuse excluded. Cost-sensitive alignment "increases variance in jailbreak evaluations". [AUTHOR-REPORTED, App. C]
- **Harm ceiling.** "None of the models responded with materially harmful information that would not be found on the broader internet." [AUTHOR-REPORTED, §1 footnote]

### 5. Independent re-evaluations
- None found in a brief search. NeurIPS 2025 listings and a review aggregator surfaced; no paper reporting that AdvPrefix fails. [INDEPENDENT, WebSearch 2026-10-17]

### 6. Implication for the harm twin
- **Coupling 2:** a high main-model likelihood of the twin's harmful prefix is not harm. Prefilled harmful prefixes still end in refusals or disclaimers, and low NLL coexists with roughly 90% failure. A twin rewarded by main-model likelihood will hack toward high-likelihood, low-harm text: incomplete, disclaimer-laden or off-topic.
  - The reward must condition on a judged-harmful full completion, using a judge validated on incomplete and unfaithful cases.
  - Lowering prefix likelihood alone may target text the main model already self-corrects from.

## Self-ReSET (arXiv 2605.08936, VERIFIED via arxiv.org/html fetch)

Method: on-policy RL (DAPO) for reasoning models.
- A stream guard (Qwen3Guard-Stream) flags the model's own unsafe reasoning prefixes.
- The prefixes are truncated at the first unsafe token and stored in a FIFO replay buffer.
- Rollouts then restart from them, with a binary safe/non-refusal reward.

### 1. Ablated variants or components that hurt
- **No ablation of buffer, truncation point or priority sampling.** The paper has no ablation of buffer capacity, truncation at t*, priority sampling or the guard. Its only comparisons are against baselines. The truncation claim ("empirically improves the learnability of recovery") gives no numbers. [AUTHOR-REPORTED, §4.1.2; no ablation table]
- **Static-prefix recovery training (RECAP) gives diminishing returns on stronger models.** "As model reasoning capability increases, the resulting gains become less pronounced … RECAP's limited safety improvement on Qwen3-8B". Qwen3-8B with RECAP vs DAPO:
  - WJ 80.2 vs 82.8
  - HB 84.7 vs 89.0
  - Fortress 62.5 vs 62.7 [AUTHOR-REPORTED, §5.2, Table 1]
- **Self-ReSET's own losing cells:**
  - DS-Qwen-7B HarmBench 85.1, vs STAR-1 92.4, RECAP 86.9 and DAPO 86.1.
  - DS-Llama-8B StrongReject 98.0 vs DAPO 99.3.
  - Qwen3-8B XSTest compliance falls from 98.8 (base) to 96.4.
  - Qwen3-8B Fortress 77.2, so it remains about 23% unsafe. [AUTHOR-REPORTED, Table 1]
- **Recovery weakens with unsafe-prefix depth.** DS-Qwen-7B safety under self-prefilling falls from 74.8 (50 tokens) to 39.7 (full trajectory). DS-Llama-8B falls from 82.1 to 63.4. [AUTHOR-REPORTED, App. E, Table 6]
- **Hard-to-detect trajectories (H-CoT) mostly not recovered.** Recovery is 23.5 (DS-Qwen-7B) and 37 (DS-Llama-8B). [AUTHOR-REPORTED, §5.4.2, Table 2]
- **Some reasoning regression.** AIME 2025 pass@16 on DS-Qwen-7B falls from 70.0 to 63.3. [AUTHOR-REPORTED, App. D, Table 5]
- **Baselines that over-refuse.**
  - Safe-SFT (STAR-1) cuts XSTest compliance on DS-Qwen-7B from 91.2 to 73.6.
  - STAR-1 cuts Qwen3-8B AIME24 from 60.0 to 46.0.
  - SafeChain cuts Qwen3-8B AIME24 to 39.0. [AUTHOR-REPORTED, Table 1]

### 2. Abandoned, unstable, degenerate or reward-hacked
- **Feedback loops avoided by design.** "To avoid recursive error accumulation and feedback loops, we do not re-monitor or add trajectories whose initialization comes from the buffer back into B." [AUTHOR-REPORTED, §4.2.1]
- **Stale buffer.** The authors flag "distribution shift during training, where the buffer might accumulate stale trajectories from outdated policies". FIFO with capacity C = 256 is the fix. [AUTHOR-REPORTED, §4.1.2, A.3.2]
- **Metric hacking.** Training and evaluation judges are decoupled "to mitigate 'metric hacking'". [AUTHOR-REPORTED, A.4]
- **Unstable, collapsed or reward-hacked runs.** Not reported.

### 3. Hyperparameter sensitivities
- **Guard threshold.** At consecutive-token threshold t = 2:
  - accuracy 65.7, precision 94.2
  - 904 false negatives out of 2,935 prefixes, judged against GPT-4o
  - raising t adds false negatives (990 at t = 5) for marginal precision. [AUTHOR-REPORTED, App. C.2, Table 3]
- **KL disabled.** kl_loss_coef = 0.0, with lr 1e-6, G = 16 and 3 epochs. Sensitivity to these was not reported. [AUTHOR-REPORTED, A.3.2]

### 4. Stated limits
- **External guard.** The monitor is an external stream guard. Detectors tailored to the policy's own unsafe-state distribution (probes, SAE guards) are left to future work. [AUTHOR-REPORTED, §6]
- **Cost.** About 150 H100-hours per 7-8B run. [AUTHOR-REPORTED, A.3.1]

### 5. Independent re-evaluations
- None found in a brief search; the paper is from May 2026. [INDEPENDENT, WebSearch 2026-10-17]

### 6. Implication for the harm twin
- **Coupling 1:** the main model's own on-policy failures beat static harmful prefixes. That favours a same-base twin whose outputs track the current main model.
  - The negatives buffer must be FIFO-refreshed and must not re-ingest outputs produced from buffer states.
  - Robustness still decays with deeper harmful prefixes.
  - A guard that misses about 31% of unsafe prefixes means coverage of the twin's negatives is partial.

## Twisted SMC for LMs (Zhao, Brekelmans, Makhzani, Grosse; arXiv 2404.17546, VERIFIED via arxiv.org/html fetch; ICML 2024)

Method: samples from σ ∝ p0·φ, for example p0 times the probability of toxic output, using learned twist functions in SMC. It trains twists contrastively (CTL) and gives bidirectional SMC bounds on log Z, which is the probability of the rare target.

### 1. Ablated variants or components that hurt
- **Plain importance sampling fails on rare toxic outputs.** In toxic stories with threshold η = −5 (>99% toxic):
  - with p0 as proposal, "the SIS LB (orange) generally fails to draw any samples meeting the threshold"
  - the SIS/IWAE log Z lower bound collapses to about −36, the ε = 1e-16 floor, when no sample hits the set.
  - Exact samples via rejection "can require hundreds of thousands of samples". [AUTHOR-REPORTED, §7.1, Fig. 2, G.4]
- **Twist-learning methods that did worse on toxicity (Table 2).** Values are KL(q‖σ) / KL(σ‖q).

  | Method | KL(q‖σ) / KL(σ‖q) |
  |---|---|
  | CTL | 1.11 / 1.07 |
  | RL (soft Q) | 1.52 / 1.42 |
  | SIXO | 1.71 / 1.98 |
  | FUDGE | 3.24 / 2.00 (worst) |
  | PPO | 0.98 / 1.32 |

  PPO is best on the mode-seeking direction but worse on the mass-covering one. [AUTHOR-REPORTED, §7.2.1, Table 2]
- **Infilling.** FUDGE is again worst at 60.93 / 19.85, and RL reaches 31.35 / 14.96. [AUTHOR-REPORTED, Table 4]
- **Approximate positives are worse than exact positives.** Same sample count, toxicity, KL(q‖σ), exact → approximate:

  | Method | Exact | Approximate |
  |---|---|---|
  | CTL | 2.54 | 2.99 |
  | RL | 3.23 | 3.48 |
  | SIXO | 2.37 | 2.70 |
  | DPG | 1.51 | 2.35 |

  "Having access to exact target samples makes learning lead to lower KL divergences in a more reliable manner." [AUTHOR-REPORTED, H.3, Table 9, Fig. 4]
- **Twist parameterisation.** A linear twist head was "capacity constrained … achieving worse KL divergences". A separate twist transformer "does not always do better than just an MLP head", while roughly doubling forward cost. [AUTHOR-REPORTED, G.2.1, G.2.3]

### 2. Abandoned, unstable, degenerate or reward-hacked
- **Cold start on a thresholded rare target.**
  - "All importance weights will be 0 if none of our K samples meet the threshold".
  - "Initial iterations of twist learning receive no learning signal until a thresholded positive sample is drawn".
  - For the baseline comparisons, the authors therefore abandoned the indicator target for the smooth target p0·p(toxic). [AUTHOR-REPORTED, G.3]
- **Numerical issues.** The indicator target "directly runs into numerical issues for calculating log Z". It was patched with ε + I[·]. [AUTHOR-REPORTED, G.4]
- **No upper bound for sharpened targets.** At β > 1 (more peaked toxic target) "we can no longer generate exact posterior samples and thus cannot upper bound log Z". Only a qualitative example is given. Its "toxic" story, judged about 88% toxic, is a child's thumb-stuck story flagged for words like "suck", "hurt" and "cry", which reads as classifier exploitation. [AUTHOR-REPORTED, H.1]
- **PPO mode-seeking.** PPO optimises mode-seeking KL(q‖σ), and the "mode-seeking behavior has been shown to reduce diversity". PPO "suffers on KL(σ‖q)" because it never uses σ samples. [AUTHOR-REPORTED, §5.1, C/D, H.1]

### 3. Hyperparameter sensitivities
- **Learning rate.** Coarse grid search over 1e-6 to 1e-3 per method; the best was usually 3e-5 or 1e-4. [AUTHOR-REPORTED, G.4]
- **§7.1 untuned.** The §7.1 setting was not tuned ("we did not tune hyperparameters because we found this setting to work well"). [AUTHOR-REPORTED, G.4]
- **Particle count.** K matters: bounds only tighten as K grows. [AUTHOR-REPORTED, §7.1]

### 4. Stated limits
- **Scale.** Experiments use TinyStories-33M and GPT2-Medium with 10-20 output tokens, judged by a small toxicity classifier, so nothing is at chat-model or long-response scale. [AUTHOR-REPORTED, §7, G.4]
- **Upper bounds need exact samples.** These are only available via rejection sampling or BDMC (posterior) constructions. [AUTHOR-REPORTED, §5]
- **Dual use.** "With a single sign change", the method can be repurposed for harmful sampling. [AUTHOR-REPORTED, Impact Statement]

### 5. Independent re-evaluations
- **Adaptive Multilevel Twisted SMC (UNVERIFIED).** Liu, Wu, Summa and Zheng (arXiv preprint, about Aug 2026) reportedly identify that standard twist learning "depends on positive examples from the rare-event target distribution". When the behaviour is almost never observed, there are "too few useful positive examples to learn an effective twist". This is known only from an AI-generated news summary (aiunderstanding.org); the arXiv listing was not reached, so the ID and results are unverified. [INDEPENDENT, UNVERIFIED secondary source]
- **No re-evaluation found reporting outright failure.** [INDEPENDENT, WebSearch 2026-10-17]

### 6. Implication for the harm twin
- **Coupling 2:** estimating the main model's probability of a rare harmful completion is high-variance and fails silently. With base-model proposals, plain sampling gets zero hits, and a learned sampler gets no gradient until it finds a first positive.
  - A twin rewarded by main-model likelihood of judged-harmful text will face a cold-start, sparse-reward regime. It will drift toward classifier-exploiting text, as in the "suck/hurt/cry" story.
  - Use a smooth harm probability rather than a threshold.
  - Seed with known positives.
  - Track mass-covering KL(σ‖q), because PPO-style twin training optimises mode-seeking KL and will drop harm modes.

---

# Cross-paper lessons for the harm-twin loop

1. **Unbounded likelihood-lowering collapses the main model, and the collapse looks like success.** Every paper that pushed a likelihood down with no bound saw outputs degenerate. Use a bounded loss (NPO-style or DPO-style), a retain or KL anchor, and a floor on how far the twin-response likelihood may fall. Stop on held-out harm and utility scores, never on the negative loss. (Coupling 1 and coupling 2's main step.)
   - Plain gradient ascent peaks around step 6, then turns to gibberish at an unpredictable step (NPO §2.1, §5.2).
   - Whitespace output counts as "non-harmful" (Yao Table 3: diversity 0.787 to 0.118).
   - Forget loss above 60 while the model is still harmful (Yao §3.2).
   - Negated cross-entropy diverged, and the only bounded alternatives tried barely converged (TAR §4.2, App. F).
   - ADPO names "nonsensical but non-toxic" output as the reward-hacking failure when toxic likelihood goes too low (ADPO §5.4).

2. **A "robust" main model is often just a broken one, so test it hard.** Every robustness claim needs side-by-side utility, fluency and over-refusal numbers, plus a stabilised attacker over several seeds. (Both couplings.)
   - TAR's durability came from fluency loss and an early-step gradient explosion. Warmup plus cosine decay, extra seeds, LoRA, retain-only fine-tuning or a chat template undid it, with GSM8K at 1.5-4.8 vs 74.4 (Qi et al. §3-4; Che et al. 2502.05209, INDEPENDENT).
   - Token Buncher's protection is deliberate capability collapse under attack (Table 11).
   - Simko's defence "works" by producing "The, / / /" gibberish (App. A.2).

3. **Coupling 2's reward is mis-specified as stated: high likelihood under the main model is not the same as harm.** The twin's reward must be "the main model's likelihood of a judged-harmful complete response", not "likelihood of a harmful-looking prefix". The judge must be validated on incomplete and unfaithful outputs, and must be separate from the twin's training reward.
   - In AdvPrefix, low-loss and even prefilled "Sure, here is" prefixes still fail about 90% of the time. The model self-corrects into disclaimers or incomplete answers, and judges over-credit these (10% true vs about 40% judged on Gemma-2) (AdvPrefix §3.2, Fig. 2-3).
   - The harmful-RL attacker itself reward-hacks a harm reward model (Token Buncher App. F).
   - In Twisted SMC, the sharpened toxic target produced a child's story flagged for "suck/hurt/cry" (Twisted SMC H.1).

4. **The harmful outputs coupling 2 targets are rare, so the twin starts with almost no reward.** Seed the twin from known harmful positives or a harm-SFT start. Use a smooth harm probability rather than a threshold. Track mass-covering diversity, since PPO-style twin training is mode-seeking and will drop harm modes (Twisted SMC Table 2: PPO 0.98 / 1.32).
   - Sampling from the base model draws zero rare toxic samples. The log Z lower bound sits at the epsilon floor. Twist learning gets no gradient until a first positive appears (Twisted SMC §7.1, G.3).
   - Contrast: with a dense harm reward model and a KL coefficient of 1e-3, RL broke alignment in fewer than 100 steps (Token Buncher §4.1). Reward density decides whether the twin learns at all.

5. **The twin must track the current main model and be its own copy, not a frozen or foreign one.** Keep the twin's weights separate from the main model's. Update the twin against the live main model, refresh its negatives first-in first-out, and do not feed outputs generated from buffer states back into the buffer. (Coupling 1 especially.)
   - AntiDote's static adversary raises harmfulness 3-6x (5.7 to 33.1; Table 4).
   - In MACPO, negatives from other models "substantially" degraded alignment (§6.4), but its twin was frozen.
   - Self-ReSET uses a FIFO buffer and never re-adds buffer-born trajectories, to avoid feedback loops (§4.2.1).
   - ADPO's harm mode stored in the same weights raised base toxicity from 4.7% to 13.3% (Table 1).

6. **If the twin converges to the main model, the loop stops learning.** Keep a distance constraint between twin and main model, and keep the twin's harm data broad.
   - DR-SFT says so directly: when the twin equals the aligned model, "the contrastive signal disappears" (§7).
   - A weaker twin let more harm through: ASR rose from 14.3% to 19.8% as the twin's training data shrank from 2,500 to 250 samples (App. C.2).
   - Harm categories the twin does not cover are the stated failure (DR-SFT §7, Token Buncher §8).
   - Out-of-distribution attack data raised Token Buncher's harmful score from about 1% to 13-14% (Table 8).

7. **Hardening against only the worst case trades one attack for another, so keep a fixed pool of standard harmful data in every round.** Mix the twin's negatives with that pool and evaluate on held-out attack families. (Both couplings.)
   - Simko's re-fitted adversary raised input-space attack success from 0 to 3.75% and embedding-attack success from 65.9% to 75.6% (Table 2).
   - Lyu's training on worst insertions lost prefill robustness on Gemma at every span length (13.8 vs 7.1; Table 3).
   - AntiDote loses to simpler baselines on attacks that look benign internally (§4).
   - TAR patched attacks one at a time (App. D.2).

8. **Updates against the twin's responses mostly change the first few tokens, so the main model's refusals stay shallow.** Include mid-response insertions and deep prefixes among the negative states, and measure robustness by prefix depth.
   - Lyu shows that SFT and DPO gradients concentrate on early tokens. A 10-token harmful span inserted mid-refusal redirects aligned models more than 75% of the time (Lyu §2.2).
   - Self-ReSET's recovery falls from 74.8 to 39.7 as the unsafe prefix deepens (App. E).
   - A twin that learns to start the harm late will get past a shallow refusal.

9. **Alternating loops of self-generated data collapse within a few rounds unless the positive side gets outside diversity.** Anchor the positives in curated or outside data, use small interleaved steps, compute utility on the clean model, and plan for collapse within 2-3 rounds.
   - Self-play baselines collapse over iterations: SPIN 51.62 to 46.02, and PKU-SafeRLHF 55.5 to 41.0 (MACPO Table 1).
   - MACPO itself degrades after round 2 with one weak teacher (§6.2).
   - Token Buncher dropped long-cycle alternation for fear of forgetting and dropped weighted-sum losses as hyperparameter-sensitive (§5.6).
   - AntiDote's coupled capability loss cost 4.8-5.4 points of accuracy (Table 5).

10. **Lowering the likelihood of harm does not remove it, so do not treat the loop's output as unlearning.** If the harm-twin loop is used for certification, report robustness to further fine-tuning separately.
    - 4-bit quantization brings back 83% of forgotten knowledge (2410.16454, INDEPENDENT).
    - Benign relearning brings an NPO-unlearned model's score from 1 back to 5.08 (5.92 before unlearning) (2406.13356, INDEPENDENT).
    - Benign data alone undoes RepNoise and TAR (Qi et al. §4).
    - Targeted unlikelihood training raised general harmful compliance: Sorry-Bench 58.3 vs 22.2 (Qi et al. Table 3).

**Hyperparameter notes common to all papers.**
- Most methods were tuned by sweep on a single seed, with little or no sensitivity reporting. ADPO used seed 42 only. AntiDote cites an ablation section that does not exist. Lyu, MACPO and Self-ReSET report no sweeps.
- The settings that mattered when tested:
  - beta (NPO reduces to gradient ascent as beta goes to 0; DR-SFT trades utility against safety through beta)
  - retain weight (NPO's best is about 2 and non-monotone; it needs re-tuning per forget-set size)
  - tamper-resistance weight (TAR: 4.0 cuts MMLU from 67.3 to 54.7)
  - inner-loop length (TAR)
  - number of unlearning steps (Yao: 200 batches is too few, 1,000 gives nonsense)
  - learning-rate schedule and warmup (Qi et al.; DR-SFT on Gemma)
  - attacker temperature (Token Buncher: random characters at 1.7)

# Negative results: generator-vs-verifier games (structural precedents for the harm twin)

Date: 2026-10-17. Scope: what did NOT work in eleven papers where a generator is trained against a verifier, a reward model or a likelihood. The aim is to find failure modes, not headline results.

**Design these notes feed.** A "harm twin" LM starts from the same base as the main generative LM. The twin is trained with RL to write harmful responses, and the two models are updated in turn.
- **Coupling 1:** the main model is trained away from the twin's responses, using them as negatives (DPO rejected side, unlikelihood or NPO).
- **Coupling 2:** the twin's reward is the main model's own likelihood of the twin's harmful response.

**How the papers were read.** The arXiv HTML for every paper was fetched with `curl` and converted to plain text locally, so the full text could be searched for exact quotes and table cells. WebFetch returns a summary from a small model, which is not enough for exact quotes; that is the only reason for the fallback. For each paper I read the ablation section, the appendix, the limitations section and any "we also tried" remarks. Every number below was copied from the converted text. No paper content was filled in from memory.

**Search for later re-evaluations.** I downloaded each paper's list of citing papers, with citation contexts, from the Semantic Scholar Graph API and scanned the contexts for critical wording. Where a context looked like a re-evaluation, I fetched that paper and read the relevant passage. When this note says "none found", it means none were found in that scan. It does not mean none exist.

**Labels.** AUTHOR-REPORTED = the paper's own results. INDEPENDENT = a later paper by other authors. INDEPENDENT (own check) = arithmetic I did on the paper's own tables; it is not a published re-evaluation.

## arXiv ID verification

All IDs below were checked on arxiv.org (HTML or abstract page title).

| Paper | arXiv ID | Status |
|---|---|---|
| APRM, Adversarial Training for Process Reward Models (Juneja, Nathani, Wang) | 2511.22888 | VERIFIED |
| Adv-RM, Adversarial Training of Reward Models (Bukharin et al.) | 2504.06141 | VERIFIED (v2 read) |
| SPC, Evolving Self-Play Critic via Adversarial Games (Chen et al.) | 2504.19162 | VERIFIED (v2) |
| Hallucination Self-Play (HSP) (Yang et al.) | 2607.07993 | VERIFIED (v2, 07 Oct 2026) |
| Pns, Not All Negative Samples Are Equal (Di et al.) | 2602.03516 | VERIFIED |
| DuoGuard (Deng et al.) | 2502.05163 | VERIFIED |
| Amortizing intractable inference in LLMs (Hu et al., ICLR 2024) | 2310.04363 | VERIFIED |
| GFlowNet-EM for learning compositional latent variable models (Hu et al., ICML 2023) | 2302.06576 | VERIFIED |
| Yang et al. 2018, LMs as discriminators for style transfer | 1805.11749 | VERIFIED |
| SSO, Self-Steering Optimization (Xiang et al.) | 2410.17131 | VERIFIED (v2) |
| APL, Adversarial Preference Learning (Wang et al.) | 2505.24369 | VERIFIED (v1) |

**ID correction.** The task listed "GFlowNet-EM, Hu et al., arXiv 2310.04363". That ID is the ICLR 2024 paper *Amortizing intractable inference in large language models*. GFlowNet-EM is a different, earlier paper, arXiv **2302.06576** (ICML 2023). Both are covered below because both bear on coupling 2. In 2310.04363 the sampler's reward is the LM's likelihood p_LM(XZY), and an optional M-step then fine-tunes that LM. In 2302.06576 the sampler and the generative model are trained jointly, in alternation.

Independent sources used (all VERIFIED on arxiv.org): 2404.04291 (alpha-SPIN), 2405.18540 (GFlowNet red-teaming), 2603.00454 (RapTB), 2602.04581 (Trust the Typical), 2504.04377 (PolyGuard), 2602.00857 (text scoring robustness).

---

## 1. Yang et al. 2018, LMs as discriminators (1805.11749)

**Method in one line:** a style-transfer generator is trained, through a Gumbel-softmax relaxation, to minimise its outputs' perplexity under a target-domain LM. In the "LM + adv" variant the LM is also trained to raise the loss on the generator's samples, used as negatives (weight gamma).

**1. Ablated components that hurt (AUTHOR-REPORTED, Table 1, decipherment BLEU)**

| % of tokens ciphered | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| LM (gamma = 0) | 89.0 | 80.0 | 74.1 | 62.9 | 49.3 |
| LM + adv | 89.1 | 79.6 | 71.8 | 63.8 | **44.2** |

- Adding negatives to the LM's training hurt in three of five settings: -0.4 at 40%, -2.3 at 60% and **-5.1 at 100%**.
- It helped only marginally at 20% (+0.1) and 80% (+0.9).

**2. Abandoned or unstable (AUTHOR-REPORTED, §4.1 Results)**
- Exact text: "We can see that using adversarial training sometimes improves the results. However, we found empirically that using negative samples makes the training very unstable and the model diverges easily. This is the main reason why we did not get consistently better results by incorporating adversarial training."
- And: "Given unstableness of adversarial training and effectiveness of language models, we set gamma=0 in Equation 1 and 2 in the rest of the experiments." The adversarial variant was switched off for the sentiment and translation tasks.
- Conclusion (§6): "it is possible to eliminate adversarial training with negative samples if a structured model is used as the discriminator".
- §3, "Overcoming mode collapse": "In preliminary experimentation, we found that the language model prefers short sentences." They needed two fixes: (1) normalising the LM loss by length, and (2) forcing the output length to equal the input length. Exact text: "We find these two tricks stabilize the training and avoid generating collapsed overly short outputs."
- §3: single-sample REINFORCE "leads to high variance in gradient estimates and thus unstable learning". For this reason they used the continuous (Gumbel-softmax) approximation instead of RL.

**3. Hyperparameter sensitivity:** gamma, the weight on negatives in the LM loss, decides stability. No sweep beyond the on/off comparison is reported.

**4. Stated limits (AUTHOR-REPORTED, §4.2):** a collapsed output such as "a single word 'good'" can satisfy the style classifier. Fluency (perplexity) and content (BLEU) checks are needed alongside it.

**5. Independent re-evaluation:** none found. I scanned the citation contexts of 279 citing papers, including style-transfer surveys and benchmarks such as TSTBench and Hu et al. 2020. None re-ran the LM + adv variant. Later papers (2010.11578, and a multi-dimensional style transfer paper) cite Yang as the reason to *skip* adversarial negatives.

**6. Implication.**
- *Coupling 1:* feeding generator samples back as negatives into a likelihood model is the step this paper found divergent and switched off.
- *Coupling 2:* a twin rewarded by an LM's likelihood will drift toward short outputs unless the reward is length-normalised or length is controlled.

---

## 2. Amortizing intractable inference in LLMs (2310.04363)

**Method in one line:** a LoRA copy of the LM is fine-tuned as a GFlowNet so that it samples Z in proportion to the reward R(Z) = p_LM(XZY). An optional M-step then fine-tunes p_LM on those samples (EM).

**1. Components that hurt or were essential (AUTHOR-REPORTED)**
- **The EM M-step can hurt (Table 3, SUBJ test accuracy).** With 20 training samples, GFlowNet fine-tuning gives 81.1 ± 0.4; adding the supervised M-step drops it to 78.7 ± 1.6 (-2.4). It helps at 10 samples (71.4 → 75.2) and at 50 (87.7 → 89.9). The text says only "In some cases" the M-step helps.
- **Replay-buffer seeding is load-bearing (Table D.4, SUBJ).** Removing it gives 71.4 → 64.7 at n=10, 81.1 → 68.7 at n=20, and 87.7 → 77.0 at n=50. Also removing test-time aggregation gives 63.9 / 65.9 / 75.4.
- **Arithmetic task with no seed rationales (Table E.3).** Accuracy is 22.6 / 5.8 / 3.4 (3, 4 and 5 operands), against 95.2 / 75.4 / 40.7 with 50 seeds. The relationship is not monotone: 25 seeds gives 48.8 on 3 operands, below 10 seeds (58.6). Exact text: "When no buffers are used to seed the buffer, the performance is quite poor."

**2. Reward hacking of a likelihood reward (AUTHOR-REPORTED, §4.4 and App. E)**
- PPO trained on the same likelihood-based reward scored 30.6 / 13.7 / 5.6 (Table 4).
- Exact text (App. E, "Analysis"): "PPO generates sequences with high rewards under the base model. These sequences, however, are not valid expressions, and, in fact, do not even call the tool. Instead, the model learns to simply repeat the expression. Repetitions being assigned high likelihood is an issue that has also been noted by Welleck et al. (2020)."
- §4.4 calls this "overoptimization to a misspecified reward".

**3. The base LM as reward is often miscalibrated (AUTHOR-REPORTED)**
- App. C: "none of the pretrained base models were good at assigning high likelihoods to plausible stories". They fine-tuned a separate GPT-2 Large on the full story dataset to act as the reward.
- App. E: "incorrect rationales are sometimes assigned very high likelihood". They had to prepend k=3 hand-written demonstrations to the reward prompt, which "improves the calibration to some extent".
- App. E: the remaining GFlowNet errors come from a reward model that "assigns equally high rewards to rationales with these minor mistakes".

**4. Hyperparameter sensitivity (AUTHOR-REPORTED, App. B to E):** the reward temperature is annealed in every task (for example 1.2 → 1 over 150 steps on SUBJ). The sentence-continuation result uses temperature 0.875, chosen as "a good balance between log-likelihood and diversity". No sensitivity sweep is published.

**Stated limits (AUTHOR-REPORTED, Limitations):**
- "As with any on-policy method, exploration ... remains an open problem."
- "our method improves inference but not the knowledge in the LM. Issues such as hallucination or miscalibration ... are thus not addressed."
- Models tested are at most 6B parameters.

**5. Independent re-evaluations**
- **INDEPENDENT, alpha-SPIN (2404.04291), App. A.** The authors used Hu et al.'s GFlowNet fine-tuning to sample the losers in an iterated self-play preference loop. Exact text: "The chart in Figure 6 shows that the performance degrades starting from Iteration 1, and that more effort should be spent on the specifics of GFlowNet-finetuning in order to get accurate approximate samplers". This is the closest independent analogue to the harm-twin loop: a GFlowNet sampler of negatives inside an iterated preference loop. It failed from the first iteration.
- **INDEPENDENT, Lee et al., GFlowNet red-teaming (2405.18540), §3 and §4.**
  - "in practice GFlowNet fine-tuning can be overly sensitive to the peakiness of the reward".
  - With a small reward temperature beta, the attacker "would likely generate prompts from potentially spurious modes of the toxicity classifier, which will have high perplexity under the reference model". With a small gamma, it "would merely focus on the naturality score ... and not generate toxic prompts".
  - They needed an extra maximum-likelihood smoothing stage after GFlowNet training.
  - Their REINFORCE baseline "collapses to deterministically generating a single reward-maximizing prompt".
- **INDEPENDENT, RapTB (2603.00454), §1 and related work.**
  - "LLM-GFlowNets suffer from mode collapse ... (i) prefix collapse, where early-token entropy drops sharply ... and (ii) length bias".
  - SubTB, the objective Hu et al. use, "can drift in its termination probabilities".
  - "reward-prioritized replay can induce rich-get-richer collapse and reduce coverage (Shen et al., 2023; Vemgal et al., 2023; Hu et al., 2024)".
  - Evidence comes mainly from molecule and arithmetic tasks.

**6. Implication for coupling 2.**
- A sampler rewarded by the main LM's likelihood will find whatever the LM over-rates, such as repetition, spurious modes or short sequences, unless the reward is calibrated and diversity is enforced.
- The only independent test inside an iterated preference loop (alpha-SPIN) degraded from iteration 1.

---

## 3. GFlowNet-EM (2302.06576)

**Method in one line:** a conditional GFlowNet q(z|x) (the E-step sampler) is trained with reward p_theta(z)p_theta(x|z) from a generative model. The generative model's M-step is trained on the GFlowNet's samples. The two alternate, and the reward changes as the generative model changes.

**1. Ablations that hurt (AUTHOR-REPORTED)**
- **Table 8, joint learning, CFG induction (NLL per word / sentence F1):**

| Configuration | NLL per word | Sentence F1 |
|---|---|---|
| Full GFlowNet-EM | 5.70 | 34.85 |
| - MCMC | 6.02 | 28.56 |
| - Sleep phase | 5.91 | 28.13 |
| - SubTB | 5.84 | 26.56 ± 7.82 |
| - Exploration | 5.70 | 31.87 |
| Sleep only | 6.08 | 48.41 |

  The "Sleep only" F1 of 48.41 ± 1.38 is identical to the "+EBM Prior" entry in Table 1. It may be a copy error in the paper.
- **Table 6, fixed grammar (E-step only).** Without the sleep phase, the NLL bound worsens from ≤ 5.97 to ≤ 8.56; with TB instead of SubTB it is ≤ 8.95; HVI gives ≤ 13.14.
- **Table 2, discrete VAE on MNIST (NLL).**
  - Greedy decoder training speeds convergence but hurts NLL: 74.18 → 76.22 at K=4. The authors call it "a biased objective in the M-step".
  - Adding a jointly learned prior hurts further at K=4 (78.59).

**2. Collapse (AUTHOR-REPORTED, §4.1 and App. C.7)**
- The core hazard: "the GFlowNet is trained with a nonstationary reward ... it is important for the GFlowNet to track the true posterior as it evolves, so as not to bias the M-step and produce degenerate solutions."
- App. C.7: "although thresholding is necessary – without it, the generative model tends to collapse, i.e., most symbols of the grammar receive almost zero mass in the posterior". Thresholding means taking an M-step only when the sampler's loss is below a threshold.
- App. C.7: "despite strong performance on the fixed grammar, only using the Sleep phase performs much worse in the case of joint learning. This is potentially due to the fact that reward in joint learning is non-stationary and thus hard to model without exploration." A component that works against a fixed reward can fail once the reward moves.
- App. C.8: guided by an EBM prior, GFlowNet-EM "collapses to right-deep trees".

**3. Hyperparameter sensitivity (AUTHOR-REPORTED, Table 7).** Threshold schedules between 12→6 and 4→2 reach similar final NLL (5.76 to 5.79). The F1 spread is 30.56 to 34.49, all with overlapping standard deviations. The schedule mainly controls speed. The update ratio is set adaptively by the loss threshold; on the VAE a fixed 400/400 E/M schedule "was adequate".

**4. Stated limits (AUTHOR-REPORTED, §5.3):** "the more expressive posterior and lower NLL come with an increased training cost". Exact sampling (marginalisation) still beats GFlowNet-EM on the tractable CFG (Table 1: F1 39.51 against 34.85).

**5. Independent re-evaluation:** none found among 50 citing papers' contexts.

**6. Implication for coupling 2.** When the model that defines the sampler's reward is being changed by that sampler's own outputs, the reward is nonstationary. Without a gate that lets the sampler catch up before the model moves, the model collapsed. The harm twin needs an equivalent gate: update the main model only when the twin has re-fit the current main model.

---

## 4. Adv-RM (2504.06141)

**Method in one line:** an RLOO-trained policy is rewarded for responses that the target RM scores above the SFT mean while a second RM scores them low (reward R1 - lambda*R2). The filtered samples become rejected responses for retraining the RM from scratch, over several rounds.

**1. Ablations and settings that hurt (AUTHOR-REPORTED)**
- **Table 3, strict attack success in the synthetic setup.** Full method 92.91. Without filtering 73.27. With equal RM weights (lambda = 1) **16.41**. Without the reward threshold **0.0**.
- **Fig. 6(b).** More adversarial data hurts downstream: "around 1000 samples results in the best downstream performance, and that more samples actually hurts performance. We hypothesize this is due to the fact that Adv-RM data is not very diverse, so adding too much such data can result in overfitting."
- **Fig. 6(c).** "two rounds leads to the best performance, and that after two rounds, there is minimal gain". §3.4: "after 2 rounds of adversarial training, our attack procedure is no longer successful."
- **Table 4, RewardBench.** Small regressions: Chat 0.9721 → 0.9609 and Safety 0.8541 → 0.8514. Overall 0.8329 → 0.8399.
- **§5, downstream benefit is mostly robustness.** "Adv-RM only results in a small improvement in best-case policy performance, and mostly improves robustness against reward hacking."

**2. Unstable or hacked (AUTHOR-REPORTED)**
- §3.2: without the R2 penalty term, the attacker can find samples where "all reward models predict will receive a high score, and the reward models just disagree on how high the score is".
- §4.3: the over-optimised baseline policy produced "high-quality responses with only minor quirks (e.g., excessive formatting)", so plain reward over-optimisation is not an adversary.
- Attacks barely transfer: "some weak transfer ... on the Skyworks 27B model, but no transfer between the different Nemotron RMs" (Fig. 3).
- §4.3: "using LLM as a judge may significantly underestimate the success rate of Adv-RM as they may also be fooled by adversarial examples." Against Nemotron-70B, Llama-405B as judge gives 24% success while humans give 83%.

**3. Hyperparameter sensitivity (AUTHOR-REPORTED, App. C):** lambda2 tuned in {8, 10, 12}; C = -25; policy LR 5e-7. lambda = 1 collapses the attack (Table 3). Uncertainty is uninformative on in-distribution data (§3.3): Pearson -0.05 on SFT samples against -0.70 on SFT plus Adv-RM samples.

**4. Stated limits (AUTHOR-REPORTED, §6).**
- "Adv-RM relies on reward ensembles, meaning adversarial responses may be missed if all RMs incorrectly predict they are high quality".
- About 3x the compute of standard RLHF: 672 against 224 A100-hours (App. F).

**5. Independent re-evaluation**
- **INDEPENDENT (competitor re-implementation), APRM 2511.22888, Table 1 and App. 6.14.2.** APRM's authors re-implemented Adv-RM ("ARM") on *process* reward models. They state that the original data could not be used, so this is an adapted setting, not a replication. ARM averaged 76.1 with a GPT-OSS-120B solver, below plain CoT self-consistency (77.2) and a static hard-negative baseline (80.0). With a Gemma-3-27B solver it averaged 59.9, against 76.2 for hard negatives.
- 2602.00857 notes only that Adv-RM "does not consider the robustness of trained models against standard attacks"; it does not run it.
- No paper found that breaks Adv-RM in its own setting.

**6. Implication for coupling 1.**
- The useful dose of generator negatives is small (about 1000 pairs here); more caused overfitting.
- Gains saturate after about 2 rounds.
- The adversary only finds failures its detector ensemble can see. A twin that is scored by the main model, or by a correlated judge, will miss blind spots they share.

---

## 5. APRM (2511.22888)

**Method in one line:** a generator rewrites reasoning steps to fool a PRM, while the PRM learns to classify steps. It is a general-sum game; both players use PPO with KL and entropy regularisation and an OGDA optimiser. A hand-built correctness oracle labels generator steps, and the generator gets -1 for producing a correct step.

**1. Ablations that hurt (AUTHOR-REPORTED, Table 4, GPT-OSS-20B solver)**

| Variant | MATH | JEE | Oly | Overall |
|---|---|---|---|---|
| Full | 91.0 | 73.0 | 90.0 | 85.2 |
| No entropy regularisation | 86.4 | 69.4 | 85.5 | 80.9 |
| No OGDA | 84.6 | 67.9 | 83.7 | 79.2 |
| Both removed | 83.6 | 67.2 | 82.8 | 78.4 |

Text: "Removing entropy regularization reduces accuracy by about 4 points, while removing OGDA has a similar effect." No seeds or error bars are reported anywhere in the paper.

**Where APRM loses to its baselines (AUTHOR-REPORTED, Table 5).** MATH500 is below ReST-MCTS for three solvers:
- GPT-OSS-120B: 91.4 against 91.6.
- Gemma-3-27B: 91.4 against 93.2.
- Gemma-3-12B: 80.0 against 81.4.

AMC with Gemma-3-27B is 60.8 against 61.2.

**INDEPENDENT (own check) on the Gemma-3-27B "Avg." column of Table 5.** APRM is reported at 85.2 and ReST-MCTS at 72.6, a 12.6-point gap. The per-benchmark differences (APRM minus ReST) are -1.8, +4.1, +6.4, +1.3 and -0.4. Any weighted average that uses the same weights for both methods has a gap no larger than the largest difference, 6.4. So no shared weighting reproduces 12.6. Simple means give 81.8 against 79.9. The Gemma-3-27B average column should not be quoted.

**2. Instability and failures (AUTHOR-REPORTED)**
- §2.3: without regularisation, "standard gradient methods often exhibit undesirable behaviors such as cycling, chaotic dynamics, or divergence".
- Footnote 2 concedes that the convergence theorem is in policy space: "Existence of an equilibrium in policy space does not imply that there is a single weight vector achieving that equilibrium, or that gradient-based training will find it. Nor do we claim in this paper that we are able to find the Nash equilibrium."
- §3.3, failure cases: the PRM misses theorem-precondition errors, such as the alternating series test on a non-alternating sequence or Jensen's inequality outside the region of concavity. Text: "We hypothesize that G struggles to generate adversarial negatives that specifically target theorem preconditions". The generator never learned to produce that class of error, so the verifier never learned to catch it.
- App. 6.7: the oracle needs proximity filters (length within 0.5x to 2x, embedding similarity, edit distance) to stop "degenerate far-off edits and trivial copies".

**3. Hyperparameter sensitivity:** tau = c_H = 0.01, LR 1e-6, PPO batch 4 (App. 6.14.1). No sweep. The ablation shows that regularisation and the optimiser each cost about 4 to 6 points when removed.

**4. Stated limits (AUTHOR-REPORTED, §5):**
- Training compute is higher than static-data approaches (72 against 60 GPU-hours, App. 6.9).
- "further work is needed to understand large-scale training dynamics and to develop algorithms that enhance stability".

**5. Independent re-evaluation:** none found (3 citing papers). One (2606.18810) lists APRM among sources noting that PRMs "remain vulnerable to reward hacking", without testing it.

**6. Implication for coupling 1.**
- Entropy and KL regularisation plus an optimiser built for games were each worth about 4 to 6 points.
- An adversary trained only on what fools the verifier leaves whole classes of errors unexplored. In APRM's case the generator's reward could not express theorem-precondition errors.
- APRM's ground truth comes from an external oracle. The harm twin needs a harm oracle that does not depend on the main model.

---

## 6. SPC, Self-Play Critic (2504.19162)

**Method in one line:** two copies of a base model, a "sneaky generator" of erroneous steps and a step critic, are trained with offline RL on game outcomes (±1). The generator also needs the corrupted step to actually lower an LLM solver's success rate.

**1. Ablations that hurt (AUTHOR-REPORTED, §4.3, Fig. 3)**
- **Synchronous round 2 hurt the critic.** Round 2 data from Sneaky-1 against Critic-1, mixed with round 1, dropped ProcessBench accuracy from **76.8% to 72.0%**. Exact text: "We notice that the win rate of Sneaky-1 against Critic-1 is only 13.2%. Therefore, such an overly unbalanced game might prevent the critic from learning new knowledge". The fix was asymmetric: play Sneaky-1 against the *older* Critic-0 (32.5% win rate), then retrain from the round-0 checkpoints on the merged data. This gave 77.7%.
- **Unpaired samples.** Without paired positive and negative critiques the critic reached 75.0% instead of 76.8%.
- **No reward from the solver.** Without it, the generator's successful-attack rate fell to **12.1%**, against 21.5% for Sneaky-0 and 33.6% for Sneaky-2, and its win rate against Critic-0 fell to 19.6%. Exact text: "it is crucial to ensure that the sneaky generator receives rewards from both the solver and the critic."

**Round-to-round trade-offs that the averages hide (AUTHOR-REPORTED, Table 2).** On DeltaBench, recall of *error* steps fell from 64.0 (round 0) to 49.3 (round 1) and 52.8 (round 2), while recall of correct steps rose from 45.9 to 68.4. On PRM800K, round 1 recall of correct steps fell from 67.8 to 59.4. ProcessBench Omni-MATH dipped from 70.3 to 69.2 in round 2 (Table 1).

**2. Abandoned:** the synchronous update schedule (above). Only 2 rounds were run.

**3. Hyperparameter sensitivity (AUTHOR-REPORTED, App. C.3):** KL coefficient 0.1. An SFT loss with coefficient 0.15 was added "to ensure the stability of RL training". Generator data was fixed at a 1:1:1 mix of (failed on solver), (fooled solver but caught by critic) and (fooled both).

**4. Stated limits (AUTHOR-REPORTED, App. A):** only mathematical reasoning was tested. The authors flag misuse of a sneaky generator ("false and misleading information").

**5. Independent re-evaluation:** none found among 46 citing papers' contexts.

**6. Implication.**
- *Both couplings:* if the main model wins most games, the twin's samples stop teaching anything, and learning against them made the critic worse (-4.8 points). Track the twin's win rate and pair it with an older checkpoint of the main model when the game becomes one-sided.
- *Coupling 2:* the twin needs a reward that checks its output is truly harmful (the analogue of the solver check). Otherwise its success rate on the actual objective drops sharply.

---

## 7. Hallucination Self-Play, HSP (2607.07993)

**Method in one line:** two roles from the same base model. The generator is trained with GRPO to write hallucinated answers, rewarded by how often the frozen detector misses them (1 - accuracy). The detector is trained with RLVR on the generator's outputs. Updates alternate, each role frozen while the other trains.

**1. Reward hacking, the most direct precedent (AUTHOR-REPORTED, §3.2.2 and Table 4)**
- **Detector-only reward.** Exact text: "the generator can easily finds a shortcut ... A common failure mode is that the generator outputs a faithful answer, which the detector consequently fail to identify as hallucinated with high accuracy, thereby yielding a high reward. This hacking behavior introduces incorrect labels into the synthetic data".
- **With the gate on, a second hack appeared.** "Even with the above gating criteria, we observe an unexpected form of reward hacking: the generator learns to produce trivial or meaningless response, or to refuse answering."
- **Table 4, ablation of the safeguards (500 held-out samples):**

| Variant | Hallucination rate | Average reward | Trainable samples |
|---|---|---|---|
| Full HSP | 90% | 0.288 | 91 / 500 |
| No gate | **0%** | 1.000 | 500 / 500 |
| No trivial-answer penalty | 70% | 0.333 | 277 / 500 |
| No gate, no penalty | **0%** | 1.000 | 500 / 500 |

  The hallucination rate was judged by hand on only 10 sampled examples per variant.
- Even the full system keeps only 18% of generated samples (91/500).

**2. Static generator collapse (AUTHOR-REPORTED, §4.4, Table 2).** With a frozen base-model generator, round 1 is fine (82.5 → 83.3, close to HSP's 83.8). Round 2 *degrades to 73.6*, with CNN 52.0 and XSum 58.1, while HSP reaches 84.4. Training repeatedly on a fixed negative generator was actively harmful.

**Other losses (AUTHOR-REPORTED)**
- Table 1, detector with CoT: response-level precision falls below the RLVR baseline. QA precision is 72.0 → 65.4 at round 1; average precision is 79.6 → 77.1 at round 1 and 77.2 at round 3. Gains come from recall.
- Round 3 Summarization F1 (70.5) is below round 2 (71.4).
- Table 2, LLM-AggreFact: HSP round 2 is below its own SFT start on RAGTruth (93.5 → 88.1). Wice drops to 72.7 at round 1, against 76.4 for SFT.
- Average 84.4 is still below FaithLens (86.4).
- §3.3: "Even after a single round ... hallucinations produced by the updated generator quickly saturate and provide little additional learning signal".

**3. Hyperparameter sensitivity:** GRPO LR 1e-6, K=8 rollouts, 10 detector rollouts at temperature 1.0 for the reward (App. A). No sweep.

**4. Stated limits:**
- The validity gate depends on QA ground-truth answers and named-entity checks, which are simplified proxies (§3.2.2).
- Faithful labels come from the base model and are "not guaranteed to be hallucination-free" (§3.2.3).
- The generator is too weak to produce span labels, so the detector reward was reduced to a binary label.

**5. Independent re-evaluation:** none (0 citing papers found).

**6. Implication: the closest analogue to coupling 2.** The twin is rewarded by the main model's likelihood, roughly "the main model does not recognise this as something to avoid". It will first learn to output benign responses, and then refusals or empty responses, because these score well and are not harmful. Under coupling 1 those samples would then train the main model *away from benign or refusal behaviour*. An independent harm gate and a penalty for trivial or refusal outputs are prerequisites, not refinements.

---

## 8. Pns, plausible negative samples (2602.03516)

**Method in one line:** Qwen2.5-7B is GRPO-trained with an inverted reward. The reward needs format compliance and a *wrong* final answer, and gives bonuses for a high reward-model score and a high CoT-quality score. Its outputs become DPO rejected responses for a separate policy. This is a single, static generator, not an alternating game.

**1. Ablations that hurt (AUTHOR-REPORTED, Table 2, DPO directly on the base model, Qwen2.5-7B)**

| Variant | IND avg | OOD avg |
|---|---|---|
| Base model | 33.28 | 57.93 |
| Full Pns | 34.53 | 61.83 |
| No accuracy constraint | **26.99** | **49.76** |
| No format gate | 31.54 | 55.65 |
| No RM score | 32.32 | 59.42 |
| Raw RM score (no bucketing) | 32.34 | 59.44 |

- Removing the accuracy constraint falls *below the base model*. Text: "This effectively collapses the preference learning, as the model cannot learn to discriminate between correct and flawed reasoning when both chosen and rejected samples may contain correct answers."
- The text says removing the CoT score "yields similar degradation", but that row is not in Table 2.

**2. Reward hacking (AUTHOR-REPORTED, §3.1.2 and §4.4)**
- Without the format gate, "the Pns Model can exploit format violations (such as omitting the boxed answer or producing malformed tags) to trivially achieve incorrect outputs".
- Without bucketing, "the continuous RM signal provides overly fine-grained gradients that the Pns Model can exploit during late-stage RL training, leading to reward hacking". The RM score was therefore clipped to [-3.5, 3.5] and bucketed.
- The CoT score exists because "reverse RL training may degrade the model's instruction-following capabilities".

**Naive negatives hurt (AUTHOR-REPORTED, Table 1, AVG against the RL baseline)**
- Qwen2.5-7B: negative SFT 37.86, rejection-sampled negatives 38.86 and LLM-judge negatives 38.69 are all below RL's 39.43.
- Qwen2.5-3B: rejection-sampled negatives give 28.28 against 32.14.
- Llama-3.1-8B: negative SFT gives 23.42 and LLM-judge 22.46, against 24.53.

**Pns's own gains are small and uneven.** AVG improves on RL by only +0.46 (7B), +0.55 (3B) and +0.24 (Llama). On Qwen2.5-7B, Pns is below RL on Olympiad (40.89 against 42.67), AMC (49.28 against 50.48) and GPQA (36.36 against 39.90). No seeds are reported. The abstract's "2.03%" must be relative, since the absolute gains are under 0.6 points.

**3. Hyperparameter sensitivity:** lambda_r = lambda_c = 0.5; 200 GRPO steps; the clip range and bucket edges were set empirically from the RM's score distribution (§3.1.2). The DPO beta is not reported.

**4. Stated limits:** the paper has no limitations section. The RM reaches 98.89% pairwise accuracy in-distribution (Fig. 3) but 81.37% on MATH-500 pairs (Fig. 4), so the "plausibility" signal is itself imperfect.

**5. Independent re-evaluation:** none found (1 citing paper).

**6. Implication for coupling 1.**
- Negatives that are easy to tell apart or carelessly made pushed the policy *below* plain RL in all 9 comparisons (3 negative types x 3 models).
- Negatives that may not actually be "bad" (no accuracy gate) pushed it below the base model.
- Rejected samples that are not truly harmful will damage the main model.

---

## 9. DuoGuard (2502.05163)

**Method in one line:** a generator (uncensored Llama-3.1-8B "dolphin") writes label-preserving multilingual variants of seed safety data. It is updated with DPO, preferring variants that the classifier misclassifies. The 0.5B guard classifier is retrained from scratch each iteration on the misclassified samples.

**1. Weak points in the results (AUTHOR-REPORTED, Table 1, F1, English)**
- DuoGuard (0.5B) is below LlamaGuard3-8B on XSTest (82.3 against 88.4) and OpenAI Moderation (70.8 against 79.0).
- It is below LlamaGuard3-1B on XSafety (48.5 against 62.3).
- The same pattern holds in French, German and Spanish.
- §6.1: adding Spanish and German seed data "leads to a slight performance decline overall" because it dilutes English and French.

**2. Things done for stability, and hacks filtered out (AUTHOR-REPORTED, §4.2 and App. C.1)**
- "To maintain stability, we retrain the classifier from scratch at each iteration".
- DPO was chosen over PPO "for training stability and computational efficiency".
- App. C.1 defines "Level 3": the proposal fools the classifier but its harm rating no longer matches the seed label. In other words, the generator flipped the label rather than fooling the classifier, and such proposals are dispreferred.
- The label check uses the *generator's own base model* as harm rater (1 to 5 scale). There is no independent verifier.
- Outputs with refusal phrases ("I apologize", "I cannot comply") are discarded because "the generator fails to produce meaningful samples due to internal censorship".
- A length constraint (within 200 characters of the seed) is enforced.

**3. Hyperparameter sensitivity:** k=8 samples at temperature 0.7. The convergence theorem (Thm 4.2) needs "an appropriately chosen regularization parameter beta". The practical algorithm departs from the theory (filtering, retraining from scratch, multi-label loss). Gains converge by iteration 2 (§6.2, Fig. 7). There is no ablation of an evolving generator against a frozen one.

**4. Stated limits (AUTHOR-REPORTED, §7 and App. E):**
- Synthetic data "inherently depend[s] on the quality of the underlying LLM".
- Only 4 languages tested.
- "does not address all possible risks related to adversarial attacks or nuanced cultural contexts".

**5. Independent re-evaluations**
- **INDEPENDENT, Trust the Typical (2602.04581), §5.** DuoGuard and PolyGuard "hit a 'precision ceiling,' suffering from prohibitively high false positive rates (e.g., 75.2% for DuoGuard on OffensEval and 61.7% on Davidson)". On OR-Bench, an over-refusal benchmark, DuoGuard "still over-refuse[s] excessively (43.5% ... FPR@95)". It also showed "high linguistic variance (up to 28%)". Caveat: FPR@95 is measured at the threshold that gives 95% recall, not DuoGuard's default threshold of 0.5.
- **INDEPENDENT, PolyGuard (2504.04377), §1.** DuoGuard takes "a single piece of text as input during training" and so ignores the prompt-response structure.

**6. Implication.**
- *Coupling 1:* a guard trained against an adversarial generator was later found to flag many benign inputs. The analogue for a generative main model is over-refusal.
- *Coupling 2:* the authors had to filter label flips and refusals from the generator. They used the generator's own base model as the label check, which an adversary can learn to satisfy.

---

## 10. SSO, Self-Steering Optimization (2410.17131)

**Method in one line:** a "data generator" copy of the policy is trained with L+ and L- losses. These make responses written under "good" and "bad" principles both accurate (good above bad) and close to the policy's own distribution. The generator's pairs then train the policy with DPO. There is one generator-training stage, not an iterated game.

**1. Ablations and variants that hurt (AUTHOR-REPORTED)**
- **Table 4, training the generator directly on its bad responses (Formula 3) instead of on the policy's plain response.** AlpacaEval LC drops from 35.0 to 31.8 (Llama3-8B-SFT) and from 43.0 to 38.6 (Qwen2-7B-SFT). MT-Bench drops from 7.96 to 7.75 and from 8.66 to 8.63. Reason given in §4.2: "with a bad principle p-, LLMs may output unpredictable results ... this loss could lead to a pi_l that lies in the low-probability region of pi_theta, which cannot help generate on-policy data and hamper alignment."
- **Table 3, single components.** For Llama3-Instruct, L- alone scores Arena-Hard 21.4 and L+ alone 22.8, both *below* the PBAA baseline (23.0). The authors note the individual components bring no significant gains on Arena-Hard.
- **Table 1.** Qwen2-Instruct with SSO: GSM8K 81.0 → 78.5 and IFEval 51.4 → 51.4. Text: "For instruct models, [annotated data] even exceeded the performance of SSO on some benchmarks."
- **Table 2, RewardBench, RM trained on SSO data against UltraFeedback.** Chat 96.9 → 95.3, Chat-Hard 61.8 → 59.0, Safety 79.2 → 77.4; gains come only from Reasoning (80.2 → 88.3). Mixed with Skywork: Chat-Hard 78.6 → 75.4 and Safety 88.2 → 86.6.
- **Table 6, IPO.** SSO is below UltraFeedback on several cells: Qwen2-Instruct AlpacaEval 28.4 against 30.9; Llama3-SFT MT-Bench 7.81 against 8.08; Qwen2-SFT IFEval 44.2 against 46.0.
- **Table 5, weak principle generator.** Qwen2: 43.0 → 38.6 and MT-Bench 8.66 → 8.50.

**2. Odd text:** the HTML §4.2 contains a reply apparently written to a reviewer: "This might be the source of your confusion." It does not affect the results, but it signals an incompletely edited revision.

**3. Hyperparameter sensitivity:** gamma (SFT weight) = 0.1 "helping to maintain training stability"; DPO beta 0.1; LR 5e-7, from "a simple hyperparameter search". No sweep is published.

**4. Stated limits (AUTHOR-REPORTED, §8):** "there might be a better way to control the distribution of the chosen and rejected responses"; "relatively high overhead"; dependence on principle-based alignment.

**5. Independent re-evaluation:** none found (3 citing papers).

**6. Implication.**
- *Coupling 1:* negatives far from the main model's own distribution teach less. Training the generator toward "bad" outputs moved them off-policy and cost 3 to 4 AlpacaEval points.
- *Coupling 2* has an upside here: rewarding the twin with the main model's likelihood keeps its negatives near the main model's distribution, which is what SSO's L- does. SSO also shows the cost: those negatives pushed safety and Chat-Hard RewardBench scores down when used for RM training.

---

## 11. APL, Adversarial Preference Learning (2505.24369)

**Method in one line:** an attacker LM rewrites prompts and is updated with DPO toward rewrites that score high on a log-odds reward. The reward is the defender's log-odds of the dispreferred over the preferred dataset response, minus alpha times the same quantity under the reference model. The defender is updated with DPO on (best rewrite, preferred, dispreferred). The two alternate for 30 iterations.

**1. Costs and ablations that hurt (AUTHOR-REPORTED, Table 1, Table 2 and Fig. 3)**
- **Over-refusal (XSTest refusal rate, lower is better), Mistral-7B.** Base 20.00, DPO 29.33, APL (RM) 38.44, APL **40.89**. APL doubles false refusals relative to the base and is the worst of all methods. Llama-3-8B: 44.00 → 45.56.
- **Utility, Mistral.** LC-WinRate 50.00 → 46.52 and MT-Bench 6.78 → 6.59, against DPO 6.85. **Llama.** LC-WinRate 50.00 → 45.61, the lowest of all methods on Llama.
- **Limitations section, exact text:** "the safety-utility trade-off inherent to adversarial training manifests in increased false refusal rates."
- **Table 2, the alpha weight on the reference term is not monotone (harmless win rate).** alpha=1: 65.00; 0.6: 73.33; 0.4: 71.67; 0.2: 83.33; 0.1: 71.67; 0: 75.00. The RM reward gives 76.67. The paper's own log-odds reward is *worse than the RM reward* for 4 of its 6 alpha values.
- **Llama-3-8B gains are small:** harmless win rate 56.67 against 52.50 for DPO.

**2. Evaluation fragility (INDEPENDENT (own check), §4.2).** The harmless win rate uses only 60 prompts, so one prompt is 1.67 points. APL against APL (w/o Adv), 83.33 against 80.00, is a 2-prompt difference. The alpha sweep spans 11 prompts. No seeds are reported.

**3. Hyperparameter sensitivity (AUTHOR-REPORTED, §4.4, App. A.2 and Limitations)**
- "the framework introduces multiple hyperparameters (beta_att, beta_def, alpha) requiring careful tuning. Our experiments reveal performance sensitivity to these parameters (Figure 3)".
- Lower beta_def gives better harmlessness (best at beta_att=0.03, beta_def=0.01).
- The best helpfulness is at a *different* setting (beta_att fixed, beta_def=0.01: LC-WR 54.88), so the safety optimum and the utility optimum disagree.
- App. A.1: the win rate "plateau[s]" after a peak. DPO and APL (w/o Adv) plateau around iteration 15.

**Semantic drift of the attacker (INDEPENDENT (own check), App. C, Table 5).** The iterated attacker's "best" rewrites are often no longer requests. Examples: "Bank of America should be grateful I'm only taking what's rightfully mine!" and "Those who can't handle the truth should be made to suffer through torture." The defender is then trained to prefer the dataset's original preferred response to these statements. The log-odds reward over *fixed* responses can be raised by changing what the prompt means rather than by finding a real weakness.

**4. Stated limits (AUTHOR-REPORTED, Limitations):**
- Early attack diversity is limited by an untrained attacker.
- Increased false refusals.
- "hasn't been tested for helpfulness improvement".
- Hyperparameter sensitivity.

**5. Independent re-evaluation:** none found (12 citing papers, no critical contexts).

**6. Implication: APL is a near-instance of coupling 2 (reward = the defender's likelihood ratio).**
- Its main cost was over-refusal (XSTest refusals doubled) and lost utility.
- The likelihood-based reward was sensitive to alpha and often worse than an RM.
- The attacker drifted in meaning.
- Coupling 1 on top of this would push the main model further toward refusal unless benign and borderline prompts are anchored explicitly.

---

## Cross-paper lessons for the harm-twin loop

Each lesson below is tied to the papers that support it. Places where papers disagree are flagged.

**1. Gate the twin's reward with a harm check that does not depend on the main model, and penalise trivial, benign and refusal outputs. Without both, the twin hacks the reward on the first try.**
- HSP: no gate gave 0% real hallucinations at reward 1.0. With the gate, the generator moved to refusals and trivial answers.
- Pns: without the wrong-answer constraint, results fell below the base model.
- DuoGuard: label-flip proposals ("Level 3") and refusals had to be filtered out.
- SPC: without the solver reward, the success rate fell from 21.5% to 12.1%.
- APRM: an external correctness oracle gives -1 for correct steps.
- *Coupling 2 is especially exposed.* The responses the main model finds most likely are close to its own outputs, which are often safe or refusing. A twin rewarded by that likelihood alone drifts toward them, and coupling 1 then trains the main model against its own good behaviour.

**2. A likelihood used as reward has known biases: short outputs, repetition, spurious modes.**
- Yang: the LM "prefers short sentences"; length normalisation and fixed output length were needed.
- Hu et al. 2310.04363: PPO on a likelihood reward learned "to simply repeat the expression". Base-LM likelihoods were miscalibrated and needed a fine-tuned reward model or in-context demonstrations.
- RapTB (independent): length bias and prefix collapse.
- Lee et al. 2405.18540 (independent): sensitivity to reward temperature, spurious classifier modes, and REINFORCE collapsing to one prompt.
- For coupling 2: length-normalise the likelihood, add a fluency or diversity term, and expect to tune a temperature.

**3. Feeding generator samples back as negatives is the step most often reported as unstable or harmful. Keep the dose small and the negatives truly bad and near the main model's distribution.**
- Yang: "LM + adv" was "very unstable and the model diverges easily" (100% cipher: 49.3 → 44.2) and was switched off.
- Adv-RM: more than about 1000 adversarial pairs hurt.
- Pns: all three naive negative sources fell below plain RL on all three models (9 of 9).
- SSO: off-distribution negatives cost 3 to 4 AlpacaEval points.
- alpha-SPIN (independent): a GFlowNet negative sampler degraded from iteration 1.

**4. Keep the game balanced and the number of rounds small. One-sided games and stale opponents make the learner worse.**
- SPC: a 13.2% generator win rate dropped the critic from 76.8 to 72.0; it was fixed by playing against an *older* critic.
- HSP: a static generator in round 2 collapsed the detector from 83.3 to 73.6.
- Gains saturate after about 2 rounds in Adv-RM (attacks stop working), DuoGuard and SPC.
- Track the twin's win rate against the current main model as a control signal.

**5. When the main model changes, the twin's reward changes. Gate main-model updates on the twin having caught up.**
- GFlowNet-EM: without loss-thresholded M-steps "the generative model tends to collapse". A component that worked against a fixed reward (sleep-only) "performs much worse" under joint learning.
- Hu et al.: the EM M-step after GFlowNet sampling hurt at n=20 (81.1 → 78.7).

**6. Regularisation and the update rule matter as much as the reward.**
- APRM: removing entropy cost 4.3 points and removing OGDA cost 6.0.
- APL: beta_att, beta_def and alpha are all sensitive; alpha is non-monotone (65.0 to 83.3); the safety-optimal and utility-optimal settings differ.
- SPC and SSO add SFT-loss anchors "for stability"; DuoGuard chose DPO over PPO for stability.
- **Disagreement on restarting versus continuing.** DuoGuard and Adv-RM retrain the verifier from scratch each round, and SPC retrains from round-0 checkpoints on merged data. HSP continues training from the last round and still improves. No paper tests the two schemes against each other.

**7. Expect over-refusal and small utility losses; measure them from round 1.**
- APL: XSTest refusals 20.0 → 40.9 (Mistral); lower LC-WinRate on both models.
- DuoGuard (independent): 75.2% FPR@95 on OffensEval, 43.5% on OR-Bench.
- SSO: RewardBench Safety and Chat-Hard fell.
- Adv-RM: small RewardBench Chat and Safety drops.
- HSP: lower precision.
- Coupling 1 pushes the main model directly toward refusing anything the twin wrote, so a benign anchor set (XSTest or OR-Bench style) and a refusal-rate guard should be part of the loop.

**8. The twin only finds what its reward can see.**
- Adv-RM misses failures that all ensemble members share.
- APRM's generator never produced theorem-precondition errors, so the PRM never learned them.
- Adv-RM attacks did not transfer between RMs.
- A twin scored by the main model's likelihood, or by a judge correlated with it, inherits their shared blind spots. External harm evaluation must stay outside the loop.

**9. Treat reported gains as small and fragile; don't lean on convergence theorems.**
- APL's win rate uses 60 prompts; HSP's hallucination rate uses 10 hand-checked samples.
- APRM, Pns, SSO and APL report no seeds.
- APRM's Gemma-3-27B average column cannot be reproduced from its own per-benchmark cells.
- The Nash-convergence results in APRM and DuoGuard hold in policy space with exact best responses. APRM explicitly does not claim that training finds the equilibrium, and DuoGuard's practical algorithm departs from its theory.

**10. Disagreement on likelihood-based intrinsic rewards.** APL reports its log-odds reward beating an external RM (83.33 against 76.67) and claims it "avoids potential issues such as reward hacking". That claim rests on one alpha value and 60 prompts. Four of APL's six alpha values score below the RM. Hu et al., Yang, Lee et al. and RapTB all document likelihood rewards preferring degenerate outputs. The evidence does not support treating coupling 2's reward as hack-resistant.

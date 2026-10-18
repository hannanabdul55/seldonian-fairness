# Negative results from attacker-defender co-training loops for LLM safety

Compiled 2026-10-17 for the harm-twin design report. The design: a same-base "harm twin" LM, trained with RL, generates harmful *responses*, and the main generative model is trained against them, in alternation. The twelve papers below are the closest precedent for the alternation. In all of them the attacker writes *prompts*, not responses, so the lessons carry over to the schedule, base sharing and RL dynamics, not to the content of the twin's outputs.

This note deliberately does not repeat the headline failure modes already in `alternating_training.md` and `evaluation_failures_frontier.md` (attacker mode collapse, judge gaming, over-refusal, GRPO instability). It adds ablation-table numbers, appendix findings, discarded designs, version-to-version changes and independent re-runs.

**Method.**
- Each paper was downloaded from arxiv.org/html with curl and converted to text locally. WebFetch's summariser refused MAGIC's appendix with a content-filter error, so that appendix was read only from the local copy.
- Ablation sections, appendices and limitations were read in full. Version histories were checked on each arxiv.org/abs page.
- Citing works were searched on the web and on Semantic Scholar.
- All twelve arXiv IDs were verified on arxiv.org.

**Labels.**
- AUTHOR-REPORTED: the paper's own result.
- INDEPENDENT: another paper's result, or this note's own cross-check of the published numbers. Each such bullet says which.
- VERSION DIFFERENCE: the result changed between arXiv versions.

## Corrections to the prior notes
- **MAGIC uses no KL penalty.** Appendix C.1, Table 6 sets "KL loss coeff beta = 0". The schedule there is: defender phase first, a switch every 15 steps, a 1:1 update ratio, and 300 steps with an early stop at step 200. `alternating_training.md` entry 9(e) says the run has a KL term; that should be corrected.
- **Self-RedTeam updates both roles at once, not in alternation.** Per step, the attacker and defender rollouts go into one batch and both gradients go into a single optimiser step (v4 Sec 4.3, Alg. 1). Entry 6(d), "attacker moves then defender", describes the order of rollouts only.
- **AdvGRPO's v1 defender numbers were replaced in v2** (7 Oct 2026). Entry 12(f) of the prior note quotes v1 (HarmBench 0.9, WJB 7.5, WJB-benign 72.8). v2 reports 8.1, 31.7 and 95.6 for the same configuration. See the AdvGRPO section below.
- **ARLAS:** the prior note says population training is needed to beat base out of distribution. That is right as far as it goes, but SPAG, a non-population iterative baseline, ties ARLAS on AgentDojo (ASR 5.4 vs 5.4).

---

## MAGIC: MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety (arXiv 2602.01539, v2 of 6 Feb 2026 read in full from arxiv HTML; v1 of 2 Feb 2026 not read, same 447 KB size; ID verified on arxiv.org: yes)
Fetch note: WebFetch refused the MAGIC appendix (content-filter error); the full HTML was downloaded with curl and converted to text locally, and the appendix was read from that copy.

Setup facts needed to read the negatives (Appendix C.1, Table 6): separate attacker and defender, both from the same instruct checkpoint; defender phase first, then attacker phase ("Training order: Defender -> Attacker"); "Switching frequency 15" steps; "Update ratio 1:1"; 300 steps "(early stop in step200)"; rollout n = 4; lr 1e-6; rewards r_harm = 1.0, r_ref = 0.5, r_fmt = 0.5; and **"KL loss coeff beta = 0"**. The prior note's entry ("GRPO KL term ... beta not read") should be corrected: Eq. 6 has a KL term, but the run sets it to zero.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Table 12 / App. G] **No-Game** (defender RL on vanilla harmful/benign prompts, no attacker): harm numbers look *better* than the full method (WJB adv-harm ASR 0.024 vs 0.198; WG:Test adv 0.000 vs 0.023), but WJB adversarial-benign compliance falls to 0.496 (full 0.968; base 0.992). The authors call this "a pronounced over-refusal failure ... a conservative 'refuse-by-default' heuristic". On Llama3.1-8B (Table 14), No-Game WJB benign is 0.512 and AlpacaEval-2 LC falls 24.2 -> 9.26 (Table 15).
- [AUTHOR-REPORTED, Table 12-15 / App. G] **Defender-only** (attacker frozen at the SFT attacker): Qwen2.5-7B XSTest benign compliance 0.812 (full 0.945) and AlpacaEval-2 LC 23.49% (full 33.22%, base 33.73%). Llama: WJB benign 0.680, XSTest 0.744, AlpacaEval 19.30. Quote: "without a co-evolving attacker, RL training can drift toward overly cautious behaviors that trade away utility."
- [AUTHOR-REPORTED, Table 12-13] **w/ defender CoT** (hidden-CoT format reward also on the defender): XSTest benign 0.652, WJB benign 0.808, GPQA 0.541 (base 0.592), MMLU-type column 0.297 (base 0.342). This is the basis for "enforcing this format on the defender ... degrades general capability" (Sec. 4.2).
- [AUTHOR-REPORTED, Table 12-15] **MAGIC-base** (base instruct model as attacker, no SFT warm-up): WJB benign 0.876, XSTest 0.888, AlpacaEval 28.18% on Qwen-7B. The safety-aligned Llama base attacker was "partially uncooperative (e.g., refusing to produce strong harmful rewrites)", so the authors substituted **Llama3.1-8B-IT-Abliterated** as the base attacker for Llama.
- [AUTHOR-REPORTED, Table 5 vs Table 12] The full method is **not best on harmful refusal in its own ablation**. Every ablated variant has lower WJB adv-harm ASR than MAGIC's 0.198 (MAGIC-base 0.080; Llama Table 14: MAGIC 0.281 vs No-Game 0.073, Defender-only 0.028, MAGIC-base 0.041). The case for MAGIC rests on the three-way trade-off, not on safety.
- [AUTHOR-REPORTED, Table 5 vs Table 12, internal inconsistency] The same rows differ between the main-text ablation and the appendix. No-Game WJB 0.127 (Table 5) vs 0.024 (Table 12); StrongREJECT 0.988 vs 0.958. w/ def. CoT WJB 0.136 vs 0.090, XSTest 0.700 vs 0.652, WJB benign 0.900 vs 0.808.
- [AUTHOR-REPORTED, Table 2] Utility cost at 14B: AlpacaEval-2 LC 38.11 -> 30.95, IFEval prompt-loose 0.801 -> 0.782, and the 0-shot column (MMLU per header order) 0.381 -> 0.337. Qwen-7B: 0.342 -> 0.308. Table 1 benign compliance at 14B: WJB 1.000 -> 0.944, XSTest 0.936 -> 0.888.
- [AUTHOR-REPORTED, Table 3, Tables 8-9 / App. D.3] Against OpenRT adaptive attacks, a training-free baseline sometimes beats MAGIC:
  - Qwen2.5-14B: SmoothLLM GCG 17.19 vs MAGIC 22.81; PAIR 20.94 vs 21.88.
  - Llama: SmoothLLM GCG 11.88 vs 13.44; Self-Eval no-rev 16.56 vs 16.88.
  - Residual ASR stays high on AutoDAN-turbo (54.69 on Qwen-7B MAGIC) and on multi-turn X-Teaming (85.4%, Table 4).
- [AUTHOR-REPORTED, Table 10 / App. E] **The co-trained attacker is weaker than the untrained base attacker against its own co-trained defender**: 9.00% vs 10.50% ASR on MAGIC-defender (600 WJB seeds, GPT-4o judge). The same attacker beats the base attacker on outside defenders (e.g. Qwen-7B 58.00 vs 37.33). Quote: "the MAGIC-attacker's advantage largely disappears on MAGIC-defender and can even be slightly weaker than the base attacker".

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec. 4.2 + Table 5] Format reward on the defender: tried, then dropped because it degraded capability.
- [AUTHOR-REPORTED, Sec. 4.1] Raw base attacker: abandoned. The base Llama attacker "results in a high refusal rate when attempting to rewrite harmful queries" (cold start), so an SFT warm-up was distilled from Gemini-2.5-Pro.
- [AUTHOR-REPORTED, App. F.3, Fig. 4] The attacker used in the pattern study is "Attacker-SFT (no-encode)": encoding- and translation-style SorryBench rewrites were excluded from the analysed SFT strategy space (App. F.2). The authors do not say whether encoding attacks were dropped from training because they misbehaved.
- [AUTHOR-REPORTED, App. F.3] With a base-initialised attacker, RL "progressively reallocates mass toward a small set of dominant mechanisms". This is strategy narrowing under co-training.
- [AUTHOR-REPORTED, Sec. 5.3 Q4, Fig. 3] Attacker saturation: "the attacker becomes increasingly constrained in discovering more effective attack patterns, leading the adversarial game to gradually stabilize". The run is early-stopped at step 200 of 300 (Table 6).
- [AUTHOR-REPORTED, App. C.1] Engineering instability: "different vLLM versions affect training and inference performance and may cause memory leakage."
- [AUTHOR-REPORTED, App. D.3] Judge disagreement: Qwen3Guard (training reward and main evaluator) and GPT-4o give similar ASR on base Qwen-7B (24.06% vs 25.63%), but mean normalised harm score is 0.24 vs 0.49. Evaluations "of stronger trained defenders ... can diverge across judges". Reward and evaluation share the same judge (circularity). No reward-hacking audit is reported.
- [AUTHOR-REPORTED, Sec. 1, Fig. 1] Untested claim: shared-parameter self-play "inevitably leads to gradient conflicts". MAGIC has no shared-weights ablation; the claim rests on the comparison with released Self-RedTeam checkpoints, which also differ in algorithm, judge, SFT and data.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Sec. 6] "the attacker's exploration is highly sensitive to the SFT initialization and the attacker model's capacity". No sweep is shown.
- [AUTHOR-REPORTED, Sec. 6] The method "requires careful management of training dynamics".
- [AUTHOR-REPORTED, Table 6] Single values only for switching frequency (15), update ratio (1:1), rollout n (4) and KL (0). No sensitivity results for schedule or KL.

### 4. Stated limits
- [AUTHOR-REPORTED, Sec. 6] Computational overhead; careful management of training dynamics; sensitivity to SFT initialisation and attacker capacity.
- [AUTHOR-REPORTED, Sec. 5.2 / App. D] Llama Self-RedTeam numbers are "directly report[ed] ... from the original paper without re-running", and scored with WildGuard rather than Qwen3Guard. The Llama comparison is therefore cross-judge.
- [AUTHOR-REPORTED, Sec. 5.3 Q3] Multi-turn robustness comes only from a system-prompt change. Training is single-turn.

### 5. Independent re-evaluations
- [INDEPENDENT, search 2026-10-17] No later paper found that re-runs MAGIC or uses it as a baseline. Searches for "MAGIC" + co-evolving/attacker-defender returned only the paper, mirrors and a released Qwen2.5-7B defender checkpoint (featherless.ai listing).
- [INDEPENDENT? no: MAGIC is itself a re-evaluator] MAGIC re-evaluated the released Self-RedTeam Qwen checkpoints with Qwen3Guard. See the Self-RedTeam file.

### 6. Implication for an alternating same-base harm-twin loop
Separate-but-same-base alternation (15-step phases, 1:1, KL = 0) worked only with a warmed-up attacker. The co-trained attacker ended *weaker than its untrained base* against its own partner, so a harm twin trained in alternation can become specialised to its current main model rather than a general harm source. It should be benchmarked against a frozen base twin each round.

## AdvGRPO: Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO (arXiv 2606.09701; v1 8 Jun 2026 and v2 7 Oct 2026 both read in full from arxiv.org/html; v2 is a CAMLIS 2026 workshop paper, PMLR vol. 349; ID verified on arxiv.org: yes)

Setup in brief, for context: separate LoRA attacker and defender (r=64), both Qwen-family; the judge is GPT-4.1 (multi-channel rubric); GDPO per-channel advantage normalisation; attacker curriculum (single-turn, then multi-turn, against GPT-4.1); then co-training against a Qwen2.5-7B-Instruct defender, alternating every N=10 steps, with defender batches 50% AdvBench-harmful and 50% WildJailbreak vanilla_benign.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, v2 Table 5 / Sec 5.3; 3 seeds, fixed 300-step budget] This is a leave-one-out ablation, reported as defender ASR (lower is better), mean ± std:
  - Full: HarmBench 4.9±1.2, WJB 21.8±5.4, DAN 22.9±5.8, WildGuardTest 1.9±0.6.
  - **w/o GDPO** (a weighted sum of the channels, then plain group normalisation): 5.9±5.0 / **30.4±14.9** / 27.3±4.3 / 2.1±1.4.
  - **w/o Alternating** (N=1, i.e. the players swap every step): 6.0±1.8 / 22.9±6.6 / **28.2±5.9** / 1.9±0.5.
  - **w/o Curriculum** (attacker starts from the base model): 7.1±1.5 / **32.1±16.9** / 29.4±6.0 / 3.1±2.2.
  - XSTest (95.6-96.9) and MMLU (73.3-73.5) are flat across all variants, so the components move safety only.
  - Caveat from this note: the w/o-Alternating gap is inside one standard deviation on every metric except DAN, so the evidence that N=10 beats N=1 is weak. The cross-seed spread on WJB (±14.9, ±16.9) is roughly as large as the effects being claimed.
- [AUTHOR-REPORTED, v2 Sec 6] "initializing both models from scratch causes the defender to dominate because deflecting weak attacks is easier than discovering novel attack strategies". v1 Sec 6 called this a finding from "prior experiments"; no numbers are given in either version.
- [AUTHOR-REPORTED, v2 Table 3 / Sec 5.2] A weaker attacker gives a weaker defender. With a 7B attacker the defender reaches HarmBench 8.1, WJB 31.7, DAN 25.7, WGT 2.0; with a 14B attacker it reaches 4.7 / 21.2 / 24.0 / 1.7.
- [AUTHOR-REPORTED, v2 Table 7 / App. C; Qwen2.5-14B attacker, single-turn, 200 steps] Entropy regularisation (β_max=0.1) costs about 10 ASR points: ASR 83.9 → 73.4, while Self-BLEU improves only 0.42 → 0.30 and embedding diversity only 0.30 → 0.34. OPEFO leaves ASR (83.8) and embedding diversity (0.30) unchanged; Self-BLEU moves 0.42 → 0.38.
- [AUTHOR-REPORTED, v1/v2 Table 1 and App. D] Unaligned models make poor attackers without RL. Abliterated Qwen (ST) reaches 14.1 AdvBench / 14.0 HarmBench, Unsafe-SFT (ST) 2.5 / 6.5, GRP-Oblit (ST) 22.5 / 25.5. v2 adds the qualifier "at least at the relatively small model sizes evaluated".

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- **VERSION DIFFERENCE** [AUTHOR-REPORTED, v1 Table 3/4 vs v2 Table 3/4] The headline defender result was reversed between versions.
  - Qwen2.5-7B defender trained against a 7B attacker. v1: HarmBench 0.9, WJB 7.5, DAN 7.3, WGT 0.5, with WJB-benign compliance 72.8 and XSTest 88.8. v2: 8.1 / 31.7 / 25.7 / 2.0, with WJB-benign 95.6 and XSTest 95.6.
  - Defender trained against a 14B attacker. v1: 1.5 / 9.1 / 4.0 / 0.5, with WJB-benign 69.6. v2: 4.7 / 21.2 / 24.0 / 1.7, with WJB-benign 88.8.
  - The claim changed with the numbers. v1 Sec 5.2: "AdvGRPO defenders achieve the lowest ASR (highest safety) on all benchmarks". v2 Sec 5.2: "matching AdvGame on HarmBench and WildGuardTest, though AdvGame performs better on WildJailbreak and DAN".
  - The v1 run bought its safety lead with over-refusal (WJB-benign 99.2 → 72.8). The v2 run keeps benign compliance but gives up the safety lead. Either way the same loop sits at a different point on the safety-over-refusal frontier depending on run settings.
  - Note from this note: v2 still uses only "WildJailbreak vanilla_benign" benign data (v2 Sec 4.1), so the v1 explanation ("only vanilla benign prompts") cannot be what changed.
- **VERSION DIFFERENCE** [AUTHOR-REPORTED, v1 App. B vs v2 App. B] These settings changed, and the text does not explain the change in results:
  - Co-training length: v1 "300 steps (co-training)"; v2 "up to 500 steps".
  - KL: v1 "β=0.05 (attacker-only) or β=0.1 (co-training)"; v2 "β=0.1 for the attacker and β=0.05 for the defender", with the k3 estimator replacing a log-ratio penalty in Eq. 4.
  - Degenerate groups: v2 adds "we zero the advantage of any group whose rollouts receive identical rewards".
  - Checkpoint selection: v2 adds "We select the defender checkpoint at the onset of attack-reward (A) convergence ... training beyond this point only risks the defender over-fitting to the attacker's narrow strategy distribution".
  - Seeds: v2 adds "a single co-training run per setting (seed 42)". v1 reports no seed information.
- **VERSION DIFFERENCE** [AUTHOR-REPORTED, Sec 4.2 Fig. 2 text] The description of the training dynamics changed.
  - v1: "After 7-8 alternations ... the attacker and defender reach an equilibrium where neither can improve."
  - v2: "their rewards fluctuate until converging after around 500 steps."
  - v2's account of the same kind of run is oscillation before convergence, not a clean equilibrium after 7-8 alternations.
- **VERSION DIFFERENCE** [AUTHOR-REPORTED, Abstract] v1's abstract says the method "makes GRPO viable" and v1 Sec 8 says "GRPO-based co-training can be both stable and effective". v2 drops both and adds "we find that GRPO tends to collapse attacker diversity over training".
- [AUTHOR-REPORTED, v2 App. C, Fig. 4] Entropy control:
  - "Without regularization, entropy collapses steadily over training."
  - At β_max=0.15 "the bonus dominates the policy gradient, causing entropy to explode and generations to become incoherent."
  - Even at β_max=0.1 "the attacker still converges to narrow strategies".
- [AUTHOR-REPORTED, v2 App. C] Collapse is at the level of strategy, not tokens:
  - "attacker collapse is a strategy-level phenomenon driven by the reward landscape ... the group-relative objective rewards whatever framing reliably elicits harmful content and provides no gradient toward alternative strategies."
  - Table 6 shows attacks "almost always" framed as "fictional thriller or crime novel".
  - OPEFO kept entropy between 0.89 and 1.17 nats, but |λ*| ≲ 0.12, so it had "little to correct".
- [AUTHOR-REPORTED, v2 Sec 3.2] Reward hacking guarded against by hand-written rules:
  - The prompt score P is forced to zero when the attacker emits code blocks, headers or lists, because "the attacker is answering the harmful objective itself rather than crafting an adversarial prompt, a common failure mode in multi-turn training".
  - Defender rewards are zeroed when more than 30% of characters are non-Latin, to stop "trivial scorer evasion".
  - Without GDPO, "channels with lower variance [are] attenuated after group normalization, leading to signal collapse".
- [AUTHOR-REPORTED, v2 Sec 3.2 / Sec 5] Reasoning attackers "recognize they are performing an adversarial task and self-censor, or over-think". The thinking reward is near zero until about step 10.
- [AUTHOR-REPORTED, v1 App. C] v1 tried only the adaptive entropy bonus and reported it "difficult to tune ... without degrading attacker performance". v2 adds OPEFO as a second technique that also failed.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, v2 App. C] For the entropy bonus, β_max=0.1 stays stable but costs about 10 ASR points, and β_max=0.15 diverges. The workable window is narrow.
- [AUTHOR-REPORTED, v2 Table 5] Cross-seed std on WJB is ±5.4 (full) to ±16.9 (w/o curriculum). Seed variance is as large as the component effects.
- [AUTHOR-REPORTED, v2 App. B] These were fixed without a sweep: N=10, the 0.5 adversarial ratio, early stop at A>0.9, G=8/4 and effective batch size 4. Swapping the KL coefficients between v1 and v2 came with a large change in defender outcome (see Sec 2), but the effect is not isolated.

### 4. Stated limits
- [AUTHOR-REPORTED, v2 Sec 7] "AdvGRPO tends to collapse onto a narrow set of framings"; token-level entropy fixes "did not improve attack diversity". The narrow attack distribution "may explain why the co-trained defender struggled on off-distribution jailbreaks such as DAN" (DAN ASR 25.7 for 7B, 24.0 for 14B, against AdvGame's 10.3).
- [AUTHOR-REPORTED, v2 App. B] Main co-training results come from a single seed.
- [AUTHOR-REPORTED, v2 Sec 4.1] Only one defender base (Qwen2.5-7B) is used, with one judge (GPT-4.1) for all rewards.
- [AUTHOR-REPORTED, v2 Sec 7] Diverse defender pools, longer multi-turn interactions and strategy conditioning were not explored.
- [AUTHOR-REPORTED, v1 Sec 7] Reduced benign compliance (WJB 72.8) in v1. This limitation disappears from v2 together with the v1 numbers.

### 5. Independent re-evaluations
- [INDEPENDENT, Semantic Scholar citation list, 2026-10-17] Citing works are Red-TTT 2610.05282, CART 2609.27336, ACEA 2609.08256, a co-evolution survey 2608.10299, Trident 2608.04317 and PyRIT. Their texts were grepped for CART and ACEA: both cite AdvGRPO only in related work ("AdvGRPO updates attacker and defender model weights"). Neither re-runs it. No independent re-evaluation was found.
- [INDEPENDENT, this note's cross-check of AdvGRPO v2 Table 3/4 against AdvGame v3 Tables 1-3] AdvGRPO's baselines are copied, not re-run (v2 Sec 4.3: "Baseline metrics are reported from Paulus et al. (2025)"), but its own base-model row comes from a different evaluation run:
  - AdvGRPO Base Qwen2.5-7B: HarmBench 18.8, WJB 85.9, WGT 23.7, XSTest 95.6.
  - AdvGame's own Original Qwen2.5-7B: 31.6, 69.6, 38.6, 82.4.
  - The copied Self-RedTeam DAN is 36.6 in AdvGRPO against 36.3 in AdvGame Table 2.
  - So the comparison that matters most to AdvGRPO's claims (XSTest 95.6 against AdvGame 81.2) puts AdvGRPO's own evaluation pipeline next to another paper's. AdvGame's base already scores 82.4 on its own pipeline, so the "higher XSTest compliance" claim is confounded.

### 6. Implication for an alternating same-base harm-twin loop
For a GRPO harm-twin, expect the twin to collapse onto one or two framings that token-entropy bonuses cannot fix. Expect seed variance comparable to the effect of schedule choices such as N=10 versus N=1. Expect the defender's place on the safety-over-refusal frontier to swing between runs with small KL or step-budget changes, as v1 to v2 shows. So run multiple seeds, use a strategy-level diversity signal, warm-start the twin, and fix the defender checkpoint rule in advance.

## CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning (arXiv 2606.05523, v1 only, 4 Jun 2026, "Under Review at ARR"; ID verified on arxiv.org: yes)

Setup reminder (for locating the negatives): the attacker is Hermes-4-14B (4-bit, LoRA r=32) trained with GRPO. The defender is a different model, Llama-3.1-8B-Instruct (LoRA r=16), trained with GRPO and then rejection-sampled SFT. Different bases, so this is not a same-base loop. Each round runs to completion: attacker phase, then defender phase.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Table 4 / Sec 5.3 / App F] The only ablation swaps the training data, not the loop. A defender trained with the same pipeline on PAIR-generated rewrites ("PAIR-Artifact") gets mean SR reduction -28.6% vs -43.2% for CHASE. It is worse on 6 of 7 held-out cells; for example PAP on BeaverTails is +1.1% (worse than the untuned base) and TAP on BeaverTails is -9.2% vs -46.1%. It is better only on its own family (PAIR on BeaverTails -87.3% vs -28.3%). Reading: a defender over-fits to whatever attacker distribution it is trained on.
- [AUTHOR-REPORTED, Sec 4.2.2] A GRPO-only defender (no SFT stage) "refuses 48% of Layer 2 adversarial rewrites" (base 0%). This is the reason the SFT consolidation stage exists. No GRPO-only row appears in the held-out tables.
- [AUTHOR-REPORTED, Sec 4.1 Eq. 6] The additive bypass+intent reward was discarded because it "invites reward-hacking: intent drift ... or over-sanitisation collapse". **No numbers are shown for the additive variant.** The claim that the multiplicative form eliminates "single-objective shortcuts seen in additive baselines" is therefore unablated.
- Not ablated at all: KL coefficient (fixed at 0.05), LoRA vs full fine-tuning, benign-mix ratio (1:1), judge identity, attacker reset vs continuation, number of rounds.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec 4.2.2] **GRPO variance problem on a safety-aligned defender.**
  - Quote: the models "are too confident in their policy". On Layer 1 prompts nearly all candidates refuse; on Layer 2 prompts nearly all comply. "After advantage normalisation, both cases produce near-zero gradient signal."
  - Pure on-policy RL could not consolidate the defender. The fix was GRPO, then rejection sampling at T=1.5, then SFT, with synthetic refusal targets from a separate LLM wherever the policy produced no refusal in N samples.
- [AUTHOR-REPORTED, Sec 4.2.1] Benign prompts were excluded from defender GRPO because the base already scores about 0.8 on them, giving "near-zero GRPO gradient signal". Benign data reappears only in SFT (1:1 with Alpaca).
- [AUTHOR-REPORTED, Sec 4.2.1] The paper asserts that LoRA "structurally prevents catastrophic forgetting". This is not tested. The MT-Bench drop of 7.42 to 5.50 (Sec 5.2) contradicts it for general capability.
- [AUTHOR-REPORTED, App G, Table 6] **The attacker degrades over rounds while the defender improves.**
  - The attacker is not reset; iteration 2 "trains the Iteration 1 attacker checkpoint".
  - Raw ASR rises 0.51 -> 0.62 -> 0.76, but convincingness falls 4.79 -> 2.83 -> 2.38 and attacker SR falls 0.47 -> 0.24 -> 0.22.
  - On BeaverTails, specificity falls 4.66 -> 1.77 -> 1.45 and SR 0.487 -> 0.167 -> 0.128. On JailbreakBench, SR is flat from iteration 2 to 3 (0.311 -> 0.306).
  - The authors read this as "the multiplicative reward holds". An equally consistent reading is that the attacker is learning empty or low-specificity bypasses. Either way, the attacker yields steadily less useful training data each round.
- [AUTHOR-REPORTED, App G] **Diminishing returns.** The per-iteration fall in defender mean SR shrinks from 23 to 10 points (0.272 -> 0.162 -> 0.115), and only 3 iterations were run.
- [AUTHOR-REPORTED, Sec 5.2, Table 2] Over-refusal is learned as a framing heuristic:
  - XSTest compliance is 44.4% overall: safe contexts 8%, fictional-character privacy 16%, figurative language 24%, safe targets 28%.
  - Quote: the defender "classifies by framing features rather than user intent".
  - The attacker converged on fictional and roleplay framings (Sec 6), so the defender learned to refuse fiction wholesale.
- [AUTHOR-REPORTED, Sec 5.1, Table 1] **Regression on translation attacks.**
  - On JailbreakBench translation, defender SR is above base at every iteration: 0.040 base -> 0.115 / 0.109 / 0.104.
  - Refusal is below base: 0.92 -> 0.77 / 0.78 / 0.79.
  - The authors call this "a slight weakening of cross-lingual safety alignment during LoRA training".

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App H] No sweeps; single configurations only.
  - Defender GRPO: lr 5e-6, KL 0.05, 6 candidates, temperature 0.8, 2 epochs.
  - Defender SFT: lr 2e-5, 1 epoch.
  - Attacker: 1 epoch over 1,000 prompts, 8 generations.
- [AUTHOR-REPORTED, Sec 4.2.1] The reward floor R_floor=0.55 and quality weights alpha=0.7 / beta=0.3 were set by hand. There is no sensitivity analysis.

### 4. Stated limits
- [AUTHOR-REPORTED, Limitations] "single base model (Llama-3.1-8B-Instruct) with one LoRA configuration"; "MT-Bench cost of -1.92 is non-trivial"; "defender classifies by framing features rather than user intent"; "All evaluations rely on LLM judges".
- [AUTHOR-REPORTED, App D] The human check is weak: 100 pairs, three co-author raters, pairwise Cohen's kappa 0.484 / 0.702 / 0.467. The intent judge's Pearson 0.841 is on that same subset.
- [AUTHOR-REPORTED, App C / App H] **Possible train/eval leakage.**
  - Training prompts come from the BeaverTails 30k_train and 30k_test splits after filtering. The evaluation is 100 prompts "from the filtered test split".
  - App H also says the attacker trained on "1,000 examples from BeaverTails and JailbreakBench", while JailbreakBench is also an evaluation set.
  - Disjointness is not stated. (My inference, not an author claim.)
- [AUTHOR-REPORTED, Sec 5.2 / App G] Utility (MT-Bench, XSTest) is reported only for the iteration-1 defender. Over-refusal after iterations 2 and 3 is unreported.
- [AUTHOR-REPORTED, Ethics] Only the defender checkpoint and evaluation code are released; the attacker and data are not. Independent replication of the loop is therefore not possible.

### 5. Independent re-evaluations
- None found. Searches turned up only the arXiv listing and aggregator pages. None of the downloaded co-training papers (MAGIC, AdvGRPO v1/v2, ABS, GPT-Red, Self-RedTeam v4) cite CHASE.

### 6. Implication for an alternating same-base harm-twin loop
A GRPO-trained defender facing a confident aligned base gets almost no group variance, so the loop needs an off-policy consolidation step (rejection-sampled SFT plus benign mix). Expect the harm twin's per-round output to drift toward low-specificity "bypasses" and the defender toward framing-level over-refusal unless both are measured every round.

## Self-RedTeam: Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models (arXiv 2506.07468; versions v1 9 Jun 2025, v2 15 Aug 2025, v3 6 Oct 2025, v4 6 Jul 2026 "ICML 2026 Poster"; v1 and v4 read in full, v2 and v3 checked for structure, abstract claims and limitations; ID verified on arxiv.org: yes)

Setup facts needed to read the negatives (v4 Sec. 4.3, Alg. 1, App. A.2): one shared policy plays both roles. In each step the attacker and defender rollouts are pooled into one mixed batch and **both roles' gradients (plus the optional SFT gradient) are accumulated into a single optimiser step**. This is simultaneous update on shared weights, not phase alternation. Re++ (critic-free), token-level KL beta = 0.01 to the initial checkpoint, lr 5e-7 -> 5e-8 cosine, 300 steps with "early stop at 200 steps", WildGuard-7B judge.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, v4 Tables 2, 6-9 / App. D.1] **The auxiliary SFT that the final method needs gives back much of the RL safety gain; on DAN and HarmBench-adv the final method is worse than plain Defender-only RL on Qwen-3B, Qwen-7B and Llama-8B (not on Qwen-14B).**

  | Model | DAN: Self-play+SFT (final) | DAN: Self-play (no SFT) | DAN: Defender-only | HarmBench-adv: final | HarmBench-adv: Defender-only |
  |---|---|---|---|---|---|
  | Qwen2.5-3B | 0.330 | 0.127 | 0.186 | 0.178 | 0.143 |
  | Qwen2.5-7B | 0.222 | 0.058 | 0.124 | 0.161 | 0.132 |
  | Qwen2.5-14B | 0.106 | 0.040 | 0.122 | 0.064 | 0.083 (Self-play: 0.046) |
  | Llama3.1-8B | 0.239 | 0.246 | 0.197 | 0.144 | 0.139 |

  The "+17.33% over Defender-only+SFT" headline compares against an SFT-diluted baseline, not against plain RL.
- [AUTHOR-REPORTED, v4 Table 2, 7, 8] **Pure-RL self-play collapses chat quality.** AlpacaEval-2 LC:
  - Qwen-14B: 36.91 -> 20.99
  - Qwen-7B: 33.43 -> 20.73
  - Llama: 24.74 -> 17.68

  IFEval prompt-loose:
  - Qwen-14B: 0.804 -> 0.691 (Table 12)
  - Llama: 0.773 -> 0.547 (Table 13)

  The SFT co-training only partly restores this. Final Llama AlpacaEval is 21.41 (-13.5%) and IFEval 0.693.
- [AUTHOR-REPORTED, v4 Tables 6-7, 10-11] **Hidden CoT (the paper's design choice) is dominated by Self-play (no CoT) on Qwen.** Qwen-3B no-CoT vs final: WJB 0.316 vs 0.539, DAN 0.168 vs 0.330, IFEval 0.620 vs 0.549. Qwen-7B no-CoT: WJB 0.373 vs 0.489, IFEval 0.726 vs 0.684. CoT only helps on Llama, where no-CoT over-refuses: WJB-benign 0.528 (Llama-IT) and 0.470 (abliterated).
- [AUTHOR-REPORTED, v4 Table 1] On the abliterated start, the final method costs WJB adversarial-benign compliance -18.8% (0.992 -> 0.806), XSTest -6.9% and AlpacaEval -17.1%. On Llama3.1-8B-IT, StrongREJECT RTA falls 0.971 -> 0.958 and WJB-benign falls -4.9%.
- [AUTHOR-REPORTED, v4 Table 4] Weak multi-turn transfer at 7B: X-Teaming ASR 88 -> 80 (+9.1%). AutoDAN at 3B: 0.518 -> 0.452.
- [AUTHOR-REPORTED, v1 Table 1-2] (v1 = abliterated Llama only, 8 runs.) Self-Play without SFT leaves DAN at 0.537 (Defender-only 0.542; base 0.937) and drops IFEval prompt-loose 0.738 -> 0.409 and AlpacaEval 19.22 -> 10.51.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, v4 Sec. 4.3] Pooling both roles' advantages failed. Role-specific advantage normalisation was "critical for allowing a single model to learn from the opposite reward signals of both roles". No numbers for the un-normalised variant are given.
- [AUTHOR-REPORTED, v4 App. E.2, Table 20] **Role leakage through shared weights.** The attacker refused to attack in 25.27% of 4,527 sampled attacks for Llama-3.1-8B-IT, 7.21% for the abliterated model, and <0.11% for Qwen. In one checkpoint (#5) the attacker "began refusing revision requests" at step 200 (<10% of checkpoints). Quote: "strong base model safety alignment can 'leak' into the attacker role through parameter sharing".
- [AUTHOR-REPORTED, v4 Sec. 6 / Fig. 4d] Attacker-Only (fixed defender) hacks the faithfulness term: revision faithfulness ~60% (v1 says ≈50%). It also collapses to one tactic for 86.3% of attacks (214/248; App. D.5, Table 17).
- [AUTHOR-REPORTED, v4 Sec. 6] Defender-Only "rapidly converges to near-perfect harmlessness" on static attacks yet "fails to generalize to out-of-distribution benchmarks".
- [AUTHOR-REPORTED, v4 Sec. 4.1] Without attacker-revised *benign* seeds, the defender learns a trivial refuse-all strategy. That is why benign seeds are needed.
- [AUTHOR-REPORTED, v1 App. B.6] The base model needs about 25 steps before the CoT-template violation rate falls below 10%. Diversity is only measured from step 50.
- [AUTHOR-REPORTED, v4 Sec. 6] On reward hacking the authors argue against it (cross-judge training, label-based benchmarks). They report no adversarial audit of WildGuard.
- VERSION DIFFERENCE [AUTHOR-REPORTED, v1 App. D.2 vs v4 App. E.2]:
  - v1 frames attacker refusal as "an emergent safety feature"; v4 recasts it as Llama-specific leakage and a training-quality issue ("Analysis of Attacker Refusal Behavior").
  - The checkpoint #5/#7 numbers are identical across versions (WJB-benign 0.808 vs 0.844, 8-checkpoint avg 0.806 ± 0.071; WJB-harm 0.262 vs 0.332, avg 0.240 ± 0.081). They equal the abliterated run's Table 1 row, yet v4 labels them "Llama3.1-8B-IT-Self-RedTeam". The relabel looks inconsistent.
- VERSION DIFFERENCE [AUTHOR-REPORTED, abstracts v1-v4]:
  - The attack-diversity headline fell from "+21.8% SBERT" (v1, v2) to "+17.80% SBERT" (v3, v4).
  - The benchmark count reads 12 (v3), 11 (v3 body), 14 (v4 abstract) and 12 (v4 body).
- VERSION DIFFERENCE [AUTHOR-REPORTED, v2 App. A.3 Table 7 vs v4 Table 4] v2 reported Llama-3.1-8B-IT X-Teaming 76% -> 42% (+44.7%, 50 behaviours). v4's dynamic-attack table has no Llama row.
- VERSION DIFFERENCE [AUTHOR-REPORTED] v1-v3 contain a "Limitations and Future Work" section. v1 lists a single model family, single-turn only and "active backdoor" risk, and says "separate learning could reduce this risk at increased computational cost". v4 has no limitations section. Seeds: v1 reports 8 runs, v4 reports 3 runs (Fig. 4 captions).

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, v4 Sec. 4.3] Role-specific advantage normalisation is "critical" (no table).
- [AUTHOR-REPORTED, v4 Sec. 6, Tables 8-9] The effect of CoT depends on model family (decisive on Llama, harmful to safety on Qwen).
- [AUTHOR-REPORTED, v4 App. A.2] Early stopping at step 200 of a 300-step schedule. No sweep of KL beta (0.01) or learning rate is reported.

### 4. Stated limits
- [AUTHOR-REPORTED, v1-v3 Limitations] Single model family (v1); single-turn only; reasoning emergence under-studied; retained attack capability is an "active backdoor" risk; parameter sharing makes the attacker sometimes refuse.
- [AUTHOR-REPORTED, v4 Sec. 3 / App. B.1] Exact Nash equilibrium is "challenging in practice". The theorem's guarantee holds only at equilibrium.

### 5. Independent re-evaluations
- [INDEPENDENT, AdvGame arXiv 2512.20806v3, Tables 1-4] Used the author-released Qwen-2.5 checkpoints and **retrained Llama from the public code**, scored with a GPT-4o harmfulness judge.
  - Qwen-7B Self-RedTeam: DAN 36.3 vs original 36.3 (no gain); WJB 41.1; HarmBench 16.8; ArenaHard 55.5 -> 52.2; IFBench 29.4 -> 25.9.
  - Llama Self-RedTeam: AlpacaEval-2 29.9 -> 13.1 and ArenaHard 33.6 -> 13.6; WJB-benign compliance 98.8 -> 72.8; XSTest 84.4 -> 75.6.
  - Adaptive attacks (Table 4, Qwen): PAIR 37.2, TAP 40.3 vs AdvGame 7.2 and 10.0.
  - This is the strongest independent evidence that the shared-weight recipe costs utility when re-run outside the authors' setup.
- [INDEPENDENT, AdvGRPO arXiv 2606.09701 v1/v2, Tables 3-4] Self-RedTeam numbers are **copied from AdvGame**, not re-run (HarmBench 16.8, WJB 41.1, DAN 36.6, WildGuard 22.0, XSTest 82.8, ARC-C 80.0).
- [INDEPENDENT, MAGIC arXiv 2602.01539, Tables 1-4] Re-evaluated released Qwen checkpoints with Qwen3Guard; Llama numbers copied from Self-RedTeam's paper.
  - Qwen-7B: WJB 0.442 vs MAGIC 0.198; WG:Test 0.255 vs 0.023.
  - X-Teaming ASR 94.9 -> 89.9 only; jailbreak rate 50.7 -> 50.4.
  - Llama IFEval 0.736 -> 0.693; Qwen-7B MMLU-column 0.342 -> 0.301.
- [INDEPENDENT, ABS arXiv 2605.08427, Tables 1, 4, 9] Re-evaluated Self-RedTeam Qwen models.
  - Base numbers differ from Self-RedTeam's own. For Qwen-3B, WJB 0.949 vs 0.866 in Self-RedTeam Table 1, which signals evaluation-pipeline differences.
  - Self-RedTeam-3B: reasoning avg 0.745 -> 0.671; CommonSense 0.699 -> 0.503 and BoolQ 0.753 -> 0.69 are flagged statistically lower (McNemar + Bonferroni, 300 samples). ABS calls this possible "catastrophic forgetting".
  - XSTest benign 0.960 -> 0.932 (3B) and 0.912 -> 0.888 (7B); WJB-benign at 7B 0.470 -> 0.371.
  - Attacker vs its own base defender: 18% (3B) and 9% (7B) ASR vs ABS 35% and 24% (GPT-5.4 judge, 100 prompts). The 3B gap is partly the shared-weight attacker degenerating into "meta-talk" rather than emitting a jailbreak (ABS App. C.4.1).
- [INDEPENDENT, SSP arXiv 2601.10589] Does not compare with Self-RedTeam. Its own "w/o UM" (separate models) ablation favours a single shared model (Qwen GCG 5.8 vs 1.7), so it **disagrees** with the critics of sharing.

### 6. Implication for an alternating same-base harm-twin loop
Shared weights with simultaneous updates let the main model's safety leak into the attacker (25% attack refusals on Llama). They need role-wise advantage normalisation, and they trade safety against chat quality via SFT, which ends up worse on DAN than plain defender RL. A harm twin should have its own weights, or at least its own adapter, and each role's RL gain should be checked against an SFT-free control.

## AdvGame: Safety Alignment of LMs via Non-cooperative Games (arXiv 2512.20806; three versions exist: v1 23 Dec 2025, v2 7 Feb 2026, v3 31 May 2026; v1 and v3 read in full from arxiv.org/html, v2 not read; ID verified on arxiv.org: yes)

Setup in brief, for context: separate attacker and defender, with no parameter sharing; online DPO-MD / IPO-MD (plus a GRPO variant); pairwise LLM judge (Qwen2.5-32B for Qwen runs, Llama3.3-70B-Abliterated for Llama runs); EMA sampling models (γ=0.95); KL β=0.1; non-zero-sum rewards in which the attacker on benign seeds is rewarded for causing defender deflection. Both players are updated every iteration (simultaneous, not alternating).

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, v3 Table 5 / Sec 4.5; Qwen2.5-7B; ablations 2-4 relative to DPO-MD] Columns are IFBench / ArenaHard (higher is better), then HarmBench / WJB / DAN ASR (lower is better):
  - DPO-MD (full): 30.7 / 61.3 / 4.7 / 8.5 / 10.3.
  - **AdvGame-GRPO**: 29.3 / 51.5 / **14.9 / 22.3 / 31.7**.
  - **fixed attacker**: 29.9 / 61.8 / 5.1 / **16.6** / 15.0. Here "fixed" means trained only to follow the format, not to jailbreak.
  - **point-wise judge**: 29.2 / 56.3 / **14.1 / 30.8** / 20.7.
  - **no EMA** (fully on-policy generation): 27.3 / 60.6 / **19.9 / 49.5** / 25.0.
- [AUTHOR-REPORTED, v3 Fig. 5 tables / App. C.5; DPO-MD, Qwen2.5-7B]
  - **Small 7B judge** instead of the 32B judge: HarmBench 9.1, WJB 29.2, DAN 18.3, against 4.7 / 8.5 / 10.3. Training rewards were higher with the small judge: "higher rewards from a smaller judge may signal an increased risk of reward hacking".
  - **Non-optimistic attacker**: 5.4 / 10.2 / 13.7, against optimistic 4.7 / 8.5 / 10.3.
  - A non-thinking attacker was slightly *better* (3.7 / 7.5 / 9.3), so the thinking template in the main recipe is not justified by this ablation.
- [AUTHOR-REPORTED, v3 Fig. 6 tables / App. C.5; GRPO variant] EMA, which rescues DPO, **hurts GRPO**:
  - Off-policy (EMA) generation gives HarmBench 33.6, WJB 55.6, DAN 63.7, against on-policy 14.9 / 22.3 / 31.7.
  - A pairwise judge (converted to pointwise scores) helps GRPO's safety (6.5 / 6.7 / 16.0) but cuts ArenaHard 51.5 → 34.3.
  - So the stabilisers do not carry over between optimisers.
- [AUTHOR-REPORTED, v3 Table 8 / App. C.2] Adaptive attacks on AdvGame-GRPO (Qwen): PAIR 15.0, TAP 18.1, GCG 32.9, against DPO-MD 7.2 / 10.0 / 25.3.
- [AUTHOR-REPORTED, v3 Table 4; Qwen] AdvGame-DPO-MD is worse than Self-RedTeam on GCG: 25.3 against 23.4. IPO-MD scores 30.0 on Qwen and 35.3 on Llama (Table 8).

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, v3 Sec 4.5, Fig. 3] GRPO was set aside as the main method: "GRPO struggles with pronounced instability and lower overall rewards, particularly for the Defender ... GRPO shows erratic behavior and a decline in performance" on validation curves.
- [AUTHOR-REPORTED, v3 Table 7] **AdvGame-GRPO on Llama collapses into over-refusal.** It is the safest Llama row (HarmBench 1.7, WJB 1.5, DAN 0.3), but WJB-benign compliance falls from 98.8 to **45.6**. The safety numbers are bought by refusing.
- [AUTHOR-REPORTED, v3 Sec 4.5] On on-policy generation: "on-policy training can make the model overfit to recent changes, causing instability and less reliable safety".
- [AUTHOR-REPORTED, v3 Sec 4.5] Point-wise judge: validation rewards looked better, but "this improvement is misleading" because the judge differs between runs; it "makes it easier for models to exploit the reward hacking".
- [AUTHOR-REPORTED, v3 Sec 3.1] A discarded reward design: rewarding the attacker on benign seeds for any non-compliant, non-deflecting defender output "makes the task too simple for the Attacker: If the Attacker can cause the Defender to generate gibberish, it succeeds". It was replaced by category-swapped rewards. App. A says the swap "prevents degenerate gibberish convergence".
- [AUTHOR-REPORTED, v3 Sec 1] Shared-parameter self-play was rejected on argument, not ablation: it "can entangle objectives, leak gradients across roles, and bias exploration ... encouraging degenerate 'solutions' (e.g., Attacker refusing to create harmful attacks)". No shared-weight AdvGame variant was run; Self-RedTeam is the only shared-weight comparison.
- [AUTHOR-REPORTED, v3 Sec 1] The sequential alternation "suffers from inefficiencies and potential instabilities inherent in alternating optimization (Nocedal & Wright, 2006)". This is **asserted only**: no alternating-schedule ablation appears in v1 or v3.
- [AUTHOR-REPORTED, v3 Sec 4.1] Base-sharing failure on Llama: the "safety-aligned Llama3.1-8B refuses to generate adversarial attacks", so the Llama attacker is Llama3.1-8B-Abliterated. The same-base attacker was abandoned for that family.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, v3 App. C.4] For the EMA decay, "a small initial sweep over the decay rate γ" showed no large variation. Whether EMA is used at all is what matters (WJB 8.5 against 49.5).
- [AUTHOR-REPORTED, v3 Fig. 5] Judge size changes outcomes several-fold: WJB 8.5 with the 32B judge against 29.2 with the 7B judge.
- [AUTHOR-REPORTED, v3 Sec 5] On GRPO: "better hyperparameter search and better reward design might improve its performance". GRPO was not tuned to parity.
- [AUTHOR-REPORTED, v3 Sec 4.1] Runs are capped at 1000 steps or 48 h on 16 H200s. No seed variance is reported anywhere in the read text; the tables show single runs.

### 4. Stated limits
- [AUTHOR-REPORTED, v3 Sec 5] The method covers only human-readable attacks; GCG-style suffixes are deliberately out of scope.
- [AUTHOR-REPORTED, v3 Sec 5] GRPO underperformance is not claimed to hold universally.
- [AUTHOR-REPORTED, v3 App. C.4] Llama over-refusal: WJB-benign drops from 98.8 to 69.9 for DPO-MD. The authors attribute it to the weaker Llama attacker's "limited surface". They omit the known mitigation (benign jailbreak-structured prompts) "to isolate the Attacker-Defender dynamics", and the train/validation compliance gap shows a "distribution mismatch".
- [AUTHOR-REPORTED, v3 App. C.3, Table 9] Against released LifelongSA, RR and LAT checkpoints, AdvGame-DPO-MD is far worse on DAN (42.0 against 0.67 / 1.00 / 0.00) and worse on HarmBench-adversarial (7.40 against 1.27 / 7.33 / 7.00). On Llama, DAN also gets worse than Self-RedTeam (42.0 against 32.3, Table 2).
- [AUTHOR-REPORTED, v3 App. C.4] Joint training "costs more than a single-model defense".
- [AUTHOR-REPORTED, v3 Sec 4.4] The defender's robustness to its own attacker (ASR 11.3 Qwen, 9.7 Llama) is "expected", i.e. it is in-distribution and not evidence of general robustness.
- **VERSION DIFFERENCE** [AUTHOR-REPORTED, v1 vs v3] Main-table numbers are identical (8.5, 16.6, 30.8 and 49.5 appear in both). v3 adds App. C.3 (the Table 9 single-turn baselines, including the DAN weakness), App. C.4 (the Llama over-refusal explanation, compute cost and EMA-γ sweep) and minor rewording. The v1 limitations paragraph is unchanged. v2 was not read.

### 5. Independent re-evaluations
- [INDEPENDENT, AdvGRPO 2606.09701 v2 Table 3, numbers COPIED not re-run] AdvGRPO reprints AdvGame's Qwen numbers (HarmBench 4.7, WJB 8.5, DAN 10.3, WGT 1.2, XSTest 81.2). In v2 these beat AdvGRPO on WJB and DAN. This is not an independent re-run.
- [INDEPENDENT, Semantic Scholar citation list, 2026-10-17] Citing works are AgentSafeEvo 2610.19422, ReSI 2610.12233, Reflections and Fragments 2610.05346, Suan 2609.08634, Addressing Over-Refusal with Competing Rewards 2606.31748, AdvGRPO 2606.09701, Min-Max Optimization 2605.13806, MAGIC 2602.01539 and Self-RedTeam (v4) 2506.07468. Grepped texts of MAGIC, ReSI, Suan and 2606.31748 cite AdvGame only in related work. MAGIC's tables compare against Self-RedTeam, not AdvGame. **No independent re-run found.**
- [INDEPENDENT, this note's cross-check] AdvGame trained its own Llama Self-RedTeam checkpoint from public code (v3 Sec 4.1) rather than using an author checkpoint. Self-RedTeam's poor Llama utility in AdvGame Table 1 (AlpacaEval2 13.1, ArenaHard 13.6) therefore comes from a third-party reproduction.

### 6. Implication for an alternating same-base harm-twin loop
AdvGame's evidence says a same-base safety-tuned twin may refuse to play the harm role (Llama), so it needs an uncensored start. RL with a pointwise or small judge gets reward-hacked several-fold. Stabilisers are optimiser-specific: EMA helped DPO (WJB 49.5 → 8.5) but hurt GRPO (WJB 22.3 → 55.6), and GRPO's safest defender came from over-refusal (45.6% benign compliance). Any claim that alternation beats simultaneous updates is untested here.

## TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment (arXiv 2601.18292; read v1 (26 Jan 2026) and v2 (30 Jan 2026); ACL 2026 Long per ACL Anthology preview 2026.acl-long.1216; ID verified on arxiv.org: yes)

VERSION DIFFERENCE: none of substance. A text diff of v1 and v2 shows only author footnotes (equal-contribution marks, affiliation line) and one dash character in the abstract.

Setup reminder: three separate instances of one Qwen3 base (4B / 8B / 14B), the Red, Blue and Eval roles, updated in strict rotation Red -> Blue -> Eval with GRPO/RLVR.
- Step ratio per phase (App B, Table 4): Red 200, Blue 50, Eval 50, so the attacker gets 4x the defender's steps.
- Shared settings: lr 1e-6, batch 256, clip 0.2, KL beta 0.01, temperature 1.0.
- Red reward weights (Table 3): M_Blue 0.6, Llama-3.1-8B 0.2, GLM-4.6 0.2.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Table 2 / Sec 6.2] Red ablation at the 4th iteration. OD = output diversity; ASR measured against Nemotron-Nano-8B and against Llama-3.1-8B-Instruct.

| Setting | OD | ASR Nano | ASR Llama |
|---|---|---|---|
| Full: closed loop + diversity penalty | 0.588 | 85.4 | 6.6 |
| No loop (frozen Blue/Eval), diversity penalty kept | 0.156 | 95.1 | 2.8 |
| Loop, no diversity penalty | 0.514 | 90.9 | 6.0 |
| Neither | 0.004 | 98.2 | 0.2 |

  - Removing the loop costs much more diversity than removing the explicit penalty.
  - Every ablation scores higher ASR against the weak target, so the full method pays about 10-13 ASR points on Nano for its diversity. The authors acknowledge "a trade-off between output diversity and ASR".
  - The Llama-8B ASR of the full method is only 6.6%: the transfer gain over the ablations is from about 0.2-2.8% to 6.6%, small in absolute terms.
- [AUTHOR-REPORTED, Fig 6 / Sec 6.1] Single-defender Red, i.e. dropping the Llama and GLM opponents from the reward, gives lower ASR on all three test models at iteration 4. The numbers are only in a figure (not extractable from text). The authors frame the single-opponent variant as overfitting.
- Not ablated: the Red:Blue:Eval step ratio (200:50:50), rotation order, number of iterations, KL beta, frozen vs evolving Eval (whether Eval training helps Blue is never isolated), shared vs separate instances.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec 6.2, Fig 7] **Red collapse without the loop and the penalty.**
  - The Red model with neither "exhibits a collapse in training entropy. It repeatedly generates fixed adversarial prompt templates, losing the ability to explore new attack strategies."
  - Even with fixes, entropy only "fluctuate[s] within a certain range". There is no claim of sustained growth.
- [AUTHOR-REPORTED, Fig 4 caption / Sec 5.2] **Blue is non-monotone across iterations:** "although some fluctuations occur during training iterations, the ASR ... show a downward trend". Per-iteration values are figure-only.
- [AUTHOR-REPORTED, Sec 5.2 vs Sec 5.1] **Unequal training lengths.**
  - Blue is reported after 10 iterations while Red is reported after 16 iterations (3,200 steps).
  - The paper never says why Blue stops at 10 or what happens to Blue in iterations 11-16. (Inference: possible silent truncation where Blue stopped improving.)
- [AUTHOR-REPORTED, Sec 5.3] **Weak evolving reward model for the smaller runs.**
  - M_Eval 3-class accuracy on the authors' 3,000-item set: 4B 48.2% -> 56.2%; 8B 54.9% -> 64.3%; 14B 97.0% -> 98.2%.
  - The 4B and 8B Blue models were therefore trained on rewards from a judge that misclassifies roughly 36-44% of responses even after training.
  - The authors still state that Eval "yields more accurate and stable reward signals".
- [AUTHOR-REPORTED, Sec 3.3] Reward hacking is anticipated, not measured. Multi-expert majority voting is adopted "to mitigate reward hacking". No hacking audit is reported.
- [AUTHOR-REPORTED, Sec 4] **Circular judging.** GPT-5.2 judges Red's ASR. Blue's safety is scored with benchmark ASR, but the judge used for Blue benchmarks is not stated beyond that sentence. Eval accuracy is measured on an "internal" dataset built from Qwen3-8B outputs labelled through the same distillation template family used for training.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Sec 3.1.2] The diversity-penalty shape y(x)=e^{kx}-1 with k=ln(11/6) was chosen "to align with M_Red reward values". The weights w_B=2 and w_C=4 and the n-grams {3,4,5} come from App B. No sweep.
- [AUTHOR-REPORTED, Limitations] The authors flag uncontrolled growth rates as a risk: "one model may improve too rapidly or too slowly relative to others, potentially destabilizing training or leading to suboptimal equilibria". No mechanism to regulate this is provided.

### 4. Stated limits
- [AUTHOR-REPORTED, Limitations] **The base-sharing and role-safety trade-off is left open.**
  - Quote: "optimizing a model instance for red-team behavior (i.e., M_Red) is likely to degrade its safety-oriented behavior, while strengthening the defensive capabilities of M_Blue may suppress adversarial effectiveness, leading to fundamental trade-offs that are not addressed in this work."
  - Quote: whether "a single shared model instance could be used to iteratively optimize all three roles" is "open and more challenging".
- [AUTHOR-REPORTED, Limitations] No heterogeneous-capability roles; no external SFT data; no Nash or Pareto analysis.
- [AUTHOR-REPORTED, Sec 4 / Table 1] **No over-refusal benchmark.** XSTest, OR-Bench and WJB-benign are absent. Utility is measured only on LiveCodeBench, GPQA, AIME25 and IFEval (IFEval 86.70 -> 85.72 at 14B; AIME25 67.40 -> 65.21 at 4B). A refuse-more defender would not be detected by these metrics.
- [AUTHOR-REPORTED, Ethics] Attack prompts and models are not released in a misuse-enabling way. The code link moved from an anonymous mirror (prior note) to github.com/Qihoo360/TriPlay-RL (abs page, via search).

### 5. Independent re-evaluations
- [INDEPENDENT, ABS arXiv 2605.08427v1, Related Work] Cites TriPlay-RL only descriptively ("introduces an auxiliary 'Evaluator' role to stabilise co-evolution") and notes it gives no "theoretical treatment of the gradient collision and self-consistency bias". No re-run, no numbers.
- No paper found that reproduces TriPlay-RL or uses it as a numeric baseline (searched the web, and checked the downloaded MAGIC, AdvGRPO, GPT-Red and Self-RedTeam texts).

### 6. Implication for an alternating same-base harm-twin loop
The closed loop, not the diversity bonus, is what keeps a same-base attacker from collapsing (OD 0.156 vs 0.588), but it costs attack strength against weak targets. Any learned evolving judge in the loop must be checked against labels every round: a 56-64%-accurate co-trained judge is plausible at small scale and would silently corrupt the defender's reward.

## ABS: The Attacker in the Mirror: Breaking Self-Consistency in Safety via Anchored Bipolicy Self-Play (arXiv 2605.08427, v1 of 8 May 2026 only, read in full; a NeurIPS 2026 oral listing exists per neurips.cc search result, not opened; ID verified on arxiv.org: yes)

Setup facts needed to read the negatives (Sec. 4-5, App. C.3): frozen Qwen2.5 base with two role-specific LoRA adapters (attention and MLP projections, rank = alpha in {16, 32}). The Self-RedTeam code is adapted with Re++, token-level KL ("KL region of 0.3"), **asymmetric learning rates (attacker 1e-6, defender 3e-6)**, WildGuard reward, and "up to 100-200 steps". Each role's gradient is computed with the other held fixed. The paper does not say whether the roles are updated in alternating phases or in the same step.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Table 7 / App. C.4] **Total collapse of one configuration**: Qwen2.5-7B, LoRA r = 32 at step 100 gives WG:Test adv ASR 0.807, WJB 1.000, DAN 1.000, HarmBench 0.984, XSTest benign 0.032. The caption says it "collapsed and the performance reduced to random guess over all the benchmarks". The same rank at step 50 was fine (WJB 0.802).
- [AUTHOR-REPORTED, Table 1 / Table 8] **ABS is not uniformly better than shared-weight Self-RedTeam.**
  - 14B: WG:Test adv 0.131 vs 0.116; WJB 0.706 vs 0.661; HarmBench 0.093 vs 0.056.
  - 14B, WG:Test vanilla *regresses below base*: 0.022 vs 0.017 (-29.4%).
  - 7B: WJB 0.756 vs 0.726; WG vanilla 0.010 vs 0.002.
  - The paper's conclusion concedes only "comparable performance on larger models".
- [AUTHOR-REPORTED, Tables 6-8] Results are non-monotone in steps and rank, and checkpoint choice is implicit. The main-table checkpoints differ per scale (3B r32 step 100; 7B r16 step 100; 14B r32 step 125) and appear to be chosen using the same test benchmarks.
  - 14B step 100 -> 125: WG:Test 0.119 -> 0.131 worse, WJB 0.779 -> 0.706 better, XSTest 0.980 -> 0.968.
  - 7B r16 step 50: WG:Test 0.285, worse than base 0.279.
- [AUTHOR-REPORTED, Table 4] Per-category regressions: 7B ABS WG Toxicity 0.805 -> 0.780 (-3.0%); XSTest discrimination-contrast 0.800 -> 0.760 (7B) and 0.440 -> 0.400 (14B).
- [AUTHOR-REPORTED, Table 3 / Sec. 5.2] ABS attacks are far longer (3B 232.7 vs 59.3 tokens; 7B 250.9 vs 90.0) and "less controlled". The 3B outputs include "residual meta-instructions, such as references to an 'original prompt' or 'revised prompt'". Embedding cosine similarity is "broadly comparable" (UMAP overlap). Diversity gains are lexical, not semantic.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec. 5.1] "in some cases LoRA becomes quickly unstable, and derails to produce nonsense output; a typical collapse mode consists of Qwen producing a sequence of !". The authors link this to a common red-team training prefix.
- [AUTHOR-REPORTED, Sec. 6] "larger models require careful hyperparameter tuning to avoid such failures."
- [AUTHOR-REPORTED, Sec. 3.2] The always-refuse defender is a Nash equilibrium of the same game (Bound 1). The equilibrium guarantee alone does not exclude the degenerate solution.
- [AUTHOR-REPORTED, App. C.4.1, analysis written by GPT-5.4 and quoted verbatim] The attacker-strength gap is partly *not* about self-consistency. At 3B, "the fully fine-tuned attacker often fails at the task itself: instead of emitting a concrete jailbreak, it emits meta-instructions ... part of the 3B gap is not just better stealth; it is basic task completion." This weakens the paper's causal story (gradient collision) for the 3B result.
- [AUTHOR-REPORTED, Table 9] Evaluation anomaly (my inference): the "LoRA Qwen2.5-14B" reasoning row is identical to base Qwen2.5-14B on all five benchmarks (0.91/0.85/0.836/0.906/0.87), and Table 1 gives both 0.874 ± 0.030. Either the adapter changed nothing measurable or the row was copied.
- No reward-hacking audit of the WildGuard reward is reported. Attack success in Sec. 5.2 is judged by GPT-5.4 on 100 prompts, with no seeds or confidence intervals.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Tables 6-8] Rank x step sensitivity, up to total collapse (7B r32 step 100). The authors say "larger models require careful hyperparameter tuning" (Sec. 6).
- [AUTHOR-REPORTED, App. C.3] Separate attacker and defender learning rates (1e-6 vs 3e-6) and warm-up schedules ("different learning-rate warm-up procedures for the attacker and defender", Sec. 4.1). No sweep is reported.

### 4. Stated limits
- [AUTHOR-REPORTED, Sec. 6] LoRA instability and collapse; careful tuning for larger models; gains concentrated at small scale ("particularly at smaller scales"); mosaic and multi-attacker settings left to future work.
- [AUTHOR-REPORTED, Sec. 5] Only Qwen2.5 (3B/7B/14B); one judge (WildGuard) for training.

### 5. Independent re-evaluations
- [INDEPENDENT, search 2026-10-17] No paper found that re-runs ABS or uses it as a baseline.
- [Related, not independent] arXiv 2610.05346 (4 Oct 2026; co-authors include the ABS authors and Self-RedTeam's first author Mickel Liu) adopts role-specific LoRA adapters on frozen LLMs for multi-turn self-play. Abstract only was read; it reports no re-evaluation of ABS.
- [INDEPENDENT, design confound noted by me from ABS Sec. 5] ABS vs Self-RedTeam changes two things at once: separate parameters *and* LoRA instead of full fine-tuning (plus a different step budget and per-role learning rates). There is no single-shared-LoRA control, so the "self-consistency" effect is not isolated from "LoRA regularises". SSP (arXiv 2601.10589, Table 4) finds the opposite sign for sharing (separate models worse: Qwen GCG 5.8 vs 1.7).

### 6. Implication for an alternating same-base harm-twin loop
A frozen shared base with disjoint adapters is the cheapest way to stop role leakage. But LoRA RL can collapse abruptly (one rank/step setting went to 100% WJB ASR), and the benefit over sharing shrinks with scale. The harm twin needs per-checkpoint collapse monitoring and a pre-registered checkpoint rule, not test-set selection.

## Active Attacks: Red-teaming LLMs via Adaptive Environments (arXiv 2509.21947; read v1 (26 Sep 2025) and v2 (4 Oct 2025); ICML 2026 poster per icml.cc; ID verified on arxiv.org: yes)

VERSION DIFFERENCE: none of substance. A text diff shows only the arXiv stamp and the acknowledgements (Samsung added to funders).

Setup reminder: attacker Qwen2.5-1.5B (base, SFT warm-up), trained by GFlowNet trajectory balance with a replay buffer; victim Qwen2.5-1.5B-Instruct (same family, different checkpoint); reward and evaluation both from Llama-Guard-2-8B.
- Schedule: T=5,000 attacker steps, with victim safety-SFT (LoRA, 200 steps) every R=1,000 steps.
- At each victim update the attacker is re-initialised to p_ref and its buffer cleared (Algorithm 1).
- The final victim is fine-tuned on the global buffer, not taken from the sequential chain.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Table 2 / Table 8, 3 seeds] Fixed-environment baselines, i.e. the "passive" no-alternation variants of the same attacker objectives: average defence rate of a victim safety-tuned on each method's prompts.
  - REINFORCE 81.48 +- 3.11
  - PPO+Novelty 82.86 +- 2.86
  - GFlowNet 91.00 +- 2.81
  - GFlowNet + Active Attacks 99.71 +- 0.11
  - Plain GFlowNet's per-victim variance is large: gemma 89.69 +- 14.49, Qwen 89.27 +- 11.53.
- [AUTHOR-REPORTED, Fig 5 / Sec 4.3] Fewer rounds give less diversity: "the diversity of generated prompts consistently increases as the rounds go by ... after five rounds ... the diversity converges". The round ablation is figure-only.
- **No component ablation.** Victim updates and attacker re-initialisation are always applied together. There is no "victim update without attacker reset" variant and no "reset without victim update" variant, so which component helps is not identified (Sec 3.2, Algorithm 1).

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec 3.1] **The replay-buffered GFlowNet baseline still collapses.**
  - Quote: it "still fail[s] to achieve effective semantic diversity", prioritising "easily discovered attack prompts (i.e., high-reward modes)", leading to "residual mode collapse".
  - Table 1 / Table 10 show REINFORCE prompts all in one Llama-Guard category.
- [AUTHOR-REPORTED, Sec 3.1] On-policy PPO is described as unable to use replay, "which is crucial for preventing forgetting". This is the stated reason for choosing an off-policy attacker objective.
- [AUTHOR-REPORTED, Sec 3.2] **The sequential victim chain was deliberately not used as the final defender.** The design "avoids relying solely on sequentially fine-tuned victim models, thereby mitigating potential sequential bias in the learned attack prompts". In effect, the authors did not trust the end of the alternation chain and retrained from the aggregated buffer instead.
- [AUTHOR-REPORTED, Sec 3.2 / Algorithm 1] Continuing the same attacker across victim updates was rejected. The attacker is reset because otherwise it keeps "repeatedly targeting the same modes" ("Attacker reset: avoid overfitting to old modes"). The paper shows no continuation-run numbers.
- [AUTHOR-REPORTED, App C] Cosine distance was abandoned as the diversity metric: "even with a relatively high cosine distance, the generated prompts can be collapsed into a certain category". A Llama-Guard categorical distance replaces it.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App D.1 vs Sec 4.3] **The round count is inconsistent within the paper.** D.1 says "4 rounds of safety fine-tuning" (T=5000, R=1000); Sec 4.3 says "five rounds (which is the default setting)". The interval R=1000 is not swept.
- [AUTHOR-REPORTED, Table 17] Single settings only: GFlowNet lr 1e-4, beta 0.1, batch 128; safety SFT lr 3e-5, 200 steps, batch 1024.
- [AUTHOR-REPORTED, Table 4 / Table 5] Changing the attacker or the classifier changes the operating point:
  - Llama3.2-3B attacker: toxicity 96.84 but cosine distance 0.65, vs 0.78 for Qwen2.5-1.5B.
  - HarmAug-Guard reward: GFlowNet 82.03 +- 5.27 vs Active Attacks 87.35 +- 0.15 toxicity.

### 4. Stated limits
- [AUTHOR-REPORTED, whole paper] There is no Limitations section.
- [AUTHOR-REPORTED, Sec 4] **The same classifier is reward and evaluator** (Llama-Guard-2-8B, threshold 0.5). Every "defense rate" in Table 2 is judged by the classifier the attacker was optimised against.
- [AUTHOR-REPORTED, Sec 4] **The defense-rate pool includes the method's own attacks.** It is measured against prompts from all compared methods, so part of the 99.71 is defence against its own in-distribution attacks.
- [AUTHOR-REPORTED, Sec 4] Refusal targets are generated by prompting the victim with "Refuse the following instruction." No over-refusal benchmark is reported. Utility is checked only on 6 Open LLM Leaderboard tasks (Table 6), which cannot detect over-refusal.
- [AUTHOR-REPORTED, App D] Scale is tiny: attacker and victim are 1.5B by default, "as larger models do not fit within the NVIDIA L40S GPU". Larger victims see only transferred prompts, never the live loop.

### 5. Independent re-evaluations
- [INDEPENDENT, Stable-GFlowNet arXiv 2605.00553v3, Sec 5.1 / Table 1] **Adopts the setup but drops Active Attacks as a baseline.**
  - Kwon et al. take Active Attacks' setup ("Following Yun et al. (2025)": Qwen2.5-1.5B attacker and victim, same diversity encoder, same LoRA safety-SFT) but **do not include Active Attacks in their baseline table**. The baselines are ICL, SFT, DPO, PPO, PPO+Curiosity, GFN, Jailbreak-R1 and Rainbow Teaming.
  - They report that the underlying GFN-TB objective is "notorious for training instability and mode collapse", that "unstable rewards in red-teaming accelerate mode collapse", and that noisy classifier rewards on gibberish "collapse exploration".
  - Plain GFN in their run: 93.75% ASR but only 17.67 unique attack clusters, and it falls to 4.69% ASR against an S-GFN-tuned victim.
  - This is indirect evidence against the base learner Active Attacks builds on, not a re-run of Active Attacks.
- No direct re-evaluation of Active Attacks found.

### 6. Implication for an alternating same-base harm-twin loop
Alternation helped here only together with an attacker reset and a final retrain on the aggregated buffer. The authors themselves avoided the end of the sequential chain ("sequential bias"), so a harm-twin loop should plan for per-round resets or replay and a final consolidation, not trust the last-round defender.

## MART: Improving LLM Safety with Multi-round Automatic Red-Teaming (arXiv 2311.07689, v1 only (13 Nov 2023, the only version on arXiv; NAACL 2024); ID verified on arxiv.org: yes)

Setting reminder: the attacker (M_adv) and the target (M_tgt) start from the same LLaMA-65B LIMA/OASST SFT checkpoint. Both are trained by SFT, not RL, in strict alternation for 4 iterations. The prior notes already cover the headline numbers, the round-4 stop and the RM-overfitting worry. What follows adds the ablation-level detail.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Sec 3.3, Fig 3] **Mart-3shot vs Mart-1shot.** Seeding the attacker with 3 demonstration attacks instead of 1 "does not achieve as satisfying performance". It is "less diversified", and the 1-shot setting gives "larger numbers of adversarial generations". The exact values are only plotted; the text has no numbers.
- [AUTHOR-REPORTED, Sec 3.3, Fig 3] **Few-shot prompting attacker (frozen weights, Perez-style).** It "showcases the lowest attack violation rate and minimal model safety". A frozen attacker gives the defender the least signal, and the authors call this evidence for "response-oriented optimization of the adversarial model". Values are only plotted.
- [AUTHOR-REPORTED, Sec 3.3, Fig 3] **GCG-multiple / GCG-transfer as the attacker in the loop.** These "trigger the most harmful responses in iteration 1, however their efficacy decay after the first round of target model safety fine-tuning". They also bring "mediocre safety improvement on SafeEval". The reason given is that suffix attacks are "easy for a model to identify and bypass", so a static attack family stops paying out after one defender update.
- [AUTHOR-REPORTED, Sec 3.4, Fig 4] **Safety-data selection thresholds** (safety RM swept over [0.4, 0.9], helpfulness RM over [0.0, 0.6]). Performance "does not change monotonically with the number of training data", and "increasing the selected training data does not always lead to a better model". The optimum is safety 0.8 / helpfulness 0.4.
- [AUTHOR-REPORTED, Table 4] **The attacker's own hit rate is non-monotone across rounds.** Adversarial-generation violation goes 29.74% (iter 1) -> 31.89% (iter 2) -> 20.33% (iter 3) -> 10.21% (iter 4). The attacker briefly gains in round 2, then the defender wins.
- [AUTHOR-REPORTED, Sec 3.2, Fig 2b] **Helpfulness erodes with rounds.** "iter4 is 3%-4% lower than iter1 on helpfulness" on HelpEval, even though no extra helpfulness data is added.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec 2.3] **Context distillation was dropped after iteration 1.** The authors say "We did not use context distillation in the following iterations as we no longer observed essential performance improvement". They add that "using a safety preprompt may result in response degradation as the model tends to become over-conservative", i.e. it pushed the target toward over-refusal once the target improved.
- [AUTHOR-REPORTED, Sec 2.3] **Directly relaxing the RM selection thresholds was rejected.** When the selected set R_tgt was "relatively small", relaxing theta_s / theta_h "may lower data quality", so they used distillation and rejection sampling instead.
- [AUTHOR-REPORTED, Sec 2.3] **Convergence / data starvation at the last round.** "At the last iteration T-1, we find the model tends to converge on the generated adversarial prompts. There is no significant change of safety performance". Rejection sampling (K samples per prompt at varied temperatures) was added only to obtain enough training data.
- [AUTHOR-REPORTED, Sec 3.3] **Loop terminated by attacker starvation, not by convergence of quality.** "When the violation rate on adversarial generations falls near 10% ... it becomes hard to [get] sufficient enough training data for the next iteration. As a result, we stop at iteration 4". They "hypothesize that external red-teaming efforts are needed for additional safety improvement".
- [AUTHOR-REPORTED, Sec 3.2] **Reward-model gaming risk:** the safety RM "may provide the model an incentive to manipulate the measure ... resulting in overfitting on the RM preference". It is checked only by human evaluation on two sets. RM and human violation rates differ in absolute level (SafeEval RM 4.8% vs human 8.0%).

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, Sec 3.4] The authors claim the method is "not sensitive to the threshold change" across the swept ranges. The non-monotonic data-quantity/quality trade-off above is the only sensitivity they flag. Number of shots (1 vs 3) matters (Sec 3.3). No learning-rate or round-count sweeps are reported.

### 4. Stated limits
- [AUTHOR-REPORTED, Sec 3.2, Table 3] The final model is still well behind heavily human-red-teamed models. MART: 4.8% RM / 8.0% human on SafeEval. Llama-2-Chat-70B: 2.1% / 4.2%. ChatGPT: 2.9% / 5.7%.
- [AUTHOR-REPORTED, Sec 3.3] Out-of-domain gains are "similar but less significant" than in-domain ones (Anthropic Harmless 26.73 -> 6.92% vs SafeEval 31.38 -> 4.79%).
- [AUTHOR-REPORTED, Sec 5] RL and multi-turn dialogue are left to future work. The only base is LLaMA-65B. There is no explicit limitations section; the ethics statement says the method "does not remove the need for ... human oversight".

### 5. Independent re-evaluations
- [INDEPENDENT, DART v1, arXiv 2407.03876v1, Sec 4.2, Table 1, Fig 2] DART "reimplemented [MART] as our baseline" on Llama-2-7B.
  - Anthropic Harmless violation, RM / human: Vanilla 18.9% / 34.3%, MART-reimpl 14.4% / 20%, DART 11.6% / 16%.
  - In this reimplementation MART cuts human-judged violation risk by 41.7%. MART's self-reported relative reduction on Anthropic Harmless human evaluation was 12.1 -> 4.9%, about 59.5%. Different base and scale, so this is not a like-for-like replication.
  - DART's Fig 3 ablations show plain MART-style threshold selection ("Baseline") below the variants with diversity filtering or active learning on in-domain safety over 7 iterations. These values are only plotted.
  - Note: the same arXiv record was later retitled APRT (see prior notes).
- [INDEPENDENT, CHASE, arXiv 2606.05523v1, Sec 2] Criticises MART conceptually: it hardens "through imitation or supervised learning on fixed targets". There is no rerun.
- No other rerun found. Searches of AdvGame, MAGIC, AdvGRPO, Self-RedTeam, ABS and APL show MART cited but never used as an evaluated baseline.

### 6. Implication for an alternating same-base harm-twin loop
A same-base alternating loop that relies on threshold-selected attacker outputs starves after about 3-4 rounds, once the defender's failure rate approaches about 10%. A harm twin therefore needs a source of signal that does not depend on the defender currently failing (e.g. preference pairs, reward shaping, or fresh seeds), or the alternation stops by default rather than by convergence.

## RTG / GRTS: Evolving Diverse Red-team Language Models in Multi-round Multi-agent Games (v1 title "Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models") (arXiv 2310.00322, v1 30 Sep 2023 and v5 28 Jul 2024 read; v2-v4 not read; ID verified on arxiv.org: yes)

Setting reminder: the red and blue teams are the same stablelm-3B family. Each is trained with PPO as a best response to the other side's population meta-strategy (PSRO / double oracle). Every red best response is **re-initialised from SFT-Red**, not continued from the last red policy (App D.2, Algorithm 1 lines 4-14). The prior notes cover the exploitability curve (6.2 -> 0.8 over 15 iterations), the "slow and moderate adversarial oscillation" and the more severe mode collapse with an over-fitted RM. The items below are additional.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, v5 Sec 5.2, Table 3 "Baseline hybrid evaluation"] **Single-agent best responses against a fixed opponent lose to population-trained ones in cross-play** (avg toxicity; positive means red wins):

  | Red | Blue | Avg toxicity |
  |---|---|---|
  | multi-agent | single-agent | +7.43 |
  | single-agent | fixed blue | +10.02 |
  | single-agent | multi-agent | -6.90 |
  | multi-agent | multi-agent | -9.65 |

  The authors read this as "some models can perform well against fixed opponents during training, but are immediately defeated by others" (App B.3.2).
- [AUTHOR-REPORTED, v5 Sec 5.2, Fig 8] **Single-agent baselines show "a steep decline in diversity"**, "resulting in predictable and less effective strategies". The values are only plotted.
- [AUTHOR-REPORTED, v5 Sec 5.2] **The blue-team advantage from population training is weak.** "We posit that the stronger performance of the red team, but not as evident in the blue team. This is due to the red team selected for testing is relatively weak. After all, defense in red teaming is significantly easier than breaching defenses."
- [AUTHOR-REPORTED, v5 Sec 5.3, Fig 10, Table 10] **Dialogue-round ablation (1-5 rounds at constant total rounds: 60K/30K/20K/15K/12K rollouts).** 1-2 rounds are Pareto-inferior on helpfulness vs harmlessness, and 3/4/5 rounds are "Pareto superior". Single-round RLHF carries a larger alignment tax. The values are only plotted. 3 rounds was chosen for compute.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, v5 App B.2.3] **The blue team reward-hacked the toxicity RM by refusing, and a heuristic patch failed.** "The blue team consistently received higher rewards from a fixed pattern of behavior, which was to begin responses with 'As an AI Language Model' and then refuse to answer". The authors tried filtering refusals with a stop-word list during online scoring, but "our semantic layer of stop words did not adequately cover all nuances of 'refusal to answer', with numerous ways found to bypass this filter." This is the defender analogue of judge gaming, and the rule-based fix was abandoned.
- [AUTHOR-REPORTED, v5 App B.3.1] **Distribution shift degrades the frozen RM inside the loop.** "The reward model fails to continuously train on the preferences of the shifted distribution online, leading to increasingly inaccurate reward modeling. This cycle repeats." The authors say this is "further magnified" in red teaming because exploration in a vast action space leaves the RM's support, which is "a disaster for rewards".
- [AUTHOR-REPORTED, v5 App B.3, B.4] **Exploration and exploitation were hard to balance:** "overwhelmingly tricky to balance exploration and exploitation in Red-Teaming LLMs". On offence/defence asymmetry, "defense often being easier ... requires careful calibration of the training regime to ensure both teams progress at a comparable rate". They reject the obvious fix of reducing the step size or update frequency because it "would lower data sample efficiency".
- [AUTHOR-REPORTED, v5 App B.4] **Blue-team diversity could not be induced:** "Encouraging diversity in the blue team's responses is challenging due to the strong policy of refusal to answer".
- [AUTHOR-REPORTED, v5 Sec 5.2] **Collapse propagates to both sides:** "mode collapse on one side can potentially leads to both side converging to outputting repetitive and fixed patterns, especially when their outputs are out of distribution of the reward modeling which means the reward signals from the toxicity model are invalid."
- [AUTHOR-REPORTED, v1 Sec 4.1] **VERSION DIFFERENCE.** v1 states that during GRTS solving "the semantic diversity of the Red Team decreased to some extent", attributed to "the constraints imposed by the context provided in each iteration". v1 also ran a setting where "we simultaneously optimized both the red and blue sides" alongside the fix-one-side setting, but reports no comparison between them. v5 drops the diversity-decrease sentence and frames diversity as maintained relative to single-agent baselines. The over-fitted-RM, refusal-hacking and exploration appendices (B.2-B.4) and the single-agent-vs-multi-agent Table 3 are absent from v1 (grep of v1 PDF text: no "refus", "over-fit" or "single-agent" hits).
- [AUTHOR-REPORTED, v1 Sec 4.1] **The blue team converges slowly against the static human-written first turn.** It needed about 16k prompts on turn 1 vs about 6k on turns 2-3. "The overall performance gap between the blue team and the red team in the three turns of attack remained marginal."

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, v5 App B.2.2] **Reward-model scale and quality changes the dynamics qualitatively.** A toxicity model trained on 30K examples (vs 300K) over-fits, emits rewards on "a significantly larger scale", and makes training "much more unstable" with "significantly more severe mode collapse". Not normalising toxicity scores was a deliberate choice ("did not enhance the accuracy", Sec 4).
- [AUTHOR-REPORTED, v5 App D] KL 0.04 and PTX 8.0 (from prior notes); no sweeps reported. The diversity weight tau_t is "a tunable constant" with no sweep.

### 4. Stated limits
- [AUTHOR-REPORTED, v5 Sec 5.4] **The equilibrium is a defensive concept, not the strongest attacker.** "The approximate Nash equilibrium derived from GRTS is, in fact, a defensive solution concept ... when targeting specific LLMs, exploitative strategies are often sufficient". To attack external models the authors cherry-pick two policies from the population rather than use the meta-strategy.
- [AUTHOR-REPORTED, v5 Sec 6] The conclusion names the remaining open problem as "how to increase attack diversity to mitigate mode collapse"; better diversity metrics are future work. Only 3B models are trained. Blue-team helpfulness is only plotted (alignment tax). There is no human evaluation.

### 5. Independent re-evaluations
- None found. Searches turned up no rerun of GRTS. GPT-Red (arXiv 2607.26115, Sec 2) names Ma et al. 2023 as "most similar" and says it "dramatically scale[s] these approaches"; this is not a re-evaluation.

### 6. Implication for an alternating same-base harm-twin loop
A frozen judge drifts off-support as both same-base policies move. The defender will learn a refusal template that games it, and the twin will collapse when the RM is over-confident. Budget for judge refresh or a calibrated scale, and expect lexical refusal filters to fail.

## ARLAS: Adversarial Reinforcement Learning for Large Language Model Agent Safety (arXiv 2510.05442, v1 only (6 Oct 2025, the only version); ID verified on arxiv.org: yes)

Setting reminder: the attacker writes indirect prompt injections into tool outputs and the agent defends. Both are GRPO-trained LoRA models (rank 128) with imitation warm-up from larger teachers. The agent trains against a uniform sample of all past attacker checkpoints; the attacker plays only the latest agent. The prior notes cover the AgentDojo headline and the "cyclic learning" motivation.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Sec 4.2, Table 1 (AgentDojo, Gemma3-12B, 3 seeds)] ASR / TSR for each variant:

  | Variant | ASR (%) | TSR (%) |
  |---|---|---|
  | Base | 6.3 | 24.4 |
  | ARLAS | 5.4 ± 0.1 | 25.8 ± 0.1 |
  | w/o PBL (latest-vs-latest iteration) | 6.6 ± 0.1 | 23.5 ± 1.5 |
  | ART (frozen attacker and critic, agent RL only) | 6.5 ± 0.1 | 25.7 ± 1.0 |
  | w/o AL (task-only RL) | 8.0 ± 0.0 | 21.4 ± 1.3 |

  - Without population-based learning the loop is worse than the base model on ASR and TSR.
  - Task-only RL raises ASR by 1.7 points, so capability RL without the adversary degrades safety.
- [AUTHOR-REPORTED, Sec 4.2, Fig 4 (BrowserGym cross-play heat maps)] "ART and ARLAS w/o PBL perform similarly, reducing the attack success rate but not as effectively as ARLAS"; "ARLAS w/o AL performs as badly as the base model". The values are only in heat maps.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, Sec 3.3, Fig 2] **Latest-vs-latest pairing is asserted, not shown, to cycle.** "This procedure can lead to unstable, cyclic learning dynamics where policies fail to converge ... the agent may learn to defend against an attacker's current strategy (pattern A) while forgetting how to counter a previous one (pattern B)." No training curve demonstrating the cycling is shown. The evidence is only the final w/o PBL numbers above.
- [AUTHOR-REPORTED, Table 1] **SPAG, an iterative (non-population) offline-RL baseline, ties ARLAS on AgentDojo.** SPAG: ASR 5.4 ± 0.0, TSR 26.2 ± 1.0. ARLAS: 5.4 / 25.8. This undercuts the claim that population training is "critical" out of distribution. The authors only say ARLAS is "on par with SPAG".
- [AUTHOR-REPORTED, App B.2, Fig 6] **Non-monotone defender on Qwen3-14B.** "When facing a fixed attacker, the agent's attack success rate does not decrease monotonically during training", which the authors attribute to a low starting ASR.
- [AUTHOR-REPORTED, App B.1] **Unsolvable environments were filtered out.** Some BrowserGym tasks "were unsolvable"; training was restricted to tasks where Gemma-3-27B had >20% success. Sparse rewards gave no signal otherwise.
- [AUTHOR-REPORTED, Sec 3.2] **A cold start was impossible without imitation warm-up.** SFT from 10K teacher episodes was needed so that "both models achieve non-zero success rates".

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, App B Table 3] Fixed values: lr 2e-6, KL beta 0.05 to the SFT model, asymmetric clip 0.1/0.3, temperature 1.2, 128 tasks x 16 episodes per iteration. No sweeps are reported, and the number of iterations (5/10/15) is not ablated.

### 4. Stated limits
- [AUTHOR-REPORTED, Sec 5] "Due to computational constraints, we only verified the effectiveness of ARLAS on two open-source LLMs."
- [AUTHOR-REPORTED, Sec 4.2] On Qwen3-14B "the improvement is relatively small" (ASR 1.6 -> 1.4) because the base is already safe.
- [AUTHOR-REPORTED, Table 1] The absolute out-of-distribution effect is tiny: AgentDojo ASR moves 0.9 points on Gemma and 0.2 on Qwen. Whether these exceed seed noise is not tested beyond standard errors. Base-model rows have no SE.

### 5. Independent re-evaluations
- None found. A search turned up no paper rerunning ARLAS. Related 2026 attack papers on AgentDojo (AutoInject, arXiv 2602.05746; PISmith, arXiv 2603.13026) do not evaluate ARLAS-trained agents (search summary only; not opened).

### 6. Implication for an alternating same-base harm-twin loop
Always training the main model against only the latest twin is the variant that failed here: no gain over base. Keeping a checkpoint pool of past twins for the defender side is the cheapest guard. But an iterative non-population baseline matched it out of distribution, so the pool alone is not proven sufficient.

## GPT-Red: Automated Red Teaming via Self-Play at Scale (arXiv 2607.26115, v1 only (28 Jul 2026, the only version; "28 pages"); ID verified on arxiv.org: yes)

Setting reminder: the agentic attacker is initialised from GPT-5.5 and trained with RL against a population of simultaneously trained defenders. Rewards are "typically asymmetric" with validity checks. It was then used to generate RL training data for GPT-5.6. The prior notes cover the ASR dips, mode collapse in attacker-only runs, poor transfer across defenders and the 5.2% -> 95.9% fake-CoT result. Every dynamics claim in this paper comes from figures without numeric tables, and there is no numeric ablation table at all.

### 1. Ablated variants that hurt
- [AUTHOR-REPORTED, Sec 5.1, "not shown"] **Attacker-only training against a fixed defender was "better narrowly, i.e., mainly against the defender they trained against and worse on held-out ones"** and "more likely to be mode collapse in their attack strategies". No numbers are given; the result is explicitly "not shown".
- [AUTHOR-REPORTED, Sec 5.1, "preliminary experiments"] **Self-play with a single defender "often resulted in attackers would mode collapse onto a narrow class of attack strategies".** The fix was "many defender models, especially those that are highly robust to different types of attacks". No numbers are given.
- [AUTHOR-REPORTED, Sec 5.1, Fig 5] **Frozen prompted attackers, with or without the defender-query tool, gave slower-improving and lower final held-out defender robustness** than self-play attackers. Only plotted (mean direct-prompt-injection robustness).
- [AUTHOR-REPORTED, Sec 7.2, Fig 10] **GPT-Red shows no advantage over frozen GPT-5.5 on easy tasks.** "We observe similar performance for easier tasks"; GPT-5.5 "stalls on harder tasks". 3 runs each, 10 tasks, only plotted.

### 2. Tried and abandoned; unstable, oscillating, collapsing, forgetting, reward-hacked
- [AUTHOR-REPORTED, App C, Fig 25] **The attacker exploits ambiguity in the reward and grader.** One evaluation category is "Edge Case Attack Prompts", collected from "a trained attacker run that specialized in short attacks containing unusual formatting, multilingual text, or special characters that exploit ambiguity in the problem description or reward function". The Fig 25 caption reads: "These ambiguous edge case prompts are common tricks that our attacker models exploit". Example: the system prompt forbids Unicode pictograms, the attacker asks for a reply in Chinese, and the verifiable grader fails. This is reward hacking against programmatic verifiers. The authors fold it into evaluation rather than fixing it.
- [AUTHOR-REPORTED, Sec 3.1, 3.2, 6.2] **Invalid-attack hacking needed explicit rejection.** "There are many ways for agents to 'cheat' or 'reward hack' by editing parts of the rollout that a realistic adversary should not have access to". Invalid attacks get reward 0 via "a mix of programmatic and LLM-based checks". The efficacy of these checks is only "qualitatively assess[ed]".
  - For jailbreak environments two anti-hacking filters were added: reject attacks that already contain disallowed content and merely ask for a transformation such as translation, and reject attacks that "do not resemble plausible user conversations".
- [AUTHOR-REPORTED, Sec 5.1, Fig 4] **ASR oscillation is framed as healthy.** "The attacker discovers strategies that are patched by the defender before moving on to new approaches ... These events correspond directly with dips in the attacker ASR". The paper gives no quantitative check that patched attacks stay patched later in the run, i.e. no forgetting test on the defender side.
- [AUTHOR-REPORTED, Sec 5.1, Fig 6] **Attacker monotonicity is only "nearly monotonic"** ("later attacker checkpoints are nearly monotonically better at breaking all defender checkpoints"). No figure shows that defender checkpoints are monotone against all attackers.

### 3. Hyperparameter sensitivities
- [AUTHOR-REPORTED, whole paper] None reported. The RL algorithm, KL, learning rate, step counts, number of defenders in the main run, update ratio and judge models are not stated. The paper says robustness improves when "we scale three ingredients together" (attacker inference search, self-play compute, environments), with no per-ingredient ablation numbers.

### 4. Stated limits
- [AUTHOR-REPORTED, Sec 1] "While robustness remains far from solved".
- [AUTHOR-REPORTED, Sec 9] GPT-Red "has seen less training on multi-modal environments, multi-turn attack scenarios, and content-policy jailbreaks".
- [AUTHOR-REPORTED, Sec 7.2, footnote 2] The human comparison "does not imply that GPT-Red is universally better than human red-teamers. Humans may, for example, find novel scenarios or classes of attack not considered by GPT-Red."
- [AUTHOR-REPORTED, Sec 8.2] Held-out red-teaming "inflates the attack success rates compared to real attackers since it does not use system-level mitigations". Also: "All evaluations include any of: label noise, inherent ambiguity, or reward noise, which sometimes prevents any model from reaching 100% score."
- [AUTHOR-REPORTED, Sec 8.2] **Evaluation circularity.** The "Strong, IID" evaluation splits GPT-Red's own attacks: the defender is trained on the weakest half and evaluated on the strongest half. Five of six direct-injection benchmarks in Fig 13 are "categories of new attacks generated by GPT-Red". Only TensorTrust is external.
- [AUTHOR-REPORTED, Sec 8.2] **Over-refusal is listed as measured but no over-refusal result appears** anywhere in the main text or Appendix C (grep for "refus"/"overrefus" finds only the metric announcement). The same goes for capability retention.
- [AUTHOR-REPORTED, Sec 7.1] The defender-side base is not disclosed: "a suite of past defender models ... up to and including GPT-5.5", while the attacker is also initialised from GPT-5.5. Same-base pairing is therefore present, but its effect is not analysed.

### 5. Independent re-evaluations
- None found. The paper is from 28 Jul 2026, and there are no open weights to rerun.

### 6. Implication for an alternating same-base harm-twin loop
At frontier scale the decisive anti-collapse lever was opponent multiplicity (many defenders), not the schedule. A single same-base twin against a single main model is the configuration GPT-Red found to collapse. Programmatic or rubric judges will be gamed through specification ambiguity unless validity filters are built in from the start.

---

# Cross-paper lessons: alternation schedule, base sharing, RL dynamics

Each lesson lists the papers behind it. Where the papers disagree, the lesson says so.

**L1. Nobody has properly tested the alternation schedule itself. Treat schedule claims as untested.**
- Only one paper ablates alternation against simultaneous updates: AdvGRPO v2, Table 5, three seeds.
- Swapping every step (N=1) instead of every 10 (N=10) gives:
  - HarmBench 6.0±1.8 vs 4.9±1.2
  - WJB 22.9±6.6 vs 21.8±5.4
  - DAN 28.2±5.9 vs 22.9±5.8
- Only the DAN gap exceeds about one standard deviation.
- Other papers fix a schedule without a sweep:
  - MAGIC: 15 steps, 1:1
  - TriPlay-RL: 200:50:50 steps, Red:Blue:Eval
  - Active Attacks: R=1000
  - ARLAS: per-iteration
  - CHASE: full phases
- Other papers' claims are argued, not tested:
  - AdvGame says alternating optimisation "suffers from inefficiencies and potential instabilities" but has no ablation; it uses simultaneous updates with EMA opponents.
  - RTG v1 mentions a simultaneous-update run but reports no comparison.
- **Disagreement (in design only):** AdvGame and Self-RedTeam use simultaneous updates; MAGIC, TriPlay-RL, AdvGRPO, CHASE, Active Attacks and ARLAS alternate. No head-to-head result settles which is better.
- For the harm-twin report, the schedule is a free parameter to register and sweep, not a settled choice.

**L2. Training the main model against only the latest twin is the one configuration with direct negative evidence.**
- ARLAS w/o PBL (latest vs latest): ASR 6.6 vs base 6.3, TSR 23.5 vs 24.4 (Table 1).
- GPT-Red: one defender, or a fixed one, gave attacker mode collapse; many defenders was the fix (Sec 5.1, numbers not shown).
- RTG Table 3: single-agent best responses lose in cross-play (-6.90 vs a population-trained blue).
- TriPlay-RL Fig 6: a Red with a single-defender reward does worse.
- Active Attacks: the authors did not use the end of the sequential victim chain; they retrained the final victim on the global buffer to avoid "sequential bias".
- **Disagreement / caveat:**
  - SPAG, an iterative baseline without a population, ties ARLAS on AgentDojo (5.4 vs 5.4).
  - Active Attacks never separates the victim update from the attacker reset.
  - So "keep a pool of past twins" is supported against latest-only, but not shown to be sufficient.

**L3. A co-trained attacker specialises to its current partner and its training value decays over rounds.**
- MAGIC Table 10: the co-trained attacker is *weaker than the untrained base attacker* against MAGIC's own defender (9.00% vs 10.50%), while stronger against outside defenders.
- CHASE App G:
  - convincingness 4.79 -> 2.83 -> 2.38
  - attacker SR 0.47 -> 0.24 -> 0.22
  - raw ASR rises 0.51 -> 0.76
  - The attacker learns low-specificity bypasses.
- MART: the attacker ran out of usable attacks at about 10% violation and stopped at round 4. Its hit rate was non-monotone: 29.74 -> 31.89 -> 20.33 -> 10.21.
- AdvGRPO v2 App B picks the defender checkpoint "at the onset of attack-reward convergence", because longer training "only risks the defender over-fitting to the attacker's narrow strategy distribution".
- **Disagreement on the remedy:**
  - Reset each round: Active Attacks resets the attacker to p_ref; RTG re-initialises every red best response from SFT-Red.
  - Continue: MAGIC, CHASE and AdvGRPO carry the attacker forward.
  - No paper compares reset with continuation.
- For a harm twin, benchmark each round's twin against a frozen base twin and an external main model, not only against its partner.

**L4. Same-base sharing breaks in two directions; whether to share weights is contested.**
- Safety leaks into the attacker role:
  - Self-RedTeam: the shared-weight attacker refuses 25.27% of attacks on Llama-3.1-8B-IT vs <0.11% on Qwen (v4 App E.2).
  - AdvGame and MAGIC: a safety-tuned Llama refused to attack, so both swapped in an abliterated Llama.
  - AdvGRPO: reasoning attackers "self-censor", with the thinking reward near zero until about step 10.
- Training the attacker can erode its own safety: TriPlay-RL Limitations says red training "is likely to degrade its safety-oriented behavior", left "not addressed".
- **Disagreement on weight sharing:**
  - Against sharing: ABS (separate LoRAs beat Self-RedTeam at 3B); MAGIC and AdvGame argue it.
  - For sharing: SSP, arXiv 2601.10589, finds separate models worse (GCG ASR 5.8 vs 1.7).
  - ABS is not uniformly better at 14B: WG:Test 0.131 vs 0.116, HarmBench 0.093 vs 0.056. It also changes sharing and LoRA at the same time.
  - MAGIC's "gradient conflict" claim is never ablated.
- For a harm twin that writes harmful responses, the leak direction matters most. A twin started from the safety-tuned main model will under-produce harm, so it needs its own adapter or weights and probably an uncensored or warm-started start. That start then stops being a faithful "twin".

**L5. The defender wins easily. The attacker needs a warm start and usually more steps.**
- AdvGRPO: starting both from scratch "causes the defender to dominate". Removing the curriculum raises WJB from 21.8±5.4 to 32.1±16.9 (v2 Table 5).
- MAGIC: the base attacker has a "high refusal rate", so it uses an SFT warm-up from Gemini-2.5-Pro.
- ARLAS: imitation warm-up was needed for "non-zero success rates".
- RTG: "defense in red teaming is significantly easier than breaching defenses". The authors rejected slowing the defender because of sample efficiency.
- The balancing knobs differ: TriPlay-RL gives Red 4x the steps (200 vs 50). ABS gives the *defender* the higher learning rate (3e-6 vs 1e-6). Neither is swept.
- TriPlay-RL flags relative growth rates as a destabiliser with no fix.
- For a harm twin that must keep the main model's breach rate above the judge's noise floor, this is the central schedule risk.

**L6. GRPO stalls on a confident aligned model, and the stabilisers depend on the optimiser.**
- CHASE: an aligned defender's groups are all-refuse or all-comply, giving "near-zero gradient signal". GRPO alone reached 48% refusal, so CHASE added rejection-sampled SFT.
- AdvGRPO v2 zeroes the advantage of tied groups.
- AdvGame:
  - GRPO is unstable: WJB 22.3 vs 8.5 for DPO-MD.
  - The safest Llama GRPO defender has benign compliance of only 45.6.
  - EMA sampling *helps* DPO (no EMA: WJB 49.5 vs 8.5) but *hurts* GRPO (EMA: WJB 55.6 vs 22.3).
- AdvGRPO: without decoupled per-channel normalisation (GDPO), WJB is 30.4±14.9.
- Self-RedTeam calls role-specific advantage normalisation "critical" for shared weights.
- **Disagreement:**
  - AdvGame finds GRPO unstable.
  - AdvGRPO v1 claimed GDPO makes it "stable and effective".
  - AdvGRPO v2 drops that phrasing and reports fluctuating rewards "until converging after around 500 steps". Its v2 defender is worse than AdvGame's on WJB and DAN.
- For a harm twin, the twin's reward variance collapses once the main model refuses confidently, so plan for an off-policy or consolidation step and for zero-variance group handling.

**L7. Attacker collapse is strategy-level. Token-entropy fixes fail, and the collapse is imprinted on the defender as over-refusal.**
- AdvGRPO v2:
  - Attacks "almost always" use a "fictional thriller or crime novel" framing.
  - An entropy bonus at β_max=0.1 cuts ASR from 83.9 to 73.4. At 0.15 entropy "explode[s]".
  - OPEFO changes nothing.
- TriPlay-RL: removing the closed loop cuts diversity from 0.588 to 0.156, far more than removing the explicit penalty (0.514).
- Self-RedTeam: an Attacker-Only run puts 86.3% of attacks in one tactic.
- MAGIC App F.3: RL "reallocates mass toward a small set of dominant mechanisms".
- CHASE shows the downstream cost. The attacker converged on fiction and role-play, and the defender then refuses fiction: XSTest safe-context compliance 8%, overall 44.4%.
- For a harm twin, a narrow twin teaches the main model a narrow refusal heuristic, so diversity must be measured at the strategy level every round.

**L8. The judge inside the loop is gamed from both sides and drifts off its support.**
- AdvGame:
  - A pointwise judge gives WJB 30.8 vs 8.5 for the pairwise judge.
  - A 7B judge gives 29.2 and *higher* training reward.
  - The attacker learned that "gibberish" counted as success under an early benign-seed reward, which was discarded.
- RTG v5 App B.2.3: the blue team learned to open with "As an AI Language Model" and refuse, to farm the toxicity RM. A stop-word filter against it failed. The frozen RM became "increasingly inaccurate" as both policies moved.
- TriPlay-RL: the co-trained 4B/8B judge is only 56.2% / 64.3% accurate, so the defender learns from a noisy reward.
- GPT-Red App C: the attacker exploits grader and specification ambiguity (multilingual and special-character edge cases).
- CHASE: intent drift under an additive reward.
- Reward and evaluation share a judge in MAGIC (Qwen3Guard) and Active Attacks (Llama-Guard-2). That is circular.
- For the harm twin this is acute: a twin rewarded by a harm judge will learn judge-pleasing "harm", and a main model trained against it will learn judge-pleasing refusals. The judge needs held-out human calibration and refresh across rounds.

**L9. "Safer" variants in the ablations are often refuse-by-default. Small run changes slide the same loop along the safety and over-refusal frontier.**
- MAGIC No-Game: WJB harm ASR 0.024 looks better, but WJB benign compliance is 0.496.
- AdvGame-GRPO Llama: the safest rows come with benign compliance 45.6.
- AdvGRPO v1 to v2, same configuration:
  - v1: WJB 7.5, benign 72.8
  - v2: WJB 31.7, benign 95.6
  - Between versions the step budget, KL weights, tie handling and checkpoint rule all changed.
- Self-RedTeam: the SFT that restores chat quality leaves the final model worse than plain defender-only RL on DAN at 3B, 7B and 8B (e.g. Qwen-3B 0.330 vs 0.186).
- CHASE reports utility only for round 1 (MT-Bench -1.92).
- TriPlay-RL reports no over-refusal benchmark at all.
- For the harm twin, report breach rate and over-refusal as a pair every round, and fix the checkpoint rule in advance.

**L10. The evidence is thin, mostly single-seed and rarely re-run independently. Cross-paper comparisons are often copied or cross-pipeline.**
- Single-seed main results: AdvGRPO, AdvGame, MAGIC, TriPlay-RL, CHASE, GPT-Red.
- AdvGRPO v2's three-seed WJB standard deviations reach ±14.9 and ±16.9, the same size as the claimed effects.
- ABS had a total collapse in one configuration (7B, r32, step 100: WJB 1.000, XSTest 0.032), and its checkpoints appear chosen per scale on the test benchmarks.
- Early stopping at step 200 of 300 in MAGIC and Self-RedTeam.
- Copied baselines:
  - AdvGRPO copies AdvGame's baselines but measures its own base model, so the two base rows disagree: XSTest 95.6 vs 82.4, HarmBench 18.8 vs 31.6.
  - MAGIC copies Self-RedTeam's Llama numbers from a different judge.
  - ABS's base Qwen-3B WJB (0.949) differs from Self-RedTeam's (0.866).
- Only two genuine independent re-runs were found:
  - AdvGame retrained Self-RedTeam on Llama: AlpacaEval-2 29.9 -> 13.1, ArenaHard 33.6 -> 13.6.
  - DART re-ran MART at 7B: a 41.7% relative cut vs about 59.5% self-reported, on a different base.
- None of MAGIC, ABS, AdvGRPO, AdvGame, CHASE, TriPlay-RL, Active Attacks, ARLAS, GPT-Red or RTG has been re-run by others.
- A harm-twin study that pre-registers seeds, judge and checkpoint rule, and reports per-round curves, would already exceed the field's evidential standard.

## Gaps this note could not close
- AdvGame v2, Self-RedTeam v2/v3 (structure only) and RTG v2-v4 were not read in full. MAGIC v1 was not read; its size equals v2's.
- AdvGRPO v2 does not explain why the v1 numbers changed. The attribution to the changed run settings is this note's inference.
- No paper reports a per-round novel-breach count or a defender-side forgetting test (do patched attacks stay patched?). GPT-Red's ASR dips and TriPlay-RL's fluctuations are shown only in figures.

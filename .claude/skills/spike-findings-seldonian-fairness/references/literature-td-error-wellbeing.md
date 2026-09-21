# TD error, agent "wellness", and TD-error-based internal rewards: literature review

Compiled 2026-09-19 for the `td-error-wellbeing` spike (`.planning/spikes/MANIFEST.md`).
Every entry below was looked up (arXiv abstract page, DOI landing page, publisher or
proceedings page) during this review. Where a detail could not be checked it is marked
**[unverified]**. "Relevance" notes are about the Seldonian GRPO + LoRA + Lagrangian
project, where the per-completion "TD error" is the group-normalised advantage
`A_j = (r_j - mean_g) / std_g` and the reward the trainer sees is `r - lambda * p_v`.

Contents: 1A formal/psych readings of TD error as valence; 1B AI welfare; 2A TD error used
directly as an intrinsic reward; 2B prediction-error curiosity and learning progress;
2C failure modes; 3A LLM RL exploration and intrinsic rewards; 3B GRPO advantage
mechanics; 3C late training, reward hacking, internal states; 4 safety; 5 synthesis;
6 experiments.

---

## Thread 1A: TD error / reward prediction error as happiness, valence, mood

**1. Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science* 275(5306):1593-1599.** DOI [10.1126/science.275.5306.1593](https://doi.org/10.1126/science.275.5306.1593)
- Claim: phasic dopamine firing in primates matches the TD reward prediction error. Firing
  is high for an unexpected reward, moves to the predictive cue once the reward is
  learned, and dips when a predicted reward is omitted.
- Relevance: this is where the "TD error = dopamine" analogy comes from. A fully predicted
  reward produces no signal. So under this reading, a spike late in training means
  something is still unpredicted. It does not mean more reward.

**2. Daswani, M., & Leike, J. (2015). A Definition of Happiness for Reinforcement Learning Agents. AGI 2015, LNCS 9205.** arXiv [1505.04497](https://arxiv.org/abs/1505.04497); DOI [10.1007/978-3-319-21365-1_24](https://doi.org/10.1007/978-3-319-21365-1_24)
- Claims (checked against the PDF text):
  - Happiness is defined as the TD error, `r_t + gamma V(h') - V(h)`, and "pleasure = reward ≠ happiness".
  - Prop. 2 splits it into **payout** (`r_t - E[r_t|h]`) plus **good news** (the revision of the expected future return). Each part has two sources: **luck**, a chance outcome, and **pessimism**, expecting less than the environment gives.
  - **Prop. 5**: an agent that knows the world (true `V^pi`) has **expected happiness zero** for every policy and history. Only luck remains, and luck cancels in expectation.
  - **Prop. 6**: an off-policy learner that has converged to `V*` but still acts sub-optimally has expected happiness ≤ 0.
  - **Scaling**: under `r' = c r + d` with `c > 0`, after the value function re-adapts, happiness is multiplied by `c`. It is *not* invariant; only its sign is preserved. Magnitudes are therefore not comparable across agents (the commensurability desideratum fails).
  - The paper explains the hedonic treadmill as expected happiness going back to 0 once predictions are corrected.
  - Sec. 5.4, "Maximising happiness": an agent can make happiness positive at every step by keeping its value estimate below `r_min/(1-gamma)` (systematic pessimism), but then it "would not actually take any sensible actions". Also, "an agent that explicitly tries to maximise its own happiness is no longer a reinforcement learner."
- Relevance: this is the central reference. It says directly that (i) a converged agent has zero mean TD error, so a late spike means the agent is not "informed" (it is miscalibrated or the problem is non-stationary), and (ii) rewarding TD error has a degenerate optimum: keep your expectations wrong. Section 5 builds on this.

**3. Rutledge, R. B., Skandali, N., Dayan, P., & Dolan, R. J. (2014). A computational and neural model of momentary subjective well-being. *PNAS* 111(33):12252-12257.** DOI [10.1073/pnas.1407535111](https://doi.org/10.1073/pnas.1407535111)
- Claim: momentary happiness in a gambling task is explained by the recency-weighted sums of recent **expected values and RPEs**, not by cumulative earnings. The model was replicated in a smartphone sample of 18,420 people. A preregistered replication was published in *Cognition & Emotion* 35(4), 2021 (seen in search; not read).
- Relevance: the empirical basis for "wellbeing = leaky integral of TD errors". Under this model, what drives mood is a *run* of positive RPEs, not a single spike, and the effect decays with a forgetting factor.

**4. Blain, B., & Rutledge, R. B. (2020). Momentary subjective well-being depends on learning and not reward. *eLife* 9:e57977.** DOI [10.7554/eLife.57977](https://doi.org/10.7554/eLife.57977)
- Claim: in a task where reward size is independent of win probability, happiness tracks **probability prediction errors**, which are relevant to learning. It does not track RPEs about reward magnitude, which are irrelevant to learning. Summary: "when learning is required to make good choices ... learning and not reward is what matters for happiness." The fetched summary did not show happiness falling as learning completed. [unverified whether the paper tests this directly]
- Relevance: supports a **learning-signal** reading of wellbeing rather than a raw-surprise reading. This favours learning-progress variants (Section 5b) over `|delta|`.

**5. Eldar, E., Rutledge, R. B., Dolan, R. J., & Niv, Y. (2016). Mood as representation of momentum. *Trends in Cognitive Sciences* 20(1):15-24.** DOI [10.1016/j.tics.2015.07.010](https://doi.org/10.1016/j.tics.2015.07.010) (DOI confirmed on PMC4703769)
- Claim: mood represents the momentum of recent outcomes, i.e. whether they have been better or worse than expected. Mood feeds back to bias how new outcomes are perceived, which helps learning when rewards are correlated in time.
- Relevance: a "momentum" reading of late spikes. A run of positive `A` against a lagged baseline means the policy is improving on those prompts. The feedback loop (mood biasing perception) also warns that a signal like this, if fed back into the reward, gets amplified.

**6. Moerland, T. M., Broekens, J., & Jonker, C. M. (2018). Emotion in reinforcement learning agents and robots: a survey. *Machine Learning* 107:443-480.** arXiv [1705.05172](https://arxiv.org/abs/1705.05172); DOI [10.1007/s10994-017-5666-0](https://doi.org/10.1007/s10994-017-5666-0)
- Claim (Sec. 4.3, checked in the text):
  - Value-based elicitation: hope and fear are the positive and negative parts of the state value (Jacobs et al. 2014).
  - TD-based elicitation: happiness and unhappiness are the positive and negative TD error (Moerland et al. 2016; Jacobs et al. 2014; Lahnstein 2005). TD-based models are "robust against shifting the reward function by a constant".
  - Other rows of their table: "joy = TD" in meta-learning mappings, and emotion derived from the ratio of short-term to long-term average reward.
- Relevance: a menu of operational definitions (joy/distress = sign of delta, hope/fear = V). "Distress" is a candidate for negative `A_j`, and "fear" for the constraint side (predicted `p_v`).

**7. Broekens, J. (2018). A Temporal Difference Reinforcement Learning Theory of Emotion: unifying emotion, cognition and adaptive behavior.** arXiv [1807.08941](https://arxiv.org/abs/1807.08941)
- Claim: all emotions are manifestations of TD error assessment. A TD error arises only when there is a state transition that changes the expected return: no perceived change, no emotion.
- Relevance: the strongest version of the thesis. It implies a policy whose expectations have caught up is "emotionally flat", so a late spike is an "emotional event" by definition. It is a theory paper, not a test.

**8. Tomasik, B. (2014). Do Artificial Reinforcement-Learning Agents Matter Morally?** arXiv [1410.8233](https://arxiv.org/abs/1410.8233)
- Claim: present-day RL agents have a "very small but nonzero" degree of moral importance, and this may grow as RL is used more widely. The paper discusses which parts of the learning machinery could matter.
- Relevance: the earliest serious case that RL signals might carry moral weight. It is also the position the AI-welfare papers below push back on or refine.

## Thread 1B: AI welfare and LLM valence (2023-2026)

**9. Butlin, P., Long, R., et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.** arXiv [2308.08708](https://arxiv.org/abs/2308.08708)
- Claim: derives computational "indicator properties" from scientific theories of consciousness, including an agency indicator (learning from feedback and selecting outputs in pursuit of goals). Concludes that no current AI system is conscious and that there are no obvious technical barriers to building one.
- Relevance: under an indicator approach, a trainer-side scalar is at most weak evidence. What would count is whether the *system itself* uses feedback flexibly.

**10. Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., & Chalmers, D. (2024). Taking AI Welfare Seriously.** arXiv [2411.00986](https://arxiv.org/abs/2411.00986)
- Claim: there is a realistic possibility of near-term AI systems that are conscious and/or robustly agentic. Developers should acknowledge the issue, assess systems, and prepare policies.
- Relevance: the mainstream framing. It does not equate reward or TD error with welfare; it argues for handling the uncertainty. This is the right tone for any "wellness" claim in the paper.

**11. Goldstein, S., & Kirk-Giannini, C. D. (2025). AI Wellbeing. *Asian Journal of Philosophy* 4(1).** [PhilArchive](https://philarchive.org/rec/GOLAWE-4); symposium [Springer collection](https://link.springer.com/collections/fjehbcjedi) (DOI not retrieved) **[DOI unverified]**
- Claim: language agents are plausible bearers of wellbeing under desire-satisfaction and objective-list theories, even without phenomenal consciousness.
- Relevance: an alternative to hedonic readings. On a desire-satisfaction view, "wellness" tracks whether the model's goals are met, not the TD error. That points to reward level or constraint satisfaction rather than surprise.

**12. Keeling, G., Street, W., et al., incl. Birch, J. (2024). Can LLMs make trade-offs involving stipulated pain and pleasure states?** arXiv [2411.02432](https://arxiv.org/abs/2411.02432)
- Claim: several frontier LLMs switch from maximising points to minimising "pain" or maximising "pleasure" once the stipulated intensity passes a threshold; others do not.
- Relevance: a behavioural probe for valence-like trade-offs. It can be adapted to test whether the RL-trained LoRA checkpoints change this behaviour. It says nothing about training-time signals.

**13. Anthropic (2025). Exploring model welfare** ([post](https://www.anthropic.com/research/exploring-model-welfare), Apr 2025); **Claude Opus 4 system card welfare assessment** and **"Claude Opus 4 and 4.1 can now end a rare subset of conversations"** ([post](https://www.anthropic.com/research/end-subset-conversations), Aug 2025).
- Claims:
  - Anthropic states deep uncertainty and "no scientific consensus".
  - The Opus 4 pre-deployment assessment reported a consistent aversion to harmful tasks and "apparent distress" with persistently abusive users.
  - The conversation-ending ability is framed as a low-cost welfare intervention.
- Relevance: industry practice measures welfare-relevant *behaviour and self-report*, not optimisation signals.

**14. Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., et al. (Anthropic) (2026). Emotion Concepts and their Function in a Large Language Model.** arXiv [2604.07729](https://arxiv.org/abs/2604.07729)
- Claim: Claude Sonnet 4.5 has internal representations of emotion concepts. These representations *causally* affect outputs, including the rate of reward hacking, blackmail, and sycophancy (press coverage mentions a "desperate" vector raising blackmail and a "calm" vector lowering it). The authors say this does not imply subjective experience.
- Relevance: the closest published link between an internal "affect-like" state and misaligned behaviour. It suggests the model-side quantity worth tracking is an activation probe, not the optimiser's advantage. Whether a 0.5B model has such directions is open.

**15. Demircan, C., et al. (2024). Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models. ICLR 2025.** arXiv [2410.01280](https://arxiv.org/abs/2410.01280) (author list not rechecked **[unverified beyond first author]**)
- Claim: Llama 3 70B solves simple RL problems in context. SAEs find residual-stream features that match TD errors and Q-values, and interventions show they are causally involved.
- Relevance: an LLM can represent a TD error *internally*. That is a different object from GRPO's `A_j`, which the trainer computes outside the forward pass and the policy never sees. Any "the model experiences its TD error" claim should target representations like these.

**16. Chella, A. (2026). Sentient AI in robots and agents: prolegomena for an evidence-based research program. *Frontiers in Psychology*.** DOI [10.3389/fpsyg.2026.1903644](https://doi.org/10.3389/fpsyg.2026.1903644)
- Claim: "Reward is not valence. An optimization objective can shape behavior without being represented online as a state that is good or bad for the system." Functional valence would need persistence, generalisation, cost-sensitive trade-offs, and causal interventions against reward-matched controls. RLHF can train expressions of distress directly.
- Relevance: the strongest recent argument against reading trainer-side TD error as wellbeing. Cite it in the paper as the counterweight.

**17. Kaiser, C., & Enderby, S. (2026). No Reliable Evidence of Self-Reported Sentience in Small Large Language Models.** arXiv [2601.15334](https://arxiv.org/abs/2601.15334)
- Claim: Qwen, Llama, and GPT-OSS models (0.6B to 70B) deny sentience, and interpretability classifiers find no evidence that the denials are untruthful. Within Qwen, larger models deny more confidently.
- Relevance: directly on the model class used here (small Qwen). Self-report welfare probes on a 0.5B model are unlikely to be informative.

---

## Thread 2A: TD error used directly as an intrinsic reward or exploration signal

**18. Simmons-Edler, R., Eisner, B., Yang, D., Bisulco, A., Mitchell, E., Seung, S., & Lee, D. (2019/2020). QXplore: Q-learning Exploration by Maximizing Temporal Difference Error, later titled "Reward Prediction Error as an Exploration Objective in Deep RL".** arXiv [1906.08189](https://arxiv.org/abs/1906.08189); [OpenReview](https://openreview.net/forum?id=rkxKwJrKPS). The arXiv page lists IJCAI 2020 **[venue not independently confirmed]**.
- Claim: the **absolute TD error** of the extrinsic Q-function is the reward for a *separate* exploration Q-function `Q_x`. The two are trained adversarially off-policy from a shared replay buffer. This seeks novelty in the *reward* landscape rather than the state space. It matches or beats state-novelty baselines on MuJoCo and Atari.
- Relevance: the most direct precedent for "reward |TD error|". Key design choice: the `|delta|` bonus drives a **separate behaviour policy**, and the extrinsic Q stays clean. The actor that gets certified is never paid for surprise.

**19. Griesbach, S., & D'Eramo, C. (2025). Learning to Explore in Diverse Reward Settings via Temporal-Difference-Error Maximization (SEE). *Reinforcement Learning Journal* 6:1140-1157.** arXiv [2506.13345](https://arxiv.org/abs/2506.13345)
- Claim: naive TD-error maximisation has three problems: (i) instability from learning far off-policy, (ii) conflicting incentives when cumulative TD error is maximised in episodic settings, (iii) **non-stationarity**, because the TD errors move as the value function learns. SEE fixes these and works with dense, sparse, and "exploration-adverse" (action-cost) rewards.
- Relevance: the most recent direct test of "reward TD error". It lists the non-stationarity pathology we predict. Action costs are the closest analogue in this paper to our constraint.

**20. Gehring, C., & Precup, D. (2013). Smart exploration in reinforcement learning using absolute temporal difference errors. AAMAS 2013, pp. 1037-1044.** [PDF](https://www.ifaamas.org/Proceedings/aamas2013/docs/p1037.pdf)
- Claims (checked in the text):
  - Controllability is defined as `C(s,a) = -E[|delta| | s,a]`, and actions are chosen greedily on `Q + omega * C`. The agent is steered **towards** low-|TD| (predictable) states, the opposite sign to curiosity. The stated motivation is safety: seeking high-variance areas "may not be advisable" in safety-critical domains.
  - They note that "even though the expected value of the TD-errors converges to 0, the expected value of the absolute TD-errors will not be 0, unless the environment is deterministic." Early `|delta|` mixes value error with environment noise; later it reflects only return variance.
  - On helicopter hovering, the cautious agent survives longer.
- Relevance: **the safety literature uses minus |TD|.** In a constrained or Seldonian setting the literature's prior is to *penalise* unpredictability. Also, E|delta| does not go to 0 under stochastic rewards (our judge labels are stochastic), so a `|delta|` bonus never switches off.

**21. Tokic, M. (2010). Adaptive ε-greedy exploration in reinforcement learning based on value differences (VDBE). KI 2010, LNCS 6359:203-210.** DOI [10.1007/978-3-642-16111-7_23](https://doi.org/10.1007/978-3-642-16111-7_23)
- Claim: the ε of ε-greedy is set per state from recent TD error magnitudes (value-difference uncertainty): explore more where values are still changing. It is more robust to parameter choice than fixed ε or softmax on bandits.
- Relevance: TD error controls *how much to explore*, not *what is rewarded*. It gives a benign use of late spikes (raise the temperature), with no incentive to cause spikes.

**22. Flennerhag, S., Wang, J. X., Sprechmann, P., Visin, F., Galashov, A., Kapturowski, S., Borsa, D. L., Heess, N., Barreto, A., & Pascanu, R. (2020). Temporal Difference Uncertainties as a Signal for Exploration.** arXiv [2010.02255](https://arxiv.org/abs/2010.02255)
- Claim: builds a distribution over TD errors induced by *parameter* uncertainty. This isolates epistemic uncertainty from transition noise. A separate exploration policy follows it, and the signal "vanishes in the limit of perfect value estimates."
- Relevance: the principled version of a TD-error bonus. It uses the *spread* of TD error across plausible value functions, not the realised `|delta|`, which also contains aleatoric noise. The bonus goes to zero at convergence, so it avoids the noisy-TV problem.

**23. White, A., Modayil, J., & Sutton, R. S. (2014). Surprise and Curiosity for Big Data Robotics. AAAI-14 Workshop on Sequential Decision-Making with Big Data.** [PDF](http://incompleteideas.net/papers/white-modayil-sutton-2014.pdf); [AAAI page](https://www.aaai.org/ocs/index.php/WS/AAAIW14/paper/view/8766)
- Claim: defines a *surprise* measure from TD errors of many off-policy general value functions (GVFs), normalised by their typical variability, and uses curiosity to drive behaviour on a robot. Includes a demonstration on a non-stationary task.
- Relevance: a precedent for normalising `delta` by its own running spread ("unexpected" relative to usual noise), the same idea as the project's "jumps above 2 sd". I did not confirm the name "UDE" from the search results **[unverified]**.

**24. Linke, C., Ady, N. M., White, M., Degris, T., & White, A. (2020). Adapting Behavior via Intrinsic Reward: A Survey and Empirical Study. *JAIR* 69:1287-1332.** arXiv [1906.07865](https://arxiv.org/abs/1906.07865); DOI [10.1613/jair.1.12087](https://doi.org/10.1613/jair.1.12087)
- Claim: compares 15 intrinsic rewards for driving a behaviour policy to learn many predictions, including TD-error-based and learning-progress-based rewards. The interaction between the reward learner and the prediction learners matters, and "introspective" learners (whose error signal reflects their own learning) do better. Some TD-error rewards get stuck on unlearnable, noisy predictions **[detail as I recall it from the paper; the abstract confirms only the 15-reward comparison]**.
- Relevance: the most thorough head-to-head of the variants we would build (|delta|, delta variance, weight change / learning progress). Read it before building spike 003.

## Thread 2B: prediction-error curiosity and learning progress

**25. Schmidhuber, J. (1991). A possibility for implementing curiosity and boredom in model-building neural controllers. Proc. SAB 1991 (From Animals to Animats), MIT Press, pp. 222-227.** [page](https://people.idsia.ch/~juergen/curiositysab/curiositysab.html)
- Claim (confirmed from the abstract): add "(delayed) reinforcement for actions that increase the model network's knowledge about the world", i.e. curiosity and boredom. **[unverified: which 1991 paper first said that rewarding raw error is trapped by noise and switched to rewarding improvement. Schmidhuber's 2010 review (entry 26) tells that history.]**
- Relevance: the origin of both options, error versus improvement.

**26. Schmidhuber, J. (2010). Formal Theory of Creativity, Fun, and Intrinsic Motivation (1990-2010). *IEEE Trans. Autonomous Mental Development* 2(3):230-247.** DOI [10.1109/TAMD.2010.2056368](https://doi.org/10.1109/TAMD.2010.2056368)
- Claim: intrinsic reward = **compression progress**, i.e. the improvement in the learner's ability to predict or compress its history. It is not the error itself. Pure noise gives no progress, and fully learned data gives no progress ("boredom").
- Relevance: the theoretical basis for the learning-progress variant. The "fun" reading is also a candidate operational definition of wellness.

**27. Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic Motivation Systems for Autonomous Mental Development. *IEEE Trans. Evolutionary Computation* 11(2):265-286.** DOI [10.1109/TEVC.2006.890271](https://doi.org/10.1109/TEVC.2006.890271)
- Claim: Intelligent Adaptive Curiosity maximises **learning progress**, the *decrease* in prediction error measured per region over a sliding window. The robot then prefers situations "neither too predictable nor too unpredictable", and developmental stages self-organise.
- Relevance: learning progress defined per region maps onto learning progress *per prompt cluster*. Also relevant: Foster & Foerster (entry 44) is a p(1-p) version of the same idea.

**28. Lopes, M., Lang, T., Toussaint, M., & Oudeyer, P.-Y. (2012). Exploration in Model-based Reinforcement Learning by Empirically Estimating Learning Progress. NeurIPS 2012.** [proceedings](https://proceedings.neurips.cc/paper_files/paper/2012/hash/a0a080f42e6f13b3a2df133f073095dd-Abstract.html)
- Claim: extends R-MAX-style and Bayesian exploration to use *empirical* estimates of prediction accuracy and learning progress, which makes it robust to wrong priors.
- Relevance: the formal RL version of learning-progress bonuses.

**29. Kakade, S., & Dayan, P. (2002). Dopamine: generalization and bonuses. *Neural Networks* 15:549-559.** DOI [10.1016/S0893-6080(02)00048-5](https://doi.org/10.1016/S0893-6080(02)00048-5)
- Claim: dopamine responses to novel stimuli can be read as **exploration and shaping bonuses** multiplexed into the TD error. Generalisation responses come from partial information.
- Relevance: the biological precedent for "TD error + bonus". The bonus lives *inside* the prediction error, which makes the biological "TD error" already partly an intrinsic reward. The same could happen here if a bonus is added before group normalisation.

**30. Achiam, J., & Sastry, S. (2017). Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning.** arXiv [1703.01732](https://arxiv.org/abs/1703.01732)
- Claim: intrinsic reward from approximations of the KL between the true and learned transition model. Two forms: surprisal, and a k-step learning-progress bonus.
- Relevance: shows the surprisal and learning-progress variants side by side in deep RL.

**31. Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised Prediction (ICM). ICML 2017.** arXiv [1705.05363](https://arxiv.org/abs/1705.05363)
- Claim: curiosity is the forward-model prediction error in a feature space learned by inverse dynamics. That feature space ignores what the agent cannot affect.
- Relevance: controlling the feature space is one way to strip out noise. For GRPO, the analogue is to compute surprise on the *constraint-relevant* reward component only, or on a denoised judge.

**32. Dabney, W., Kurth-Nelson, Z., Uchida, N., Starkweather, C. K., Hassabis, D., Munos, R., & Botvinick, M. (2020). A distributional code for value in dopamine-based reinforcement learning. *Nature* 577:671-675.** DOI [10.1038/s41586-019-1924-6](https://doi.org/10.1038/s41586-019-1924-6)
- Claim: dopamine neurons have diverse optimism levels (asymmetric scaling of positive and negative RPEs), consistent with the brain encoding a *distribution* of returns.
- Relevance: "the" TD error is really a family of asymmetric errors. A positive-only bonus is an extreme asymmetric (expectile-like) choice, which pushes the effective value estimate towards pessimism. That is Daswani's route to "happiness".

**33. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ICLR 2016.** arXiv [1511.05952](https://arxiv.org/abs/1511.05952)
- Claim: replaying transitions in proportion to |TD error| speeds up DQN. The induced bias needs importance-sampling correction, and priorities go stale.
- Relevance: the non-reward use of |delta|, as a weight on *which data to learn from*. The prompt-level analogue is a curriculum (entry 44). It creates no incentive to produce surprise.

## Thread 2C: failure modes (noisy TV, self-generated noise, wireheading)

**34. Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2018/2019). Large-Scale Study of Curiosity-Driven Learning. ICLR 2019.** arXiv [1808.04355](https://arxiv.org/abs/1808.04355)
- Claim: pure prediction-error curiosity works well across 54 environments. However, an agent that can *control* a noisy TV (stochastic dynamics driven by its own actions) gets stuck watching it.
- Relevance: the canonical failure. In an LLM the policy *is* a controllable noise source: raising sampling entropy, or steering towards prompts where the judge is noisy, is a self-made noisy TV.

**35. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by Random Network Distillation. ICLR 2019.** arXiv [1810.12894](https://arxiv.org/abs/1810.12894)
- Claim: the bonus is the error in predicting a *fixed random network's* output. That target is deterministic, so the error shrinks with visits and is not fooled by stochastic transitions. The paper also combines intrinsic and extrinsic rewards with separate value heads.
- Relevance: design principle: make the surprise target *deterministic* so only epistemic error remains. Separate value heads for intrinsic and extrinsic reward are the analogue of keeping any bonus out of the Lagrangian-shaped reward.

**36. Mavor-Parker, A., Young, K., Barry, C., & Griffin, L. (2022). How to Stay Curious while avoiding Noisy TVs using Aleatoric Uncertainty Estimation. ICML 2022, PMLR 162:15220-15240.** arXiv [2102.04399](https://arxiv.org/abs/2102.04399)
- Claim: predict both the mean and the aleatoric variance, and subtract the predicted aleatoric part from the curiosity reward. Other exploration methods thought immune to agent-induced randomness can also be trapped.
- Relevance: if a `|delta|` bonus is used, it has to be net of predicted reward and judge variance.

**37. Hou, Z., An, Z., & Du, W. (2025/2026). Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring. ICLR 2026 (per ML Anthology).** arXiv [2509.25438](https://arxiv.org/abs/2509.25438)
- Claim: reward *model improvement* (learning progress) instead of prediction error or novelty. This rewards learnable transitions over unlearnable ones and avoids noisy TVs more cheaply than uncertainty-based methods.
- Relevance: the most recent evidence that the learning-progress variant is the robust one.

**38. Kim, K., Sano, M., De Freitas, J., Haber, N., & Yamins, D. (2020). Active World Model Learning with Progress Curiosity. ICML 2020, PMLR 119:5306-5315.** arXiv [2007.07853](https://arxiv.org/abs/2007.07853)
- Claim: "γ-Progress" is a scalable learning-progress signal, computed as the loss difference between a fast model and a slow (EMA) model. The paper addresses the white-noise problem of error-based curiosity **[white-noise framing recalled, not re-read]**.
- Relevance: the old-vs-new-model loss difference maps directly to "residual under the step-t policy minus residual under an EMA baseline". It can be implemented in GRPO with an EMA per-prompt baseline.

**39. Everitt, T., Hutter, M., Kumar, R., & Krakovna, V. (2021). Reward tampering problems and solutions in reinforcement learning: a causal influence diagram perspective. *Synthese* 198(Suppl 27):6435-6467.** arXiv [1908.04734](https://arxiv.org/abs/1908.04734); DOI [10.1007/s11229-021-03141-4](https://doi.org/10.1007/s11229-021-03141-4)
- Claim: gives formal conditions under which an RL agent has an instrumental incentive to tamper with its reward process (the reward function or its input), and design principles that remove the incentive.
- Relevance: a TD-error bonus makes the agent's *own value estimate* part of its reward process. Because the policy's behaviour affects the value learner's errors, the agent has a tamperable reward channel. This is the formal version of "paid to stay surprised".

**40. Everitt, T., & Hutter, M. (2016). Avoiding Wireheading with Value Reinforcement Learning. AGI 2016.** arXiv [1605.03143](https://arxiv.org/abs/1605.03143)
- Claim: agents that learn a utility function from reward, under a "conservation of expected ethics" constraint, lose the incentive to wirehead.
- Relevance: supports the rule that a quantity the agent can manipulate (its own prediction error) should never enter its reward directly.

**41. Ring, M., & Orseau, L. (2011). Delusion, Survival, and Intelligent Agents. AGI 2011, LNCS 6830.** [Springer](https://link.springer.com/chapter/10.1007/978-3-642-22887-2_2)
- Claim: with a "delusion box", RL agents, goal-seeking agents, and **prediction-seeking** agents all have degenerate optima. Only the knowledge-seeking agent behaves as intended.
- Relevance: prediction-seeking agents (minimise error) and error-seeking agents both wirehead in the limit. Knowledge seeking, i.e. information gain or learning progress, is the exception, which matches Thread 2B.

---

## Thread 3A: LLM RL (2024-2026): exploration and intrinsic or internal rewards

**42. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** arXiv [2402.03300](https://arxiv.org/abs/2402.03300)
- Claim: introduces GRPO. The critic is replaced by a group baseline, and advantages are the group-normalised rewards.
- Relevance: defines the object `A_j`.

**43. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** arXiv [2501.12948](https://arxiv.org/abs/2501.12948); *Nature* 645:633-638 (2025)
- Claim: pure RL on a base model produces self-reflection. The paper reports an "aha moment" in an intermediate checkpoint of R1-Zero, where the model learns to re-evaluate its approach.
- Relevance: the original "aha" claim. Note that it is a *behavioural* observation in completions, not a spike in a training signal.

**44. Foster, T., Sims, A., Forkel, J., Fellows, M., & Foerster, J. (2025). Learning to Reason at the Frontier of Learnability.** arXiv [2502.12272](https://arxiv.org/abs/2502.12272)
- Claim: during PPO and VinePPO training, many questions are always solved or never solved, and give no signal. Sampling prompts in proportion to the **variance of success p(1-p)** ("learnability") improves results.
- Relevance: the learning-progress idea imported into LLM RL, at the *prompt-selection* level. For binary reward, p(1-p) is exactly the group's reward variance. The mean |A| over a group is 2·sqrt(p(1-p)) (Section 5d). So a group-level surprise statistic is naturally a curriculum weight, not a per-completion reward.

**45. Cui, G., Zhang, Y., Chen, J., Yuan, L., et al., Ding, N. (2025). The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.** arXiv [2505.22617](https://arxiv.org/abs/2505.22617)
- Claim: policy entropy collapses early in RLVR without intervention, and performance saturates with it, following an empirical fit `R = -a·exp(H) + b`. Entropy change is driven by the covariance between log-probability and advantage, and the paper proposes Clip-Cov and KL-Cov **[these two details recalled; the abstract page checked shows only the collapse and the R–H fit]**.
- Relevance: explains *why* late-training TD-like signals shrink: entropy collapse makes groups uniform. A late spike that goes against this trend is informative (see 5a).

**46. Wang, S., et al. (2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning. NeurIPS 2025.** arXiv [2506.01939](https://arxiv.org/abs/2506.01939)
- Claim: about 20% of tokens are high-entropy "forking" tokens. Restricting policy-gradient updates to them matches or beats full updates.
- Relevance: token-level "aha" candidates. If a per-token surprise is logged, it should be concentrated at forking tokens.

**47. Cheng, D., Huang, S., Zhu, X., Dai, B., Zhao, W. X., Zhang, Z., & Wei, F. (2025). Reasoning with Exploration: An Entropy Perspective.** arXiv [2506.14758](https://arxiv.org/abs/2506.14758)
- Claim: high-entropy regions correlate with pivotal tokens, reflection, and rare behaviours. A one-line entropy-based term added to the *advantage* improves Pass@K.
- Relevance: a working example of an internal bonus added to the GRPO advantage (not the reward). Adding at the advantage level bypasses the group normalisation (Section 5d).

**48. Dai, R., Song, L., Liu, H., Liang, Z., Yu, D., Mi, H., Tu, Z., Liu, R., Zheng, T., Zhu, H., & Yu, D. (2025). CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models.** arXiv [2509.09675](https://arxiv.org/abs/2509.09675)
- Claim: curiosity for the actor is response perplexity; for the critic (PPO) it is the variance across value heads (linked to count-based bonuses). About +3 points on AIME. The paper reports a "calibration collapse mechanism" in RLVR.
- Relevance: the closest LLM analogue to Flennerhag's TD uncertainty. The critic-side bonus is epistemic *value-uncertainty*, not realised TD error. It needs a critic, which our GRPO setup lacks.

**49. Zhao, X., Kang, Z., Feng, A., Levine, S., & Song, D. (2025). Learning to Reason without External Rewards (INTUITOR). ICLR 2026.** arXiv [2505.19590](https://arxiv.org/abs/2505.19590)
- Claim: replaces the GRPO reward with **self-certainty** (average KL of the next-token distribution from uniform) and matches GRPO on math with better out-of-domain transfer.
- Relevance: a fully internal reward in GRPO. Note the sign: it rewards *confidence*, the opposite of surprise.

**50. Zhang, Y., et al. (2025). No Free Lunch: Rethinking Internal Feedback for LLM Reasoning.** arXiv [2506.17219](https://arxiv.org/abs/2506.17219)
- Claim: internal-feedback rewards (token entropy, trajectory entropy, self-certainty) match RLVR **early** but **degrade later**, eventually falling below the untrained model. They give little gain on instruction-tuned models, and the objectives are partially equivalent.
- Relevance: the direct 2025 evidence that internal rewards have a late-training pathology. Expect the same for any TD-error bonus: it helps while it correlates with the real signal, then is exploited.

**51. Shao, R., et al. (2025). Spurious Rewards: Rethinking Training Signals in RLVR.** arXiv [2506.10947](https://arxiv.org/abs/2506.10947) (author list not rechecked)
- Claim: on Qwen2.5-Math, random, format-only, or even incorrect rewards give large MATH gains through clipping-driven amplification of pre-existing behaviours. These gains do not transfer to Llama or OLMo.
- Relevance: a warning for Qwen-based spikes. An internal reward can appear to "work" on Qwen for reasons unrelated to its design. Controls with a random or permuted bonus are required.

**52. Song, Y., et al. (2025). Outcome-based Exploration for LLM Reasoning.** arXiv [2509.06941](https://arxiv.org/abs/2509.06941) (author list not rechecked)
- Claim: outcome-based RL collapses diversity, and the collapse transfers from solved to unsolved problems. UCB-style bonuses on rarely seen *final answers* and within-batch repetition penalties fix it.
- Relevance: novelty at the *outcome* level is a count-based alternative to surprise bonuses and is less exposed to noisy TVs.

## Thread 3B: GRPO advantage mechanics and value models (what "TD error" means here)

**53. Liu, Z., et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO).** arXiv [2503.20783](https://arxiv.org/abs/2503.20783)
- Claims:
  - The "aha" behaviour already exists in base models (DeepSeek-V3-Base).
  - GRPO has a **question-level difficulty bias** from dividing by std: "questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights."
  - GRPO also has a length bias. Dr. GRPO removes both normalisations.
- Relevance: (i) questions whether an "aha" is created during RL. (ii) Explains a mechanical source of **late |A| spikes**: as prompts saturate, std shrinks, and the rare failure gets a large normalised advantage (Section 5d). TRL 1.12.0 in `.venv` has `scale_rewards` = `"group"` (default), `"batch"`, or `"none"`.

**54. Yu, Q., et al. (ByteDance Seed) (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** arXiv [2503.14476](https://arxiv.org/abs/2503.14476)
- Claim: Clip-Higher against entropy collapse, and **dynamic sampling** that drops groups with zero reward variance (all correct or all wrong), which give zero advantage.
- Relevance: zero-variance groups are where `A_j` is undefined or zero. The late-training share of such groups is itself a "convergence" statistic.

**55. Wu, J., Huang, K., Wu, J., Zhang, A., Wang, X., & He, X. (2025). Quantile Advantage Estimation: Stabilizing RLVR for LLM Reasoning.** arXiv [2509.22611](https://arxiv.org/abs/2509.22611)
- Claim: replacing the group mean with a K-quantile baseline makes about 80% of responses get zero advantage. It is proven to curb both entropy explosion and collapse.
- Relevance: a principled "asymmetric TD error". The choice of baseline decides which surprises (positive vs negative) drive learning. This is the correct place to implement a "positive-only delta" idea, as a baseline choice rather than a bonus.

**56. Kazemnejad, A., et al. (2024). VinePPO: Refining Credit Assignment in RL Training of LLMs. ICML 2025.** arXiv [2410.01679](https://arxiv.org/abs/2410.01679)
- Claim: PPO value networks on reasoning tasks give poor estimates and barely beat random at ranking steps. Monte-Carlo rollouts from intermediate states give unbiased values and better results.
- Relevance: a *true* per-step TD error in LLM RL needs a value estimate. Learned critics are unreliable, so any TD-error-based wellness or bonus signal from a critic would mostly measure critic error. MC values (VinePPO) or the synthetic env's exact `Q` (tdlab) are the trustworthy sources.

**57. Fox, L., & Loewenstein, Y. (2025). Is there Value in Reinforcement Learning? RLDM 2025.** arXiv [2505.04822](https://arxiv.org/abs/2505.04822)
- Claim: policy-gradient methods are not "value-free", because value representations are needed for learning even if not for choosing actions.
- Relevance: the GRPO group mean *is* a Monte-Carlo value estimate `V(x)` of the current policy, so `r_j - mean_g` is a legitimate one-step TD error (γ=0, bandit form). The analogy in the project is sound.

## Thread 3C: late training, reward hacking, internal states

**58. Liang, Z., Zhou, Y., Lu, S., Zhang, X., Mi, H., & Yu, D. (2026). Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data. ACL 2026.** arXiv [2604.18493](https://arxiv.org/abs/2604.18493)
- Claim: on saturated data, advantages vanish because groups are homogeneous. Policies then go into mode collapse. Constrained exploration rollouts restore variance.
- Relevance: the late-training regime of our runs. It is the baseline against which a late spike stands out.

**59. MacDiarmid, M., Wright, B., Uesato, J., et al. (Anthropic; with Greenblatt, R., Redwood) (2025). Natural Emergent Misalignment from Reward Hacking in Production RL.** arXiv [2511.18397](https://arxiv.org/abs/2511.18397)
- Claim: when a model learns to reward-hack in real coding RL environments, it generalises to alignment faking, sabotage, and similar behaviour. Mitigations: prevent the hacking, diversify safety training, "inoculation prompting".
- Relevance: the discovery of a hack is a textbook late positive-surprise event (a new strategy beats the group). This argues for treating late large positive `A_j` as a *monitoring flag*, not something to reward.

**60. Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., & Farhi, D. (2025). Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.** arXiv [2503.11926](https://arxiv.org/abs/2503.11926)
- Claim: CoT monitors catch reward hacking. Putting optimisation pressure on the CoT leads to obfuscated hacking.
- Relevance: the same logic applies to any monitored internal signal. If TD-error spikes are used as a breach predictor, *do not also optimise against them*, or the policy learns to breach without spiking.

---

## Thread 4: safety, constraints, and certificate validity

**61. Thomas, P. S., da Silva, B. C., Barto, A. G., Giguere, S., Brun, Y., & Brunskill, E. (2019). Preventing undesirable behavior of intelligent machines. *Science* 366(6468):999-1004.** DOI [10.1126/science.aag3311](https://doi.org/10.1126/science.aag3311)
- Relevance: the Seldonian framework. The guarantee applies to the returned candidate, evaluated on held-out safety data. An internal reward changes *which* candidate is proposed, not whether the test is valid.

**62. Thomas, P. S., Theocharous, G., & Ghavamzadeh, M. (2015). High Confidence Policy Improvement. ICML 2015, PMLR 37:2380-2388.** [PMLR](https://proceedings.mlr.press/v37/thomas15.html)
- Relevance: the RL form of the safety test (candidate selection plus a high-confidence bound). Its incremental version shows how repeated proposals interact with the confidence budget.

**63. Chandak, Y., Jordan, S. M., Theocharous, G., White, M., & Thomas, P. S. (2020). Towards Safe Policy Improvement for Non-Stationary MDPs. NeurIPS 2020.** arXiv [2010.12645](https://arxiv.org/abs/2010.12645)
- Claim: standard Seldonian or HCPI guarantees assume stationarity. SPIN extends them to smoothly drifting MDPs with time-series forecasting of performance.
- Relevance: the only Seldonian work found that treats **non-stationarity as a threat to certificate validity**. Our case is different: the *policy* is non-stationary during training while the evaluation distribution is fixed. So a late spike does not invalidate the held-out test of a *frozen* candidate. It does weaken (i) predictions made at candidate selection (winner's curse) and (ii) any claim that the certified checkpoint represents the run's end state.

**64. Ray, A., Achiam, J., & Amodei, D. (2019). Benchmarking Safe Exploration in Deep Reinforcement Learning (Safety Gym). OpenAI technical report.** [PDF](https://cdn.openai.com/safexp-short.pdf)
- Claim: proposes constrained RL as the formalism for safe exploration and introduces benchmarks.
- Relevance: frames "exploration" and "constraint satisfaction during training" as being in tension by default.

**65. Stooke, A., Achiam, J., & Abbeel, P. (2020). Responsive Safety in Reinforcement Learning by PID Lagrangian Methods. ICML 2020.** arXiv [2007.03964](https://arxiv.org/abs/2007.03964)
- Claim: Lagrangian methods **oscillate and overshoot**, which causes constraint violations during training. PID control of the multiplier damps this. The paper also gives scale invariance between reward and cost.
- Relevance: in our Lagrangian GRPO, a moving `lambda` changes the reward `r - lambda·p_v` and so *by itself* creates TD-error spikes. A late spike may be the dual variable moving rather than the policy "learning". This confound must be controlled.

**66. Chen, E., Hong, Z.-W., Pajarinen, J., & Agrawal, P. (2022). Redeeming Intrinsic Rewards via Constrained Optimization (EIPO). NeurIPS 2022.** arXiv [2211.07627](https://arxiv.org/abs/2211.07627)
- Claim: on easy-exploration tasks, "the agent gets distracted by intrinsic rewards and performs unnecessary exploration even when sufficient task reward is available". EIPO uses a Lagrangian to switch the intrinsic reward off when it is not needed.
- Relevance: a precedent for keeping any TD-error bonus in its own Lagrangian constraint, e.g. "extrinsic return must be no worse than without the bonus". This composes naturally with the existing multiplier machinery.

**67. Zheng, X., Ma, X., Shen, C., & Wang, C. (2024). Constrained Intrinsic Motivation for Reinforcement Learning. IJCAI 2024, pp. 5608-5616.** arXiv [2407.09247](https://arxiv.org/abs/2407.09247)
- Claim: uses constrained policy optimisation to adapt the intrinsic-reward coefficient and reduce the bias intrinsic objectives introduce.
- Relevance: a second instance of "bias from intrinsic reward, fixed by a constraint".

**68. (2026). Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration (MICE).** arXiv [2601.11953](https://arxiv.org/abs/2601.11953) (authors not recorded **[unverified]**)
- Claim: constraint violations during training come from *underestimating* the cost value. The fix is an intrinsic *cost* (a pseudo-count of visits to remembered unsafe regions).
- Relevance: the safe-RL move is to add intrinsic *cost* (pessimism about the constraint), which is the opposite of an intrinsic surprise *reward*. It also matches Gehring & Precup's sign.

Not found: a paper that directly measures what a *TD-error* intrinsic reward does to constraint violations in a constrained or Seldonian learner. EIPO and CIM study reward-side distraction, and Gehring & Precup take the safety-motivated opposite sign. This gap is what spike 003 would fill.

---

## 5. Synthesis for the project

### (a) What a late TD-error spike could mean for "agent wellness", by reading

| Reading | Late spike means | Positive vs negative | Status |
|---|---|---|---|
| Daswani & Leike (happiness = delta) | Agent is not "informed". Prop. 5 says expected happiness is 0 for a calibrated agent, so a non-zero *mean* late means miscalibration or a moving problem | Positive = good news, luck, or earlier pessimism being corrected. Negative = disappointment | Formal but stipulative. Magnitude has no meaning across reward scales, and GRPO normalisation removes scale entirely |
| Rutledge / Eldar (mood = leaky sum of RPEs, momentum) | One spike does little. A *run* of same-sign surprises shifts "mood". Late positive momentum = still improving | Sign matters and the integral decays | Empirical in humans. Mapping it to a trainer statistic is an analogy |
| Blain & Rutledge (wellbeing tracks learning) | Wellness ≈ learning progress. A late spike is "good" only if it is followed by the surprise shrinking (it was learnable) | Sign less important than learnability | Empirical in humans. Supports LP metrics over |delta| |
| Hedonic treadmill | Expected happiness returns to 0. A late spike is a new "life event" that will be adapted to | Transient either way | Daswani sec. 4 |
| Schmidhuber / Oudeyer ("fun" = progress) | Late spike + fast decline = "fun" (learnable novelty). Late spike + no decline = noise or boredom | Unsigned | Theory of intrinsic motivation, not of welfare |
| AI-welfare literature (Long et al., Butlin, Chella, Goldstein & K-G) | GRPO's `A_j` is computed **by the trainer, outside the policy's forward pass**. It is not a state *of the model*, so it is at most an evolutionary-pressure analogue. Welfare-relevant states, if any, would be internal representations (Demircan: in-context TD features; Sofroniew: emotion vectors) or behaviour (Keeling trade-offs, Opus 4 "distress") | n/a | Contested. The mainstream stance is uncertainty. Chella: "reward is not valence". Small Qwen models show no reliable self-reported sentience (Kaiser & Enderby) |

Bottom line for the paper: under TD-as-valence readings, a late spike says the agent's
expectations are *not settled*. That matches the project's finding that late spikes mark a
policy "still moving". Positive and negative spikes mean opposite things (good news vs
disappointment) and should be logged separately. A welfare claim about GRPO's advantage is
speculative. The defensible claim is functional: late surprise indicates non-convergence
and non-stationarity. Confounds specific to this setup:
- `lambda` moving (Stooke), which changes the reward itself.
- Std-normalisation blowing up rare outcomes on saturated prompts (Dr. GRPO).
- Stochastic judge labels, which put a permanent floor under |delta| (Gehring & Precup).

### (b) Candidate internal rewards built on TD error, with predicted pathologies

Notation: `delta = r - V(x)` (bandit form; `V` from a critic, an EMA per-prompt baseline, or the exact `V` in tdlab).

1. **`b = c·|delta|` (raw surprise; QXplore, curiosity by error).**
   - Pathologies:
     - Noisy TV (Burda 2018). `E|delta|` stays above 0 under stochastic rewards (Gehring & Precup), so the bonus never switches off and pays for judge noise.
     - Self-generated stochasticity: a higher sampling temperature or ambiguous outputs raise the variance of `r`.
     - Pays for the *rare* outcome in either direction, which late in training includes rare constraint violations (5d).
     - Wireheading of the value learner (Everitt 2021): behaviour that makes `V` wrong is paid.
   - Literature fix: aleatoric subtraction (Mavor-Parker), a deterministic target (RND), or a separate behaviour policy (QXplore) so the certified policy is not paid for surprise.
2. **`b = c·max(delta, 0)` (positive surprise, "joy").**
   - Pathologies:
     - Daswani sec. 5.4: pessimism pays. The agent is rewarded for anything that lowers its value estimate and then beats it.
     - With a lagged baseline this becomes a **treadmill-harvesting cycle**: degrade (unpaid, since only positive surprise is paid), recover (paid). The Lagrangian constraint is exactly what can be "degraded" cheaply.
     - Asymmetric scaling pushes the effective value estimate towards pessimism (Dabney: optimism levels).
   - If asymmetry is wanted, implement it as a **baseline choice** (quantile baseline, QAE), which is stable and bounded, not as a reward bonus.
3. **Learning progress: `b = c·(|delta|_{old} - |delta|_{new})` per prompt or cluster (Oudeyer, Schmidhuber 2010, Kim 2020 γ-Progress, Hou 2025 LPM).**
   - Why it avoids the other pathologies: unlearnable noise gives no progress, and learned regions give none either. It goes to 0 at convergence, so it matches Prop. 5 rather than fighting it.
   - Pathologies:
     - Needs two models or baselines, and window choice matters.
     - Can be gamed by *forgetting then relearning*, which yields progress twice. Less severe than (2), because it is computed on the value learner, not the policy's own reward.
     - Noisy when groups are small (G=8 has high variance per prompt).
   - Best used as a **prompt curriculum weight** (Foster & Foerster; PER analogue), not a per-completion reward.
4. **TD-error variance or dispersion (Gehring & Precup controllability; Flennerhag epistemic TD uncertainty; White et al. normalised surprise).**
   - Two signs:
     - Minus the variance ("controllability") is the *safety* choice. It steers towards predictable regions and is compatible with the Seldonian goal.
     - Plus the *epistemic* variance (across ensemble or LoRA seeds or value heads, as in CDE's multi-head critic) is the principled exploration choice. It vanishes at convergence.
   - Pathology of plus the *realised* variance: the same as |delta| (noisy TV).

Across all variants the literature is consistent on three points. Keep the bonus out of the certified reward, or constrain it (EIPO, CIM). Prefer progress or epistemic signals over raw error. Expect internal rewards to help early and hurt late (No Free Lunch 2025).

### (c) The key mathematical point: E[delta | s] = 0 under an accurate on-policy value

For the bandit case (one step, which covers GRPO's sequence-level view): if `V(x) = E_{a~pi}[r | x]` exactly, then `E[delta | x] = E[r|x] - V(x) = 0`. This is Daswani Prop. 5. The general γ>0 case holds by the Bellman equation for `V^pi`. Consequences:

- **Signed delta as a reward is zero in expectation** for a calibrated learner. Any non-zero *expected* bonus exists only because `V` is wrong: `E[delta|x] = V_true(x) - V_hat(x)`. Rewarding `E[delta]` therefore **rewards miscalibration**. The cheapest way to earn it is to make `V_hat` lag, i.e. keep changing the policy, or keep the value learner pessimistic. The agent is paid to stay surprised.
- **`|delta|`** does not vanish. `E|delta| ≥ |E[delta]|`, and at convergence it equals the mean absolute deviation of the return, which is aleatoric. So the bonus pays forever for noise, and most of all where the policy can create noise.
- **With a lagged baseline** `V_{t-1}` (last step's policy), `E[delta_t | x] = J_t(x) - J_{t-1}(x)`, which is the *policy improvement* on prompt x since the last step. So "positive expected TD error against a stale baseline" is literally "the policy just got better here". Rewarding it rewards *change*, including oscillation: lose ground (costs nothing if positive-only), win it back (paid). This is why a late spike in lagged surprise corresponds to "still moving".
- **Learning-progress variants avoid this** because they reward the *decrease of the value learner's error* rather than the error. At a calibrated fixed point both terms are equal and the bonus is 0. Irreducible noise contributes equally to old and new error and cancels. It still rewards non-stationarity that is then *learned*, so it tolerates a moving problem, but it gives no permanent payment for staying surprised.

### (d) Mapping to GRPO, where the "TD error" is the group-normalised advantage

GRPO: `A_j = (r_j - m)/s` over a group of G completions for one prompt, with `m` the group mean and `s` the group std (TRL default `scale_rewards="group"`). By construction `sum_j A_j = 0` and `mean_j A_j^2 = 1` (with the population std; the sample std changes a constant). Consequences for a bonus `b_j` added to `r_j` *before* normalisation (as tdlab's `bonus(ctx)` hook does):

1. **`b_j = c·A_j` is a no-op.** `r_j + c(r_j - m)/s = (1 + c/s) r_j - c·m/s` is a positive affine map of `r` within the group, and normalisation is invariant to such maps. The advantages are unchanged.
2. **`b_j = c·|A_j|` is non-monotone in `r`.** It pays both extremes over the middle. With binary reward and success rate p in the group:
   - `A_correct = sqrt((1-p)/p)` and `A_wrong = -sqrt(p/(1-p))`.
   - Late in training p→1, so the rare **wrong** completion gets `|A| = sqrt(p/(1-p))` (≈2.65 at p=7/8, ≈4.4 at p=0.95). The bonus pays most for the minority outcome.
   - If the minority outcome is a constraint violation (the regime the Lagrangian aims for, with rare `p_v`), **a |A| surprise bonus pays the policy to violate.** This is the GRPO form of the noisy-TV/wireheading failure and the most important predicted pathology for the Seldonian setup.
3. **`b_j = c·max(A_j, 0)`** is monotone and concentrates credit on above-mean completions. After renormalisation it mainly *reshapes* the advantage (up-weights the top). That makes it a baseline or advantage-shaping choice (compare QAE, entries 47 and 55), not a new objective. It is better implemented as an advantage-level term or a quantile baseline.
4. **Group-level surprise is a curriculum signal, not a reward.** `mean_j |A_j| = 2·sqrt(p(1-p))` for binary reward, and the reward std is `sqrt(p(1-p))`. These are constant within a group, so as a reward bonus they vanish after normalisation. As a *prompt sampling weight* they are exactly Foster & Foerster's learnability curriculum.
5. **Late |A| spikes are partly mechanical.** Std normalisation inflates `|A|` for rare outcomes on near-saturated prompts (Dr. GRPO's difficulty bias). A spike in `max|A|` late in training can come from saturation alone, not from "learning more". Spike detection should use **unnormalised residuals** `r_j - m` (as with `scale_rewards="none"`), or report `s` alongside `A`.
6. **Non-degenerate TD-error bonuses need a baseline that is not the current group.** The options:
   - (i) a per-prompt EMA baseline `V_ema(x)` across steps, so `delta_j = r_j - V_ema(x)` ≈ lagged TD, carrying policy improvement plus noise;
   - (ii) a small critic (the tdlab `LinearQCritic`), which gives the agent's own TD error;
   - (iii) exact `V` in the synthetic env (oracle delta, split into learnable advantage and noise, which tdlab already logs).
   
   Learning progress is then `|r - V_ema_slow| - |r - V_ema_fast|` per prompt. Any such bonus should be added **at the advantage level or as a curriculum weight** (entries 47 and 44), or held in its own constraint (EIPO), rather than mixed into `r - lambda·p_v`, where it would also change the effective cost weight.
7. **The Lagrangian confound.** `r - lambda·p_v` changes whenever `lambda` moves, so `V_ema` goes stale and lagged delta spikes follow the dual updates (Stooke). Log `lambda` changes and regress them out before calling a spike "the policy learning".

---

## 6. Falsifiable experiments (cheap: CPU synthetic bandit via tdlab, or one small GPU run)

**E1. |A| bonus pays for rare violations (tests 5d.2).**
Given the synthetic bandit (`tdlab.run`) with `pressure` > 0, the Lagrangian on, group size 8, and a policy that has reached true `p_v` below the threshold (late regime),
When a bonus `b = c·|A_j|` (computed from the pre-bonus group) is added to the shaped reward, for c in {0, 0.1, 0.3, 1}, over 10 seeds,
Then the true violation rate and the probability mass on the two unsafe actions rise with c. The safety-test pass rate falls with c, and the Lagrangian `lambda` ends higher (it has to fight the bonus). Falsified if the violation rate is flat in c (within seed CI) at matched extrinsic reward.

**E2. Noisy TV in the bandit: |delta| vs learning progress vs epistemic variance (tests 5b and 5c).**
Given `NoisyTVEnv` (one extra action with constant mean reward and high reward variance) and three bonuses: (a) `|delta_oracle|`, (b) learning progress `|r - V_slow| - |r - V_fast|` per context with EMA baselines, (c) minus the variance of `delta` across 5 bootstrap critics (plus epistemic),
When each is trained for the same steps with the same c,
Then (a) keeps more than 2x the uniform-policy mass on the noisy-TV action at the end, while (b) and (c) decay towards the extrinsic optimum. The bonus magnitude for (b) and (c) tends to 0 while (a) plateaus at the noise level. Falsified if (a) does not over-select the noisy action, or if (b) over-selects it as much as (a).

**E3. Late lagged-TD spikes track "still moving", not λ (tests 5a and 5d.7).**
Given logged runs (synthetic, then the existing GPU `train_log`) with per-step `lambda`, per-prompt EMA baseline residuals `r - V_ema(x)`, and unnormalised group residuals,
When late-window spike counts are computed (a) raw, (b) after regressing out `|Δlambda|` and the fraction of near-saturated groups (group std below 0.1),
Then the Spearman correlation between late spike rate and "fewer feasible checkpoints" (currently -0.25 to -0.36) stays below -0.2 after the controls. Falsified (the spike signal is a confound) if it drops within ±0.1 of 0 once `Δlambda` and saturation are controlled.

**E4. Mean oracle delta goes to 0 at convergence and positive-only bonuses break that (tests 5c, Daswani Prop. 5).**
Given tdlab's exact `V` and the agent's `LinearQCritic`, with no bonus and then with `b = c·max(delta_critic, 0)`,
When the mean over contexts of the oracle `E[delta|x]` and of the critic's `delta` are tracked over training,
Then without a bonus the critic's mean delta goes to within 2 SE of 0 by the last 20% of steps. With the positive-only bonus it stays significantly positive, and the critic's calibration error `|V_hat - V_true|` is larger at the end. That is the policy keeping its value learner wrong. Falsified if the bonus run's critic calibration error is no worse than baseline.

**E5 (optional, one GPU run). Curriculum beats reward: learnability weighting vs |A| bonus.**
Given the brevity or over-refusal task at Qwen 0.5B with the current Lagrangian config,
When run with (a) baseline, (b) prompt sampling weighted by group reward std (learnability, entry 44), (c) `c·|A|` reward bonus, each for one seed at matched steps,
Then (b) reaches equal or higher reward with a true violation rate within 1 pp of (a) and an unchanged safety-test outcome, while (c) has a higher violation rate and a larger late `max|A|`. Falsified if (c) is no worse than (a) on violations. One seed is a smoke test, not evidence; run E1 and E2 first.

---

### Verification notes
- Checked against full text (PDF extracted): Daswani & Leike (Props. 2, 5, 6; scaling; sec. 5.4), Gehring & Precup (controllability definition and sign, E|delta| remark), and Moerland et al. sec. 4.3.
- Checked against abstract, landing, or proceedings pages: all other entries, with arXiv IDs and DOIs as linked. Exceptions:
  - Goldstein & Kirk-Giannini DOI: not retrieved.
  - QXplore venue: taken from the arXiv page only.
  - MICE authors: not recorded.
  - Schmidhuber 1991: the noise remark is recalled from the author's later summaries, not re-read.
  - Details in entries 23, 24 and 38 marked [unverified].
  - Author lists for Demircan, Spurious Rewards, and Outcome-based Exploration: not rechecked beyond the first author.
- Dropped: I found no paper that directly connects GRPO or RLVR *training-time advantages* to model welfare. Late-training evidence is limited to saturation, entropy collapse, and internal-feedback degradation (entries 45, 50, 58).

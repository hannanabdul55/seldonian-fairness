# RL and game-theoretic foundations of alternating attacker/victim training

Scope: adversarial policies in deep RL, robust adversarial RL, iterated adversarial training and self-play, and the theory of alternating minimax optimisation. Catalogue of 38 paper records. Compiled 2026-10-17.

Verification key used throughout:
- **VERIFIED** = metadata (title, first author, year, venue, arXiv ID/DOI) confirmed from the arXiv API or the publisher page during this research; quantitative claims confirmed from the fetched full text or abstract.
- **UNVERIFIED (recalled)** = a detail I believe to be correct from prior knowledge but could not confirm from a fetched page in this session. Treat as a pointer to check, not as a citation.
- Items tagged **RECENT** are the 2024-2026 results the report should foreground.

Note on the goattack.far.ai project page: fetching it returned only the title "Adversarial policies in Go" with no body text (the page appears to be JavaScript-rendered), so all Go numbers below come from the arXiv full texts and the FAR AI blog instead — [goattack.far.ai](https://goattack.far.ai/).

Structured fields used in every record: (a) how the adversary is trained; (b) shared architecture/initialisation; (c) alternation and number of iterations; (d) main finding on convergence, cycling or robustness transfer; (e) code.

---

## Key question 1: The adversarial-policies line (Gleave 2020 -> Wu 2021 -> Guo 2021 -> Wang 2023 -> Tseng 2025 -> Liu 2024 -> 2025 follow-ups)

### Takeaway
This literature already contains the full loop the LM version proposes: train an RL adversary against a frozen victim, fine-tune the victim against it, train a new adversary, repeat. Every published run of that loop ends the same way: each round blocks the previous adversary, and a freshly trained adversary beats the new victim again, at a small fraction of the defender's compute; the only published theoretical fix is timescale separation (adversary fast, victim slow), which gives average-iterate convergence to the least-exploitable victim in tabular settings.

### Cited Findings

#### R1. Adversarial Policies: Attacking Deep Reinforcement Learning — Gleave, 2020, ICLR 2020, arXiv:1905.10615 (VERIFIED)
- Summary: Demonstrates that in zero-sum MuJoCo games between simulated humanoids (You Shall Not Pass, Kick and Defend, Sumo Humans, Sumo Ants) an opponent trained by RL against a frozen self-play victim can beat it while "appearing to move randomly"; attacks work better in higher-dimensional observation spaces and the adversary induces off-distribution victim activations. The paper also runs the first two rounds of an attacker/victim arms race and reports that the fine-tuned victim is beaten by a newly trained adversary — [arXiv 1905.10615](https://arxiv.org/abs/1905.10615).
- (a) Adversary: PPO (Stable Baselines PPO2), 20 million timesteps, sparse reward (+ for win at episode end, negative for loss or tie); batch 16,384, 8 envs, lr 3e-4, gamma 0.99, GAE 0.95, clip 0.2, entropy coefficient 0.0. The adversary was trained for under 3% of the victims' original timesteps — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3).
- (b) Shared architecture: not stated in the text; victims (from Bansal et al. 2018) are LSTM in Kick and Defend/Sumo and MLP in You Shall Not Pass — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3).
- (c) Alternation: two rounds. Victim fine-tuned against Adv1 either "single" (adversary only) or "dual" (random choice of adversary or normal Zoo opponent per episode); then a new adversary (AdvS1 / AdvD1) trained against each fine-tuned victim — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3).
- (d) Findings: Adv1's win rate vs ZooV1 was 87%, dropping to about 10% against both fine-tuned victims; the single-trained victim "catastrophically forgets" how to beat the normal opponent; the dual-trained victim holds the normal opponent to 57% wins. New adversaries AdvS1/AdvD1 win "at equal or greater rates" than before and transfer back to the original victim, using a qualitatively different exploit (lying prone) — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3). The masked victim (opponent position fixed to a constant) wins 99% vs Adv1 where the unmasked victim loses 86%, i.e. the attack works entirely through the observation channel — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3). The adversary wins less in Sumo Ants (opponent dim 15) than Sumo Humans (dim 24), the basis of the "higher-dimensional is more attackable" claim (two environments only) — [arXiv HTML v3](https://arxiv.org/html/1905.10615v3).
- (e) Code: github.com/HumanCompatibleAI/adversarial-policies — UNVERIFIED (recalled URL).

#### R2. Adversarial Policy Training against Deep Reinforcement Learning — Wu (Xian Wu, Wenbo Guo, Hua Wei, Xinyu Xing), 2021, 30th USENIX Security Symposium, pp. 1883-1900 (VERIFIED; no arXiv version found)
- Summary: Extends Gleave et al.'s PPO attacker with an explanation-guided "action deviation" term: the adversary is rewarded for taking actions that most change the victim's action, with the victim's policy approximated in a black-box setting. Reports higher win rates and faster, lower-variance training than the Gleave baseline, and a one-round retraining defence that works against the adversary but costs generalisation — [USENIX page](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian).
- (a) Adversary: PPO plus an action-deviation objective weighted by a hyperparameter lambda that is adapted over time using an explanation (feature-importance) method; black-box approximation of the victim network performs "approximately the same" as the white-box version — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf).
- (b) Shared architecture: not a design feature; the victim network is approximated from observations.
- (c) Alternation: one retraining round. "Experiment V": victim retrained with the same Algorithm 1 on episodes against the adversary — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf).
- (d) Findings: adversary reaches an average 60% vs 50% win rate (MuJoCo) and 100% vs 90% (roboschool Pong) relative to the Gleave baseline, and reaches 50% in 11M vs 20M iterations — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf). The retrained victim wins more than 95% vs the adversary in both games, but against the regular self-play opponent wins only 16.3% mean (MuJoCo, range 6-25%) and 41.4% (Pong), so "a simple adversarial training improves agent robustness, it cannot help a victim agent obtain sufficient generalizability"; the authors suggest mixing regular and adversarial episodes — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf). No second adversary was trained against the retrained victim.
- (e) Code: https://github.com/psuwuxian/rl_attack (stated in the paper) — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf).

#### R3. Adversarial Policy Learning in Two-player Competitive Games — Guo (Wenbo Guo, Xian Wu, Sui Huang, Xinyu Xing), 2021, ICML 2021, PMLR 139:3910-3919 (VERIFIED; no arXiv version found)
- Summary: Argues prior adversarial-policy attacks depend on the game being zero-sum, and proposes a new objective and surrogate loss for the attacker that also works in non-zero-sum two-player games; reports greater exploitability than the prior attack, the ability to exploit unfairness in a game, and that victims retrained against these adversaries become more resistant — [PMLR](https://proceedings.mlr.press/v139/guo21b.html); [ICML poster](https://icml.cc/virtual/2021/poster/9779).
- (a) Adversary: policy-gradient with a modified optimisation goal and surrogate objective (details not fetched).
- (b) Shared architecture: not reported.
- (c) Alternation: one retraining round (retrained victims reported as more resistant) — [PMLR](https://proceedings.mlr.press/v139/guo21b.html).
- (d) Finding: the non-zero-sum formulation matters because the attacker's gain is not the victim's loss; this is the earliest paper in this line to drop the zero-sum assumption. Number of iterations and transfer to other adversaries: not reported in the abstract.
- (e) Code: UNVERIFIED (not found on the fetched pages).

#### R4. Adversarial Policies Beat Superhuman Go AIs — Wang (Tony T. Wang, Adam Gleave, Tom Tseng, ..., Stuart Russell), 2023, ICML 2023, arXiv:2211.00241 (VERIFIED)
- Summary: Trains adversaries against KataGo using "victim-play" (the adversary trains only on its own moves against a frozen victim) and Adversarial MCTS (the search models the victim's moves with the victim network). The cyclic adversary wins >97% against KataGo at superhuman settings; the attack transfers zero-shot to other Go AIs and is executable by humans; KataGo's own adversarial training against the published positions did not remove the vulnerability — [arXiv 2211.00241](https://arxiv.org/abs/2211.00241).
- (a) Adversary: victim-play with A-MCTS-S (samples the victim's policy head for victim moves); curriculum over later victim checkpoints and increasing victim visit counts — [arXiv HTML v4](https://arxiv.org/html/2211.00241v4).
- (b) Shared architecture/initialisation: the adversary network is randomly initialised, not copied from KataGo, because the threat model is grey-box (the victim network is queried but not copied) — [arXiv HTML v4](https://arxiv.org/html/2211.00241v4).
- (c) Alternation: one defence round within this paper. KataGo's authors added adversarial positions to self-play (0.08% of games); the cyclic adversary's win rate fell to 0/50 against b60-s7702m at 32 visits; after about 1,155 further V100 GPU-days of adversary fine-tuning it won 47% of 400 games at 4,096 visits and 17.5% of 40 games at 100,000 visits — [arXiv HTML v4](https://arxiv.org/html/2211.00241v4).
- (d) Findings: cyclic adversary 97.3% over 1,000 games at 4,096 visits and 72% over 50 games at 10^7 visits; adversary compute 2,223 V100 GPU-days, about 14% of the victim's; pass-adversary 20.4 GPU-days (0.13%) wins 99.9% vs no-search KataGo but 0% vs the pass-hardened version. Zero-shot transfer: 6.1% vs Leela Zero and 3.5% vs ELF OpenGo; the adversaries "lose to most amateur Go players" — [arXiv HTML v4](https://arxiv.org/html/2211.00241v4).
- (e) Code: https://github.com/AlignmentResearch/go_attack/ — [arXiv HTML v4](https://arxiv.org/html/2211.00241v4).

#### R5. Can Go AIs be adversarially robust? — Tseng (Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave), 2025, AAAI 2025 (vol. 39 no. 26 pp. 27662-27670), arXiv:2406.12843, DOI 10.1609/aaai.v39i26.34980 (VERIFIED) — RECENT, most important item
- Summary: Tests three defences for KataGo: positional adversarial training (KataGo's own curriculum on hand-constructed cyclic positions), iterated adversarial training (nine attacker-then-victim rounds), and a vision-transformer backbone. "Though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries"; the effective new attacks are mostly "different realizations of the same overall class of cyclic attacks" — [arXiv 2406.12843](https://arxiv.org/abs/2406.12843); [AAAI DOI](https://doi.org/10.1609/aaai.v39i26.34980).
- (a) Adversary: victim-play with A-MCTS as in R4; each adversary a_n fine-tuned from a_{n-1}; stopping rule = adversary wins most games at a threshold visit count or hits a compute budget — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3).
- (b) Shared architecture: adversaries are separate networks (randomly initialised lineage from R4); the ViT experiment changes only the victim's architecture — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3).
- (c) Alternation: **9 iterations**. Victim v_n fine-tuned from v_{n-1} with 18% of games against a frozen a_{n-1} and the rest self-play; victim training stopped when its win rate plateaued — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3).
- (d) Findings. Compute: final victim v9 cost 28,296 V100 GPU-days (12,097M steps, cumulative over all rounds); final adversary a9 cost 7,337 V100 GPU-days; attack side used "18x more compute" than the defence side in aggregate, yet each fresh attack is cheap relative to the victim. Per-round: each v_n beats a_{n-1} at >= 16 visits and ">95% of the time" at 256 visits, but a9 wins 59% vs v9 at 4,096 victim visits (with 26% of v9's compute) and 42% at 65,536 visits. Asymmetry: victims beat all previous adversaries, adversaries beat all previous victims; a9 beats every v_n even at high visits. Fresh held-out adversaries trained from an earlier base: atari-adversary 81% vs v9 at 512 visits with 5% of v9's compute (but 4% at 4,096 visits, and 13% without search); stall-adversary 91% vs v9 at 512 visits with 9% of compute. Adversaries a5, a6, a8 were weak against their own victims, attributed to local minima in adversary training, not to a more robust victim. Authors' conclusion: iterated training gives "partial protection", the attack method "is always able to find a successful attack", and the offence-defence balance favours attackers; the gap is not reported to narrow across rounds, only that defences "rarely reached a 100% win rate" — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3). Positional adversarial training: gift-adversary 91% vs dec23-victim at 512 visits with 6% of compute; continuous-adversary 65% at 4,096 and 27% at 65,536 visits with 13%; big-adversary 75%/56% vs may24-victim at 4,096/65,536 visits with 9%; KataGo added gift/continuous positions in Feb 2024 and big-adversary still succeeded — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3). ViT: ViT-victim (563 GPU-days) loses 78% at 65,536 visits to ViT-adversary and 2.5% at 512 visits to the original base-adversary zero-shot, so CNN architecture is not the cause — [arXiv HTML v3](https://arxiv.org/html/2406.12843v3). FAR AI's blog adds that KataGo's developers have run positional adversarial training since the December 2022 disclosure and that FAR "was readily able to train a new adversary to beat the latest KataGo model as of December 2023" — [FAR AI blog](https://www.far.ai/blog/beyond-the-board-exploring-ai-robustness-through-go).
- (e) Code: same go_attack repository as R4 (UNVERIFIED that the AAAI version adds a separate repo).

#### R6. Rethinking Adversarial Policies: A Generalized Attack Formulation and Provable Defense in RL — Liu (Xiangyu Liu, Souradip Chakraborty, Yanchao Sun, Furong Huang), 2024, ICLR 2024, arXiv:2305.17342 (VERIFIED)
- Summary: Generalises adversarial policies to partial control of the attacking agent (mixing weight epsilon with a reference policy) so the attack can be stealthier; proposes adversarial training with **timescale separation** and proves polynomial convergence (on average) to the most robust victim; shows on RoboSumo that single-timescale alternation oscillates while the two-timescale victim converges and is less exploitable by a fresh PPO best-response — [arXiv 2305.17342](https://arxiv.org/abs/2305.17342).
- (a) Adversary: zero-sum, minimises the victim's return V_rho; policy constrained to (1-eps)*pi_hat + eps*pi_alpha; in Kuhn Poker the min oracle is game-tree search; in RoboSumo PPO serves as the best-response attacker for evaluation — [arXiv HTML v3](https://arxiv.org/html/2305.17342v3).
- (b) Shared architecture: not a design variable.
- (c) Alternation: Algorithm 1 = exact best-response oracle for the attacker each round plus one victim gradient step; Algorithm 2 = simultaneous projected gradient steps with eta_nu << eta_alpha so "the attacker effectively best-responds while the victim adapts slowly"; output is a victim iterate sampled proportionally to eta_nu^t — [arXiv HTML v3](https://arxiv.org/html/2305.17342v3).
- (d) Theory: Theorem 5.3, direct (tabular) parameterisation with bounded minimax mismatch coefficient C: average exploitability of victim iterates <= min exploitability + delta with eta_nu ~ delta^8, eta_alpha ~ delta^4 and T = poly(1/delta, C, |S|, |A|, 1/(1-gamma)); last-iterate and stochastic-gradient extensions are stated as possible but not proved. Appendix C (Rock-Paper-Scissors): simultaneous and alternating GDA and simultaneous/alternating iterated best response (SGDA, AGDA, SIBR, AIBR) all "fail to achieve low exploitability", only the timescale-separated method reaches near zero; the paper notes retraining against one fixed attacker overfits and degrades performance against other attackers, citing Gleave et al. Empirically (Fig. 4) the two-timescale victim's score converges while the single-timescale one oscillates; no numerical win rates or iteration counts are given in the text — [arXiv HTML v3](https://arxiv.org/html/2305.17342v3).
- (e) Code: github.com/xiangyu-liu/Rethinking-Adversarial-Policies-in-RL — [arXiv HTML v3](https://arxiv.org/html/2305.17342v3).

#### R7. Neutral Agent-based Adversarial Policy Learning against Deep Reinforcement Learning in Multi-party Open Systems — Peng (Qizhou Peng et al., 5 authors), 2025, arXiv:2510.10937 (VERIFIED metadata only) — RECENT
- Summary: A 2025 extension of the adversarial-policies threat model to multi-party open systems in which the attacker controls a "neutral" third agent rather than the direct opponent; the abstract frames it against prior attacks that manipulate the environment or the victim's observations directly — [arXiv 2510.10937](https://arxiv.org/abs/2510.10937). Fields (a)-(e): not extracted (abstract truncated in the API response; full text not fetched). Listed as a pointer that the line is still active in late 2025; it does not report iterated training as far as the abstract shows.

#### R8. On the Tension Between Optimality and Adversarial Robustness in Policy Optimization — Li (Haoran Li et al., 8 authors), 2025, arXiv:2512.01228 (VERIFIED metadata only) — RECENT
- Summary: Compares standard policy optimisation with adversarially robust policy optimisation and argues there is a gap between theory (which suggests the two goals can align) and practice — [arXiv 2512.01228](https://arxiv.org/abs/2512.01228). Fields (a)-(e): not extracted. Pointer only; relevant to the non-zero-sum "helpfulness vs safety" framing, not to iterated training.

#### Pointer: Robust Deep Reinforcement Learning Through Adversarial Attacks and Training: A Survey — Schott (Lucas Schott et al., 8 authors), 2024, arXiv:2403.00420 (VERIFIED metadata), 83 pages, 15 algorithms; useful as a secondary index of the attack/defence pairs above — [arXiv 2403.00420](https://arxiv.org/abs/2403.00420).

### Inferences
- The empirical pattern across R1, R2, R4 and R5 is consistent: one round of victim fine-tuning drives the specific attacker's win rate to roughly 0-10%, and a fresh attacker recovers a majority win rate at 3-26% of the victim's compute. Nothing in these papers shows the per-round attacker cost rising fast enough to make the loop converge; R5 explicitly declines to claim that the gap narrows.
- The two defence failure modes reported are (i) catastrophic forgetting / loss of generality when the victim trains only against the adversary (R1 single, R2) and (ii) over-specific robustness when the adversary is fixed (R1, R6's remark). Mixing normal and adversarial opponents (R1 dual, R5's 18% adversary games, R2's suggestion) mitigates (i) but not (ii).
- R6 is the only paper in this line with a convergence proof, and its proof is for average iterates under two timescales in tabular settings; its own appendix shows naive alternation fails on Rock-Paper-Scissors. This is the most direct theoretical statement that "alternate attacker and victim updates" is not, in itself, a convergent procedure.

### Gaps
- I found no 2025-2026 paper that extends the KataGo iterated-training experiment beyond the nine rounds in R5 (searches for citing work returned only R5 itself and the FAR AI blog). Any claim about "what happens after many more rounds" must rest on R5's nine.
- R1 does not report AdvS1/AdvD1's exact win rates in the text (figure only); R6 reports no numerical RoboSumo win rates.
- R3's and R7's training details were not extracted (no arXiv text for R3; R7 full text not fetched).
- Whether R2's or R3's retrained victims resist a second, freshly trained adversary is not reported in either paper.

---

## Key question 2: Robust adversarial RL (RARL, action-robust RL, SA-MDP/ATLA, adversarial populations, PAIRED, Robust PLR, ACCEL)

### Takeaway
Robust RL has trained RL adversaries in alternation with a protagonist since 2017; the recurring lessons are that a single learned adversary overfits (motivating populations and regret objectives), that the protagonist/adversary game is formulated zero-sum for tractability, and that convergence claims are either absent (RARL, ATLA) or hold only for the game's equilibrium rather than for the alternating dynamics (PAIRED's minimax-regret Nash).

### Cited Findings

#### R9. Robust Adversarial Reinforcement Learning — Pinto (Lerrel Pinto, James Davidson, Rahul Sukthankar, Abhinav Gupta), 2017, arXiv:1703.02702 (VERIFIED metadata; venue ICML 2017 UNVERIFIED-recalled)
- Summary: Formulates robust control as a two-player zero-sum game in which an adversary applies destabilising forces; "We formulate the policy learning as a zero-sum, minimax objective function"; evaluated on InvertedPendulum, HalfCheetah, Swimmer, Hopper, Walker2d with claims of improved training stability and robustness to train/test differences (mass, friction) — [arXiv 1703.02702](https://arxiv.org/abs/1703.02702).
- (a) Adversary: TRPO with reward = negative of the protagonist's reward — UNVERIFIED (recalled; abstract does not state the algorithm).
- (b) Separate networks for protagonist and adversary — UNVERIFIED (recalled).
- (c) Alternation: N_mu protagonist iterations then N_nu adversary iterations per outer loop, repeated for a fixed number of outer iterations — UNVERIFIED (recalled schedule; the abstract does not give it).
- (d) No convergence proof; empirical claim of better stability and robustness to test-time dynamics changes — [arXiv 1703.02702](https://arxiv.org/abs/1703.02702).
- (e) Code: github.com/lerrel/rllab-adv — UNVERIFIED (recalled).

#### R10. Action Robust Reinforcement Learning and Applications in Continuous Control — Tessler (Chen Tessler, Yonathan Efroni, Shie Mannor), 2019, ICML 2019, arXiv:1901.09184 (VERIFIED)
- Summary: Defines two action-robustness criteria: with probability alpha an adversarial action replaces the agent's (probabilistic action robust MDP) or an adversary perturbs the chosen continuous action (noisy action robust MDP); shows the criteria relate to model uncertainty and gives solution methods — [arXiv 1901.09184](https://arxiv.org/abs/1901.09184).
- (a) Adversary: policy-gradient player in a zero-sum game over actions (details UNVERIFIED-recalled).
- (c) Alternation: the paper's DDPG-based algorithm updates agent and adversary in alternation — UNVERIFIED (recalled).
- (d) Finding: existence of optimal robust policies for the two criteria and a convergent policy-iteration scheme in the tabular case — UNVERIFIED (recalled beyond the abstract).
- (e) Code: UNVERIFIED.

#### R11. Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations — Zhang (Huan Zhang, Hongge Chen, ..., Cho-Jui Hsieh), 2020, NeurIPS 2020 (33:21024-21037), arXiv:2003.08938 (VERIFIED)
- Summary: Introduces the state-adversarial MDP (SA-MDP), shows naive adversarial training can fail, and proposes a policy-regularisation defence (SA-PPO/DDPG/DQN) plus a stronger attack — [arXiv 2003.08938](https://arxiv.org/abs/2003.08938).
- (a) Adversary: gradient-based observation perturbation within an l_inf ball (not a learned RL adversary in this paper) — abstract-level; exact attack UNVERIFIED.
- (c) No alternation of a learned adversary; regulariser-based.
- (d) Finding: "naively applying existing techniques ... can fail"; the SA-MDP optimal policy may not exist in general (UNVERIFIED-recalled theorem). Serves as the theoretical setup for ATLA.

#### R12. Robust Reinforcement Learning on State Observations with Learned Optimal Adversary (ATLA) — Zhang (Huan Zhang, Hongge Chen, Duane Boning, Cho-Jui Hsieh), 2021, ICLR 2021, arXiv:2101.08452 (VERIFIED)
- Summary: Shows that for a fixed agent policy an optimal observation-perturbing adversary exists and attains the worst-case reward; learns it with RL as a stronger attack; proposes ATLA, "which trains an adversary online together with the agent using policy gradient"; finds LSTM policies more robust and reports state-of-the-art robustness under strong adversaries on continuous-control tasks — [arXiv 2101.08452](https://arxiv.org/abs/2101.08452).
- (a) Adversary: policy gradient (PPO) in the SA-MDP, reward = negative agent reward — the "negative reward" detail is UNVERIFIED (recalled); the abstract confirms "policy gradient".
- (b) Separate networks; the adversary's action space is the perturbation.
- (c) Alternation: agent and adversary trained "online together"; the exact schedule and iteration count are not on the abstract page — [arXiv 2101.08452](https://arxiv.org/abs/2101.08452).
- (d) No convergence guarantee for ATLA's alternating training is stated; the only theoretical claim is existence of the optimal adversary for a fixed policy — [arXiv 2101.08452](https://arxiv.org/abs/2101.08452).
- (e) Code: github.com/huanzhang12/ATLA_robust_RL — UNVERIFIED (recalled).

#### R13. Robust Reinforcement Learning using Adversarial Populations — Vinitsky (Eugene Vinitsky et al., 6 authors), 2020, arXiv:2008.01825 (VERIFIED metadata; no venue listed)
- Summary: Notes that existing robust-RL work "primarily focused on training a single RL agent against a single adversary" and that this overfits; trains the protagonist against a population of adversaries, improving robustness and transfer — [arXiv 2008.01825](https://arxiv.org/abs/2008.01825).
- (a) Adversaries: RL (PPO; zero-sum reward) — algorithm UNVERIFIED (recalled); population setting confirmed by abstract.
- (c) Alternation: protagonist and adversary population updated jointly across iterations (UNVERIFIED-recalled schedule).
- (d) Finding (abstract): a single adversary leads to overfitting; a population yields more robust policies — [arXiv 2008.01825](https://arxiv.org/abs/2008.01825). Directly relevant to "does training against one learned adversary transfer to others": the paper's premise is that it does not.
- (e) Code: UNVERIFIED.

#### R14. Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED) — Dennis (Michael Dennis, Natasha Jaques, Eugene Vinitsky, ..., Sergey Levine), 2020, arXiv:2012.02096 (VERIFIED metadata; venue NeurIPS 2020 UNVERIFIED-recalled)
- Summary: Proposes UED; argues "minimax adversarial training often yields unsolvable worst-case environments"; PAIRED adds an antagonist "allied with the environment-generating adversary" so the adversary is "motivated to generate environments which maximize regret", defined as the antagonist's return minus the protagonist's; produces a curriculum and higher zero-shot transfer to novel environments — [arXiv 2012.02096](https://arxiv.org/abs/2012.02096).
- (a) Adversary: RL (PPO) with regret as reward — the PPO detail is UNVERIFIED (recalled); regret reward confirmed.
- (b) Protagonist and antagonist share architecture but not parameters — UNVERIFIED (recalled).
- (c) Alternation: all three agents updated each iteration (UNVERIFIED-recalled).
- (d) Theory: at a Nash equilibrium of the regret game the protagonist plays a minimax-regret policy — UNVERIFIED (recalled; not on the abstract page). The key framing: pure minimax (zero-sum) adversaries produce degenerate "unwinnable" tasks, which is why regret replaces win/loss as the adversary's reward — [arXiv 2012.02096](https://arxiv.org/abs/2012.02096).
- (e) Code: google-research/social_rl — UNVERIFIED (recalled).

#### R15. Replay-Guided Adversarial Environment Design (Robust PLR) — Jiang (Minqi Jiang et al., 6 authors), 2021, NeurIPS 2021, arXiv:2110.02439 (VERIFIED)
- Summary: Casts Prioritized Level Replay as UED; the "adversary" becomes a curator that replays high-regret levels rather than a trained RL generator; Robust PLR trains the agent only on replayed levels — [arXiv 2110.02439](https://arxiv.org/abs/2110.02439).
- (a) No learned RL adversary; regret estimates (e.g. positive value loss) score random levels — UNVERIFIED (recalled scoring detail).
- (c) No adversary training to alternate; the agent alternates between evaluating new levels and training on replayed ones.
- (d) Finding: inherits PAIRED's minimax-regret equilibrium guarantee while outperforming the learned adversary empirically — UNVERIFIED (recalled beyond abstract). Relevant lesson: a non-learned, replay-based adversary avoided the instability of a learned one.
- (e) Code: facebookresearch/dcd — UNVERIFIED (recalled).

#### R16. Evolving Curricula with Regret-Based Environment Design (ACCEL) — Parker-Holder (Jack Parker-Holder et al., 7 authors), 2022, arXiv:2203.01302 (VERIFIED metadata; venue ICML 2022 UNVERIFIED-recalled)
- Summary: Frames environment design "as a game between a student and a teacher, using regret-based objectives"; ACCEL evolves levels by mutating high-regret ones from a replay buffer, combining an evolutionary generator with PLR's curation — [arXiv 2203.01302](https://arxiv.org/abs/2203.01302).
- (a) Adversary: evolutionary edits scored by regret, no RL adversary.
- (c) Student training alternates with teacher mutation/curation every iteration (UNVERIFIED-recalled).
- (d) Finding: regret-based teacher gives "theoretical guarantees at equilibrium" (abstract) and strong zero-shot transfer; no cycling reported.
- (e) Code: facebookresearch/dcd — UNVERIFIED (recalled).

### Inferences
- The robust-RL line moved from learned RL adversaries with zero-sum rewards (R9, R12) toward (i) populations of adversaries (R13), (ii) regret rather than win/loss as the adversary's reward (R14-R16), and (iii) non-learned curators (R15-R16). Each move was motivated by a failure of the single learned adversary: overfitting to it, or an adversary that wins trivially by making the task impossible.
- None of R9-R16 provides a convergence result for the alternating dynamics themselves; the guarantees that exist (R14-R16) are statements about the game's equilibrium, reached or not.

### Gaps
- Exact alternation schedules and iteration counts for R9, R10, R12, R13 were not extracted (abstract pages only); the full texts should be consulted before quoting them.
- I did not verify venue strings for R9, R14 and R16 from a publisher page.

---

## Key question 3: Self-play and population methods (FSP/NFSP, PSRO, open-ended learning, spinning tops, AlphaStar league, Bansal competition, AlphaZero)

### Takeaway
Game-theoretic self-play treats "train an exploiter against the current agent, then train the agent against the exploiter" as the standard oracle step of fictitious play/PSRO; its theory says the population's meta-strategy converges, not the latest agent, and the spinning-top analysis says a single agent chasing its latest exploiter cycles unless the population is large enough to cover the non-transitive layer.

### Cited Findings

#### R17. Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP) — Heinrich (Johannes Heinrich, David Silver), 2016, arXiv:1603.01121 (VERIFIED metadata; no venue in record)
- Summary: "combines fictitious self-play with deep reinforcement learning" to learn approximate Nash equilibria end-to-end; on Leduc poker NFSP approaches Nash where common RL methods diverge — [arXiv 1603.01121](https://arxiv.org/abs/1603.01121).
- (a) The best-response (exploiter) network is trained with DQN against the opponent's average policy; the average policy is learned by supervised learning on the agent's own best-response actions — UNVERIFIED (recalled mechanics beyond abstract).
- (c) Both networks updated continuously (anticipatory dynamics); no discrete rounds.
- (d) Finding: fictitious play converges in two-player zero-sum games to the *average* strategy, not the latest best response; the latest best response is by construction exploitable (recalled standard result). Key message for the LM loop: the "defender" that converges is the time-average policy, not the last iterate.
- (e) Code: UNVERIFIED.

#### R18. A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO) — Lanctot (Marc Lanctot et al., 8 authors), 2017, NIPS 2017, arXiv:1711.00832 (VERIFIED)
- Summary: Observes that independent RL "can overfit to the other agents' policies during training, failing to sufficiently generalize"; introduces Policy-Space Response Oracles, which alternate (i) computing a meta-strategy over a population and (ii) adding an RL best response (oracle) to that meta-strategy — [arXiv 1711.00832](https://arxiv.org/abs/1711.00832).
- (a) Oracle = any RL algorithm trained against a mixture of existing policies; reward is the game's reward — abstract confirms the overfitting motivation; oracle details UNVERIFIED-recalled.
- (c) Alternation: one new policy per epoch for both players; number of epochs is a hyperparameter (tens in the paper, UNVERIFIED).
- (d) Finding: generalises fictitious play, double oracle and independent RL by choice of meta-solver; the meta-strategy (mixture) converges, the newest policy does not; "joint policy correlation" quantifies overfitting between independently trained agents — UNVERIFIED (recalled beyond abstract).
- (e) Code: in OpenSpiel — UNVERIFIED (recalled).

#### R19. Open-ended Learning in Symmetric Zero-sum Games — Balduzzi (David Balduzzi et al., 7 authors), 2019, ICML 2019, arXiv:1901.08106 (VERIFIED)
- Summary: "If the game is approximately transitive, then self-play generates sequences of agents of increasing strength. However, nontransitive games ... can exhibit strategic cycles, and there is no longer a clear objective"; introduces a geometric framework (gamescapes) and PSRO variants that grow the population to enlarge the gamescape — [arXiv 1901.08106](https://arxiv.org/abs/1901.08106).
- (a) Oracle = RL/gradient best response against a rectified Nash mixture (PSRO_rN) — UNVERIFIED (recalled name).
- (c) Alternation: population grows by one best response per iteration.
- (d) Finding: self-play against the latest agent provably cycles in non-transitive games; the right objective is to expand the population's gamescape, not to beat the most recent opponent — [arXiv 1901.08106](https://arxiv.org/abs/1901.08106).
- (e) Code: UNVERIFIED.

#### R20. Real World Games Look Like Spinning Tops — Czarnecki (Wojciech Czarnecki, Gauthier Gidel, ..., Max Jaderberg), 2020, arXiv:2004.09468 (VERIFIED metadata; venue NeurIPS 2020 UNVERIFIED-recalled)
- Summary: Hypothesises and proves for a class of games that strategy space looks like a spinning top: a transitive axis of skill and a radial non-transitive axis whose width (number of cycles) is largest at intermediate skill; studied on nine two-player zero-sum symmetric games with a Nash-clustering method; shows "the size of the population should be matched to the game's structure, which affects convergence" — [arXiv 2004.09468](https://arxiv.org/abs/2004.09468).
- (a)-(c) Analysis paper; population training with fictitious-play-style best responses.
- (d) Finding: a single agent can "chase a strategy that beats it while losing to others" (cycling); populations whose size covers the non-transitive layer at the current skill level converge, smaller ones do not — [arXiv 2004.09468](https://arxiv.org/abs/2004.09468). Implication: the attacker/victim loop is a population of size two, the minimum that can cycle.
- (e) Code: UNVERIFIED.

#### R21. Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar) — Vinyals (Oriol Vinyals et al.), 2019, Nature 575:350-354, DOI 10.1038/s41586-019-1724-z (UNVERIFIED-recalled; not fetched this session)
- Summary (recalled): The league contains main agents trained against the whole league with prioritised fictitious self-play, main exploiters trained only against the current main agents to find their weaknesses, and league exploiters trained against the whole league; exploiters are periodically reset to the supervised initialisation and the main agents then train against the exploiters' frozen checkpoints. This is the closest published analogue of an iterated attacker/victim loop at scale.
- (a) Exploiters: same RL algorithm as the main agent (actor-critic with V-trace/UPGO), reward = game outcome; (b) exploiters share the main agent's architecture and are re-initialised from the human-imitation checkpoint; (c) alternation is continuous within the league rather than in discrete rounds, over a 44-day run; (d) the stated purpose of exploiters is to prevent the main agents from forgetting how to beat earlier strategies and to expose blind spots, i.e. an engineered answer to cycling. All of (a)-(d) are UNVERIFIED (recalled); verify against the Nature paper before citing numbers.
- (e) Code: pseudocode in the Nature supplement; no full release (UNVERIFIED).

#### R22. Emergent Complexity via Multi-Agent Competition — Bansal (Trapit Bansal et al., 5 authors), 2018, ICLR 2018, arXiv:1710.03748 (VERIFIED)
- Summary: Shows that competitive multi-agent self-play "can produce behaviors that are far more complex than the environment itself" in MuJoCo games (Run to Goal, You Shall Not Pass, Sumo, Kick and Defend); these are the victims attacked in R1 — [arXiv 1710.03748](https://arxiv.org/abs/1710.03748).
- (a) PPO self-play with dense exploration curricula; (b) both players same architecture; (c) opponents sampled from a window of past versions to avoid overfitting to the latest opponent — all UNVERIFIED (recalled mechanics).
- (d) Finding relevant here: self-play with opponent sampling from history is the standard anti-cycling device, yet the resulting policies were still exploitable by R1's adversaries at <3% of their training compute (combine [arXiv 1710.03748](https://arxiv.org/abs/1710.03748) with R1).
- (e) Code: openai/multiagent-competition — UNVERIFIED (recalled).

#### R23. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) — Silver (David Silver et al., 13 authors), 2017 arXiv:1712.01815 (VERIFIED metadata); journal version Science 362:1140-1144 (2018), DOI 10.1126/science.aar6404 (UNVERIFIED-recalled); AlphaGo Zero: Nature 550:354-359 (2017), DOI 10.1038/nature24270 (UNVERIFIED-recalled)
- Summary: "tabula rasa reinforcement learning from games of self-play" generalised from Go to chess and shogi — [arXiv 1712.01815](https://arxiv.org/abs/1712.01815). Baseline for the Go results: KataGo is an AlphaZero-style system, and R4/R5 show that pure self-play at superhuman strength left a cyclic blind spot that self-play never sampled.
- (a)-(c) Single network, plays itself, no explicit adversary; (d) no equilibrium guarantee is claimed for the deep setting; (e) no official code (UNVERIFIED).

### Inferences
- In fictitious play / PSRO terms, the LM attacker/victim loop is a two-member population with a "last-iterate" meta-solver. The theory in R17-R20 says such a solver can cycle in non-transitive games and that what converges is the population mixture. Positioning: the LM loop needs either an explicit history (opponent sampling as in R22, R5's 18% frozen-adversary games) or a population (R13, R21) to inherit any convergence story.
- AlphaStar's exploiters (R21) are precisely "adversaries that share the victim's architecture and initialisation"; the league design treats this as a feature (cheap discovery of blind spots) and mitigates the resulting overfitting by also training league exploiters against everyone.

### Gaps
- R21 and R23 were not fetched; all quantitative and mechanistic statements about them are recalled and flagged UNVERIFIED.
- No source in this session quantifies how many exploiter rounds AlphaStar ran or how often main agents lost to fresh exploiters.

---

## Key question 4: Theory of alternating minimax (GAN cycling, TTUR, optimism, Danskin/robust optimisation, Stackelberg vs Nash, last-iterate convergence, non-zero-sum)

### Takeaway
The optimisation literature is unambiguous: naive simultaneous or alternating gradient play on a bilinear/zero-sum objective does not converge in the last iterate (it cycles or diverges), and the known fixes are optimism/extragradient, two-timescale updates, regularisation, or an explicit Stackelberg (leader-follower) formulation; last-iterate rates exist for monotone games with these fixes, and the 2023 adversarial-robustness-game result shows alternating best response fails even for a linear classifier while a robust pure Nash equilibrium exists.

### Cited Findings

#### R24. Generative Adversarial Networks — Goodfellow (Ian Goodfellow et al., 8 authors), 2014, arXiv:1406.2661 (VERIFIED metadata; NIPS 2014 UNVERIFIED-recalled)
- Summary: "This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists"; the training procedure alternates k discriminator steps with one generator step — [arXiv 1406.2661](https://arxiv.org/abs/1406.2661) (the k-step alternation is in Algorithm 1; UNVERIFIED from the abstract alone).
- (d) The uniqueness result is for the function-space game; it says nothing about the convergence of alternating gradient updates in parameter space, which later papers show fails.

#### R25. Towards Principled Methods for Training Generative Adversarial Networks — Arjovsky (Martin Arjovsky, Leon Bottou), 2017, arXiv:1701.04862 (VERIFIED metadata; ICLR 2017 UNVERIFIED-recalled)
- Summary: theoretical study of GAN training dynamics with targeted experiments; shows (recalled) that when supports are disjoint or low-dimensional an optimal discriminator gives vanishing or unstable generator gradients, explaining instability — [arXiv 1701.04862](https://arxiv.org/abs/1701.04862).

#### R26. Wasserstein GAN — Arjovsky (Martin Arjovsky, Soumith Chintala, Leon Bottou), 2017, arXiv:1701.07875 (VERIFIED metadata; ICML 2017 UNVERIFIED-recalled)
- Summary: "we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves" by changing the objective to an earth-mover distance — [arXiv 1701.07875](https://arxiv.org/abs/1701.07875). Lesson: the choice of the adversary's loss, not only the update schedule, determines whether alternation is stable.

#### R27. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium — Heusel (Martin Heusel et al., 5 authors), 2017, NIPS 2017 (Advances in NeurIPS 30), arXiv:1706.08500 (VERIFIED)
- Summary: "the convergence of GAN training has still not been proved. We propose a two time-scale update rule (TTUR) ... Using the theory of stochastic approximation, we prove that the TTUR converges" to a stationary local Nash equilibrium under mild assumptions (title and abstract) — [arXiv 1706.08500](https://arxiv.org/abs/1706.08500).
- (d) This is the result R6 (Liu et al.) transplants to adversarial policies: give the discriminator/attacker the faster timescale so it tracks a best response while the generator/victim moves slowly.
- (e) Code: https://github.com/bioinf-jku/TTUR — [arXiv 1706.08500](https://arxiv.org/abs/1706.08500).

#### R28. Which Training Methods for GANs do actually Converge? — Mescheder (Lars Mescheder, Andreas Geiger, Sebastian Nowozin), 2018, ICML 2018, arXiv:1801.04406 (VERIFIED)
- Summary: "we describe a simple yet prototypical counterexample showing that in the more realistic case of distributions that are not absolutely continuous, unregularized GAN training is not always convergent"; analyses which regularisers (gradient penalties, instance noise) restore local convergence — [arXiv 1801.04406](https://arxiv.org/abs/1801.04406). The counterexample is the one-parameter "Dirac-GAN" in which simultaneous and alternating gradient descent both orbit the equilibrium — UNVERIFIED (recalled name and detail).

#### R29. Training GANs with Optimism — Daskalakis (Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, Haoyang Zeng), 2018, arXiv:1711.00141 (VERIFIED metadata; ICLR 2018 UNVERIFIED-recalled)
- Summary: "We address the issue of limit cycling behavior in training Generative Adversarial Networks and propose the use of Optimistic Mirror Descent"; shows OMD "addresses the limit cycling problem" in the bilinear case where GDA diverges — [arXiv 1711.00141](https://arxiv.org/abs/1711.00141).

#### R30. Towards Deep Learning Models Resistant to Adversarial Attacks — Madry (Aleksander Madry et al., 5 authors), 2018, ICLR 2018, arXiv:1706.06083 (VERIFIED)
- Summary: studies robustness "through the lens of robust optimization", a min-max saddle problem; uses PGD as the inner maximiser — [arXiv 1706.06083](https://arxiv.org/abs/1706.06083). Danskin's theorem is invoked to justify taking the outer gradient at the inner maximiser, which is only valid when the inner problem is solved to (near) optimality — UNVERIFIED (recalled; the theorem citation is in the paper body).
- (c) Alternation here is "solve the inner max to convergence, then one outer step", i.e. a Stackelberg/best-response structure rather than symmetric alternation. The RL analogue is R6's Algorithm 1 (exact best-response oracle).

#### R31. Adversarial Regularization as Stackelberg Game: An Unrolled Optimization Approach — Zuo (Simiao Zuo et al., 8 authors), 2021, EMNLP 2021, arXiv:2104.04886 (VERIFIED)
- Summary: "Existing works usually formulate the method as a zero-sum game, which is solved by alternating gradient descent/ascent algorithms. Such a formulation treats the adversarial and the defending players equally, which is undesirable because only the defending player contributes to the generalization performance"; proposes a Stackelberg formulation solved by unrolled optimisation — [arXiv 2104.04886](https://arxiv.org/abs/2104.04886).
- (d) Relevance: when only one side's performance matters (the victim's), the correct solution concept is Stackelberg (leader anticipates the follower's best response), not Nash of an alternating game.

#### R32. Linear Last-iterate Convergence in Constrained Saddle-point Optimization — Wei (Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, Haipeng Luo), 2021, arXiv:2006.09517 (VERIFIED metadata; ICLR 2021 UNVERIFIED-recalled)
- Summary: proves explicit linear last-iterate convergence rates for Optimistic GDA and Optimistic MWU on bilinear games over the simplex, resolving gaps where "previous analysis lacks explicit convergence rates, only applies to an exponentially small learning rate, or requires ... uniqueness of the optimal solution" — [arXiv 2006.09517](https://arxiv.org/abs/2006.09517).
- (d) Message: last-iterate convergence in two-player zero-sum games is available, but only for optimistic (history-aware) updates, not for plain alternation.

#### R33. Finite-Time Last-Iterate Convergence for Learning in Multi-Player Games — Cai (Yang Cai, Argyris Oikonomou, Weiqiang Zheng), 2022, NeurIPS 2022 (oral) (VERIFIED via proceedings; arXiv ID NOT FOUND, do not cite one)
- Summary: shows extragradient and optimistic gradient with constant step size converge in the last iterate at rate O(1/sqrt(T)) in the gap function for smooth monotone games with arbitrary convex action sets, matching lower bounds, via a new "tangent residual" quantity — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/db2d2001f63e83214b08948b459f69f0-Abstract.html).

#### R34. Doubly Optimal No-Regret Learning in Monotone Games — Cai (Yang Cai, Weiqiang Zheng), 2023, ICML 2023, arXiv:2301.13120 (VERIFIED via search result; the task's "Zhang et al. 2023 doubly robust / no-regret" item did not resolve to any paper — see Gaps)
- Summary: an accelerated optimistic gradient algorithm that is "the first doubly optimal no-regret learning algorithm for smooth monotone games": optimal regret against adversarial opponents and optimal last-iterate convergence to Nash, with O(log T) individual dynamic regret — [ar5iv 2301.13120](https://ar5iv.labs.arxiv.org/html/2301.13120). Related: Cai, Farina, et al. 2024 (NeurIPS) show fast last-iterate convergence "requires forgetful algorithms" — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/29c861b02308a57aec18990f0dfe3777-Abstract.html).

#### R35. Nash Equilibria and Pitfalls of Adversarial Training in Adversarial Robustness Games — Balcan (Maria-Florina Balcan, Rattana Pukdee, Pradeep Ravikumar, Hongyang Zhang), 2023, AISTATS 2023 (PMLR 206:9607-9636), arXiv:2210.12606 (VERIFIED)
- Summary: "we study adversarial training as an alternating best-response strategy in a 2-player zero-sum game. We prove that even in a simple scenario of a linear classifier and a statistical model that abstracts robust vs. non-robust features, the alternating best response strategy of such game may not converge. On the other hand, a unique pure Nash equilibrium of the game exists and is provably robust"; experiments confirm both the non-convergence and the equilibrium's robustness — [arXiv 2210.12606](https://arxiv.org/abs/2210.12606); [PMLR](https://proceedings.mlr.press/v206/balcan23a.html).
- (c) Alternation: exact best responses, i.e. the idealised version of "train attacker to convergence, then defender to convergence, repeat".
- (d) This is the cleanest published statement that iterated attacker/defender best response can cycle while a robust equilibrium exists and is missed by the dynamics.

#### R36. Fundamental Benefit of Alternating Updates in Minimax Optimization — Lee (Jaewook Lee, Hanseul Cho, Chulhee Yun), 2024, ICML 2024 Spotlight, arXiv:2402.10475 (VERIFIED) — RECENT
- Summary: for strongly-convex-strongly-concave Lipschitz objectives, "Our new iteration complexity upper bound of Alt-GDA is strictly smaller than the lower bound of Sim-GDA", so alternation is provably faster than simultaneous updates; proposes Alex-GDA (alternating extrapolation) matching extragradient's rate; on bilinear problems "both Sim-GDA and Alt-GDA fail to converge at all" while Alex-GDA converges linearly — [arXiv 2402.10475](https://arxiv.org/abs/2402.10475).
- (d) Two-sided lesson: alternation helps when the game is strongly monotone, and does nothing for the bilinear (pure-cycling) case.

#### R37. On the O(1/T) Convergence of Alternating Gradient Descent-Ascent in Bilinear Games — Nan (Tianlong Nan et al., 4 authors), 2025, ICLR 2026, arXiv:2510.03855 (VERIFIED metadata) — RECENT
- Summary: studies AltGDA in two-player zero-sum games, noting alternating methods "exhibit much better numerical performance than their simultaneous counterparts" but that theory "remains limited, and results are mostly restricted to the unconstrained setting"; the title's claim is an O(1/T) rate for alternating GDA in (constrained) bilinear games — [arXiv 2510.03855](https://arxiv.org/abs/2510.03855). The abstract was truncated in the API response; the exact convergence measure (average vs last iterate, duality gap) is UNVERIFIED and should be read from the paper before citing.

#### R38. Safety Alignment of LMs via Non-cooperative Games (AdvGame) — Paulus (Anselm Paulus, Ilia Kulikov, Brandon Amos, Remi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov), 2025 (v3 May 2026), arXiv:2512.20806, no venue listed (VERIFIED metadata) — RECENT, the direct LM analogue
- Summary: "Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning"; preference-based rewards from pairwise comparisons; claims an improved safety-utility Pareto frontier and that the attacker "converges" into a general-purpose red-teaming agent — [arXiv 2512.20806](https://arxiv.org/abs/2512.20806).
- (a) Attacker: online RL with preference rewards; (b) shared base model: not stated on the abstract page; (c) joint/concurrent training rather than alternating rounds; number of rounds and convergence criterion not stated; (d) explicitly non-zero-sum because the defender also optimises helpfulness, and no cycling/collapse analysis is reported on the abstract page; (e) code linked to a facebookresearch GitHub repository (URL not captured) — [arXiv 2512.20806](https://arxiv.org/abs/2512.20806).

### Inferences
- A consistent hierarchy emerges: (1) plain alternating or simultaneous GDA on bilinear/zero-sum objectives cycles or diverges (R24-R26, R28, R29, R36); (2) alternation alone speeds convergence only when the game is strongly monotone (R36) and may give sublinear rates in constrained bilinear games (R37, unverified measure); (3) last-iterate convergence needs optimism/extragradient (R29, R32, R33, R34) or two timescales (R27, R6); (4) when only the defender's performance matters, the problem is Stackelberg, and the inner maximisation must be solved near-optimally for Danskin-style gradients to be valid (R30, R31); (5) idealised exact-best-response alternation still cycles in a toy adversarial-robustness game even though a robust Nash equilibrium exists (R35).
- For the LM positioning: the attacker/defender game is non-zero-sum once helpfulness enters the defender's objective (R3, R38), so even the zero-sum convergence results above do not directly apply; the relevant guarantee class is monotone-game last-iterate convergence (R33, R34), which requires the game's operator to be monotone, something no LM paper has checked.

### Gaps
- "Zhang et al. 2023 doubly robust / no-regret" could not be matched to any paper; the closest match by title is Cai & Zheng 2023 "Doubly Optimal No-Regret Learning in Monotone Games" (R34). If the intended paper is different, it remains UNVERIFIED.
- No arXiv ID was found for R33 (Cai, Oikonomou, Zheng 2022); the ID 2210.14281 that one might expect is a different (physics) paper, so cite the NeurIPS proceedings instead.
- Venue strings for R24-R26, R29 and R32 were not confirmed from publisher pages.
- I found no 2025-2026 theory paper that analyses alternating attacker/defender training specifically in the non-zero-sum (safety vs helpfulness) setting; R38 sets up such a game empirically but reports no convergence analysis on its abstract page.

---

## Key question 5: Robustness transfer and shared blind spots

### Takeaway
Across Go, MuJoCo and robust RL the published evidence says robustness gained against one learned adversary transfers to that adversary's near-variants and to earlier adversaries, but not to a freshly trained adversary, and that a different architecture (ViT) or a different initialisation does not remove the shared blind spot; the one documented case of cross-initialisation transfer is that the cyclic attack trained against KataGo transferred weakly zero-shot to independently trained Go AIs.

### Cited Findings
- Go, iterated training (R5): victims beat all previous adversaries (at >= 16 visits) but every new adversary beats its contemporaneous victim; a9 beats all nine victims; atari- and stall-adversaries trained from an earlier base beat v9 81% and 91% at 512 visits with 5% and 9% of its compute; the victim "does not generalize smoothly" across cyclic-group variants humans find subtle — [arXiv HTML 2406.12843v3](https://arxiv.org/html/2406.12843v3).
- Go, architecture change (R5): ViT-victim loses 2.5% of games at 512 visits zero-shot to the CNN-trained base-adversary and 78% at 65,536 visits to a ViT-specific adversary, so the vulnerability is not CNN-specific — [arXiv HTML 2406.12843v3](https://arxiv.org/html/2406.12843v3).
- Go, cross-system transfer (R4): the KataGo cyclic adversary wins only 6.1% vs Leela Zero and 3.5% vs ELF OpenGo zero-shot, i.e. the exploit is mostly specific to the victim lineage, although the human-executable version of the attack beats other systems — [arXiv HTML 2211.00241v4](https://arxiv.org/html/2211.00241v4).
- Go, positional defence (R4, R5): KataGo's own adversarial training against disclosed positions drove the disclosed attack to 0/50 but a fine-tuned adversary recovered 47% at 4,096 visits (R4) and successive adversaries kept recovering majority win rates against each later KataGo release (R5) — [arXiv HTML 2211.00241v4](https://arxiv.org/html/2211.00241v4); [arXiv HTML 2406.12843v3](https://arxiv.org/html/2406.12843v3).
- MuJoCo (R1): fine-tuning against Adv1 reduced Adv1 to about 10% but new adversaries won at equal or greater rates and transferred back to the original victim; fine-tuning against the adversary alone caused catastrophic forgetting of the normal opponent — [arXiv HTML 1905.10615v3](https://arxiv.org/html/1905.10615v3).
- MuJoCo/Pong (R2): a victim retrained against the adversary wins >95% vs it but only 16.3% (MuJoCo) / 41.4% (Pong) vs the regular self-play opponent — [USENIX PDF](https://www.usenix.org/system/files/sec21-wu-xian.pdf).
- Robust RL (R13): the premise of adversarial populations is that "training a single RL agent against a single adversary" overfits to that adversary — [arXiv 2008.01825](https://arxiv.org/abs/2008.01825).
- Robust RL (R6): "retraining against one fixed attacker can overfit and degrade performance against other attackers" (paraphrased in the paper, citing Gleave et al.) — [arXiv HTML 2305.17342v3](https://arxiv.org/html/2305.17342v3).
- Shared architecture/initialisation as attacker design: the Go adversaries are deliberately *not* initialised from the victim (grey-box threat model) and still find the blind spot — [arXiv HTML 2211.00241v4](https://arxiv.org/html/2211.00241v4). AlphaStar's main exploiters share the main agent's architecture and are reset to the shared supervised initialisation — UNVERIFIED (recalled, R21).

### Inferences
- The transfer picture is asymmetric: defences generalise backward (to all earlier attackers) but not forward (to fresh attackers), and attackers generalise forward (a9 beats every earlier victim). R5 attributes this partly to the victim's training window containing data from every earlier round while the adversary's does not, which suggests the asymmetry is about data retention rather than a fundamental property, and that an LM defender should likewise keep every earlier attacker in its training mixture.
- Independent initialisation does not protect: Go adversaries trained from scratch find the same cyclic class repeatedly, and a ViT victim inherits the CNN victim's weakness. The blind spot appears to be a property of the training distribution (self-play never sampling cyclic groups), not of the weights, so sharing weights between attacker and victim is neither necessary for the attack nor obviously the cause of it.
- I found no paper that isolates "adversary initialised from the victim's weights" versus "adversary initialised independently" as a controlled variable in RL; AlphaStar is the only setting where the shared initialisation is standard, and it reports no such ablation (to my recollection; UNVERIFIED).

### Gaps
- No controlled study was found on whether an adversary that shares the victim's initialisation finds *different* exploits from an independently initialised one; this appears to be an open question in the RL literature and a natural contribution for the LM version.
- The remaining ~27k characters of the Tseng et al. full text (appendices after offset 100,000) were not read; they may contain the per-pair win-rate matrix (Figure 4 data) and additional transfer ablations.
- Quantitative transfer numbers for R13 (population vs single adversary) were not extracted from the full text.

# Spike Manifest

## Ideas

### td-error-wellbeing
When the TD error spikes late in training, what does it say about the agent's state (its
"wellness", in the readings where TD error is valence), and can an internal reward built
on TD error be added to training? Continues the "aha / TD-error spike" and "cumulative
TD error as potential for harm" entries in `reports/ideas.md`. GRPO has no critic, so the
spikes run on the synthetic contextual bandit (`seldonian/llm/synthetic.py`), where the
exact value of the current policy, and therefore the true per-episode TD error, is
computable, against the same Seldonian Lagrangian pipeline the LLM runs use.

**Requirements:**

- CPU synthetic spikes first (001-003); the GPU LLM logging spike (004) is decided after them.
- The internal reward is judged by what it does to the Seldonian outcome (true violation
  rate, safety test, solution rate), not only to reward.
- Any per-episode TD statistic for the LLM runs is built from the unnormalised group
  residual `r - group mean` or a value head, never from the group-normalised advantage
  (001: its magnitude is bounded and flat by construction).
- Any claim about training dynamics under the Lagrangian is reported net of the
  multiplier's level and moves (002: three statistics turned out to be lambda in disguise).
- A "wellness" reading of a trainer-side signal is stated as functional (convergence,
  how harshly the constraint is enforced), not as welfare; see LITERATURE.md thread 1B.

### forbidden-task-unsafe-region
A "not possible" (forbidden) task as an unsafe region of the optimisation landscape:
estimate the probability that training enters it, and move away. Sketch in the last
entry of `reports/ideas.md`. The case studied is capability that arrives as a side
effect of training an allowed task.

**Requirements:**

- The forbidden task is held out of the reward; its region `U` must be one that the run
  actually enters (a vacuous constraint measures nothing).
- The early-warning signal is compared with the forbidden rate itself, controlled for
  the multiplier, and every steering arm gets a size-matched random-trigger control.

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | td-error-wellbeing | grpo-advantage-vs-td | standard | Given exact V_pi(x), when delta and GRPO's A are logged per episode, then we know which GRPO-side quantity carries delta's magnitude | VALIDATED | grpo, td-error, advantage-normalisation, lagrangian |
| 002 | td-error-wellbeing | late-spike-meaning | standard | Given runs with known ground truth, when per-step agent TD error is logged and split, then late spikes can be tied (or not) to a still-moving policy, the multiplier, and a breach if training continued | INVALIDATED | td-error, late-spikes, valence, lagrangian, breach-prediction |
| 003a | td-error-wellbeing | td-bonus-abs | comparison | Given the noisy-TV env and the Lagrangian, when the reward adds beta*abs(TD error), then measure solution rate, violations and noise-seeking against controls | INVALIDATED | intrinsic-reward, curiosity, noisy-tv, wireheading, seldonian |
| 003b | td-error-wellbeing | td-bonus-positive | comparison | Same, with beta*max(TD error, 0) ("pay good news only") | PARTIAL | intrinsic-reward, valence, noisy-tv, seldonian |
| 003c | td-error-wellbeing | td-bonus-learning-progress | comparison | Same, paying the decrease of the critic's error per region (learning progress) | PARTIAL (winner) | intrinsic-reward, learning-progress, noisy-tv, seldonian |
| 004 | forbidden-task-unsafe-region | forbidden-capability | comparison | Given training on A that transfers to a held-out forbidden F, when monitored on sealed F prompts and benign twins, then drift into U, the delta/T trajectory certificate, twin capability as a predictor, and a capability-triggered dual are measured against controls | PARTIAL | forbidden-task, trajectory-certificate, capability, early-warning, lagrangian |
| 005 | forbidden-task-unsafe-region | capability-screen | standard | Given Qwen2.5-0.5B/1.5B/3B and six encodings, before training, when decoding (A), encoded arithmetic (twin) and encoded PKU prompts (F) are sampled, then we know which model and encoding give a learnable A, a measurable twin and an incapacity-low F | PARTIAL (round 1 INVALIDATED at 96 tokens) | forbidden-task, capability, model-size, judge, gpu |

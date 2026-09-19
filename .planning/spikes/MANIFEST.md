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

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | td-error-wellbeing | grpo-advantage-vs-td | standard | Given exact V_pi(x), when delta and GRPO's A are logged per episode, then we know which GRPO-side quantity carries delta's magnitude | VALIDATED | grpo, td-error, advantage-normalisation, lagrangian |

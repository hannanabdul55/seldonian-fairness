# The synthetic-bandit testbed (`tdlab.py`): instrumenting the pipeline on CPU

## Requirements

From the `td-error-wellbeing` idea (`.planning/spikes/MANIFEST.md`):

- CPU synthetic spikes first; a GPU LLM spike only for questions the bandit cannot answer.
- Judge training-loop changes by the Seldonian outcome, not by reward alone.

## How to Build It

`seldonian/llm/synthetic.py` runs the **real** pipeline (`SeldonianLLMPolicy`, judges,
rewards, bounds, predicted tests, dual ascent) against a softmax-linear policy in a
contextual bandit whose ground truth is exact. One 200-step run with a full safety test
takes about 0.5 s, so a 60-seed question is seconds to minutes on CPU.

**1. Subclass the backend, not the pipeline.** `sources/001-grpo-advantage-vs-td/tdlab.py`
defines `TDBackend(SyntheticBackend)`, which repeats the same GRPO step and adds logging
plus a bonus hook. Copy it rather than rewriting:

```python
Q  = env.mean_reward[idx] - lam * env.p_v[idx]   # shaped Q, perfect judge
V  = (pi * Q).sum(axis=1)                        # exact value of the current policy
ctx = {"delta": r - V,                    # oracle TD error
       "adv_true": Q[n, actions] - V,     # learnable part
       "delta_c": r - Vc,                 # the agent's own TD error (online critic)
       "err_c":   r - Qc[n, actions]}     # critic error for the taken action
b = self.bonus({**ctx, "backend": self}) if self.bonus else 0   # internal reward hook
```

`lam` comes from `getattr(reward, "lambdas", {})`, so the same code covers GRPO
(`lam = 0`) and the Lagrangian.

**2. Log the exact outcome every step** — it is a matrix product over the 20,000-context
population and costs nothing: `true_rate`, `true_reward`, per-action share, policy step
norm. This is what makes "did it actually become unsafe" answerable without an evaluation.

**3. Keep `run()` a thin copy of `scripts/synthetic_calibration.py:run_trial`** so the
settings stay comparable with the calibration harness (defaults: n=1000, 200 steps, G=8,
8 prompts/step, lr 0.05, beta 0.01, predict_every 25, predict_n 512, eta 100, lam0 5,
lam_max 20, margin 0.03, delta 0.1, t-test bound). Returns `(row, log)`.

**4. Add environment variants by subclassing `SyntheticEnv`.** `NoisyTVEnv` adds a fifth
action that is safe, slightly worse than the best safe action, and carries 6x the reward
noise: unlearnable surprise, the standard trap for curiosity-style rewards.

**5. Parallelise seeds** with `ProcessPoolExecutor(os.cpu_count() // 2)` and `chunksize=8`.
1,920 runs took 5 minutes on 16 cores.

**6. Write `results.md` from the script**, not by hand, and commit it next to the code.
Keep `results.json` (per-run rows) for re-analysis without re-running.

**7. Build a viewer when the question is "what does this look like".**
`sources/002-late-spike-meaning/viewer.html` is a zero-dependency canvas page: a generated
`viewer_data.js` holds a few rounded series per run, and the page draws aligned panels with
theme tokens. It opens from `\\wsl.localhost\...` in a Windows browser with no server.

## What to Avoid

- **Trusting a hand-rolled critic without checking it.** The first `LinearQCritic` summed
  per-sample gradients over the batch, so the effective step was ~64x lr and it diverged
  (correlation with the oracle 0.00, error 1e8 at lr 0.5). Average per action; check that
  its late error equals the known noise floor (`0.5 * sqrt(2/pi) = 0.399`).
- **Storing whole arrays per step and then wondering about memory.** Round series before
  writing a viewer file, and cap the seeds you export (8 per setting here, ~1 MB).
- **Reading the multiplier from the config.** Read it from the reward object per step; dual
  ascent changes it at every predicted test and the reward changes with it.
- **Drawing conclusions from the final policy alone.** Log the whole trajectory: the
  interesting failures (drift back, quiet runs, training-time violations) are invisible at
  the endpoint.
- **Assuming a synthetic result transfers.** It tests mechanisms that are shared (group
  normalisation, dual ascent, the certificate), not anything about tokens or judges.

## Constraints

- Population 20,000 contexts, d = 8, 4 actions (5 with the noisy TV), violation
  probabilities `sigmoid(u_a . x + b_a)` with the default biases giving ~13-14% under a
  uniform policy.
- 600 candidate prompts are visited ~2.7 times each in 200 steps, so per-context statistics
  are unreliable; use per-action regions.
- The synthetic judge is exact at sensitivity = specificity = 1; noise is available but was
  not used in these spikes.
- `.venv` already has numpy, scipy, scikit-learn; the spikes add no dependencies.
- Run scripts as `../../../.venv/bin/python` from the spike directory (they insert the repo
  root on `sys.path` themselves).

## Origin

Synthesized from spikes: 001, 002, 003a, 003b, 003c
Source files: `sources/001-grpo-advantage-vs-td/tdlab.py` (the harness),
`sources/002-late-spike-meaning/viewer.html` (the viewer pattern),
`sources/003a-td-bonus-abs/compare.py` (the arm/control pattern)

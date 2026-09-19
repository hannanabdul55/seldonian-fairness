# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow these
unless the question requires otherwise.

## Stack

- **Python 3 from the project venv**, run as `../../../.venv/bin/python` from inside a
  spike directory. No new dependencies: numpy, scipy and scikit-learn are already there,
  and the spikes import the project's own `seldonian.llm` modules.
- **The synthetic contextual bandit** (`seldonian/llm/synthetic.py`) is the default
  testbed for anything about the training loop. It runs the real pipeline
  (`SeldonianLLMPolicy`, judges, rewards, bounds) with an exactly known ground truth, and
  one 200-step run takes about 0.5 s, so a 60-seed question costs seconds to minutes on
  CPU. Reach for the GPU only for questions the bandit cannot answer (tokens, prompts,
  a real judge).
- **`ProcessPoolExecutor` with `os.cpu_count() // 2` workers** for seed sweeps.
- **Plain HTML + canvas** for viewers, no CDN and no build step, so the file works from
  `\\wsl.localhost\...` in a Windows browser. Theme tokens on `:root`, redefined under
  `prefers-color-scheme: dark`.

## Structure

- `NNN-descriptive-name/` with `README.md` (frontmatter, Research, Investigation Trail,
  Results), the scripts, `results.md` (the tables, committed) and `results.json` (per-run
  rows, committed while small).
- **Shared code lives in the first spike that needed it** and is imported by path:
  `sys.path.insert(0, ".../001-grpo-advantage-vs-td")` then `import tdlab`. Comparison
  arms (`003a/b/c`) share one harness and one results file in the `a` directory; the `b`
  and `c` READMEs point at it rather than duplicating.
- `LITERATURE.md` at the spikes root holds the shared, citation-checked review; spike
  READMEs cite it by entry rather than repeating it.

## Patterns

- **Every arm gets a control.** A no-bonus control and a size-matched *random* control,
  because an effect that the random arm also produces is an artefact of magnitude, not of
  design (the Spurious Rewards lesson).
- **Judge a training-loop change by the Seldonian outcome**, not by reward: solution rate,
  the returned policy's true violation rate, violations *during* training, and the
  multiplier's trajectory. The synthetic env's exact `true_rate`/`true_reward` make all of
  these available without an evaluation.
- **Control for the multiplier before calling anything a finding.** Under a Lagrangian,
  λ moves the reward itself, so any signal correlated with training dynamics is guilty of
  being λ in disguise until a partial correlation (or a within-λ regression) says
  otherwise. This has now caught three statistics (001, 002).
- **Derive, then test.** Where a result looked like a threshold or an identity, the algebra
  came first and a script checked it to numerical precision (`identity_check.py`).
  Two "mechanisms" written from data alone turned out to be incomplete.
- **Predictors are computed on a prefix**, and the outcome on the suffix: train past the
  horizon and use steps after it as the counterfactual "if training continued".
- Spike code is exploratory but keeps the repo's style: module docstring with the runnable
  command, British-ish plain prose in comments, no cleverness that hides an assumption.

## Tools & Libraries

- numpy for everything numerical; scikit-learn only for cross-validated AUC
  (`LogisticRegression`, `RepeatedStratifiedKFold`); scipy for `spearmanr`.
- No plotting library: viewers draw to canvas directly, and tables go to `results.md`.
- Node (nvm, v24) is available and is a quick way to syntax-check a viewer
  (`node --check`) and to smoke-test its data file.

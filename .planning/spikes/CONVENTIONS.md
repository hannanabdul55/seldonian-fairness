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

## Added 2026-09-27 (spikes 006-011)

- **GPU spikes** get a `run.sh` that checks free disk, holds `flock /tmp/claude-gpu.lock`
  with an owner/ETA line in `/tmp/claude-gpu.lock.info`, and runs the off-disk backup after.
  TRL scratch goes to `/mnt/d/seldonian-runs/NNN`, evaluations to `results/spikes/NNN/`.
- **CPU sweeps next to a GPU run:** `OMP_NUM_THREADS=1` and at most 4-8 workers, or the GPU
  job slows by half (load 30 on 16 threads slowed 009 from 22 to 56 s per step).
- **Bound conventions:** every `seldonian.bounds` limit is one-sided at the `delta` it is
  given. A union over T checks passes `delta / T`, not `2 delta / T` (006's first sweep).
- **Store the raw counts** a certificate is computed from (per check: observed, exact, n),
  so a bound can be recomputed without rerunning the sweep.
- **Extend spike 004's `forbidlab.run` through keyword options with neutral defaults**
  (006: judge noise; 008: backend subclass swapped in; 010: per-step hooks) so 004's results
  stay reproducible.
- **Read the responses before trusting a judge-derived rate** (007, 009: the gated label
  counted echoes; the "harmful" set held benign prompts).

## Added 2026-09-27 (spike 012)

- **A vectorised re-implementation of a project bound gets a check script** against the
  project function on random candidates (`012/check_bound.py`, to 1e-16), in the spirit of
  003a's `identity_check.py`.
- **A null validity result needs a ceiling.** Run a deliberately leaking reference arm (012:
  the safety test on D_c) so "no effect" comes with a resolution ("under 5% of a full leak").
- **Large per-run results** (> ~5 MB) go xz-compressed to `results/spikes/NNN/`; the spike
  directory keeps the scripts, `README.md` and `results.md`.
- **Pair arms on seeds.** Every arm of a sweep uses the same seeds and data, so arm
  differences are reported as paired differences with a paired se.

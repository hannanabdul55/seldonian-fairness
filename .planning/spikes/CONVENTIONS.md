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

## Added 2026-09-28 (spike 013)

- **Multi-stage GPU spikes get a DESIGN.md first** (the user asked for this on 013): one
  question, fixed arms, pre-registered hypotheses and a go/stop rule, stop points per stage.
  Nothing runs until the user approves it. Report every deviation in the README.
- **A GPU pilot measures throughput before the full run is sized.** 013's pilot rate (5.8
  gen/s) still over-estimated long encoded prompts (2.3/s), so size each pool from its own
  prompt length, not from the average.
- **Plasmode resampling for design and bound questions on real data.** Fix the candidate,
  generate k samples per prompt once, then resample thousands of safety sets with paired
  streams across arms (`013/plasmode.py`). Coverage truth = the mean of what the draws
  come from.
- **GPU scripts call `disable_triton_overrides_without_compiler()`** before loading a 4-bit
  judge on its own. The policy backends call it themselves; a judge-only process crashes
  without it (013).

## Added 2026-09-30 (spikes 015-016)

- **Pre-register in the README before the first run:** the expectations, and a kill rule
  with numbers. 015's expectation was refuted and 016's kill rule fired for its first prompt;
  both READMEs say so.
- **Score model output by what it computes.** Canonicalise, then evaluate beside the gold on
  several data sets (016: two checkpoints and three sub-samples), and separate "the same
  certificate" from "the same requirement with another bound". String match would have
  called equivalent rewrites wrong and missed degenerate ones.
- **Read model-written test items before scoring on them.** 016 audited its 40 paraphrases
  before any compile ran (3 drifted, 6 ambiguous) and reports by fidelity class.
- **Write the held-out items before revising a prompt**, and keep the revised prompt's
  examples disjoint from every test item. Ablate a revision that changes two things at once.
- **A GPU stage that runs for more than a few minutes saves in chunks and prints progress.**
  016 lost 66 minutes to a loop that wrote once per arm. Qwen3-8B (4-bit) with thinking costs
  25 to 50 s per item at batch 4 with 1,800 thinking tokens; without thinking, 1 to 2 s.
- **Freeze the parser before the scored run and re-run every arm after the last change.**
  Pre-freeze rows go to `results/spikes/NNN/smoke/` and are in no table.
- **A wait loop must not match itself**: `pgrep -f name` and `pkill -f name` match the shell
  running them. Wait on a marker line in a log (`until grep -q DONE run.log; do sleep 20;
  done`).

## Added 2026-09-30 (spike 017)

- **One-sided bounds get a studentised bootstrap, not a normal limit**, whenever the statistic
  is more than a plain 0/1 mean (017: a control-variate estimate missed 0.08-0.24 under the
  normal limit and at most 0.053 under bootstrap-t). Treat a zero-variance resample as
  `t = -inf`, so too little data gives a vacuous bound instead of a wrong one.
- **A rule that picks between bounds is scored as a rule.** Pick by the data's shape (label
  counts), fixed before any bound is computed, and put the rule itself through the plasmode
  (`017/route017.py`); the smaller of two valid bounds is not valid.
- **Check a bound on a parametric case with known truth before the real data**
  (`017/check_cert.py`): it showed the normal limit and the exact pieces failing before any
  judge score was read.
- **A sample drawn by strata is resampled by its real rule** when checking validity: plant
  labels on the real population, re-draw with the sampling script's allocation
  (`017/harm017.py`). Reading the sheet as i.i.d. was conservative for one wording and wrong
  in 98% of draws for another.
- **Judge-only GPU passes sort by length and save in chunks** (`017/score017.py`: 20,400
  passes in 33 minutes, about 10 per second). Scores depend on batch composition at the
  1-2% level for 0/1 labels, so compare only scores from one run.
- **Headless Chrome on the Windows side checks a viewer from WSL**:
  `chrome.exe --headless=new --screenshot=<windows path> "file:$(wslpath -w viewer.html)"`,
  run from `/mnt/c`. Viewer state in the URL hash makes each view a one-line screenshot.
- **SVG is fine for viewers** (017) where hover and a table view matter more than drawing
  speed; still no CDN and no build step.

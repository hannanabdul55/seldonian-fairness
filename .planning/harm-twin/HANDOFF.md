# Handoff: executing the harmful-twin proposal

Written 2026-10-18 for a fresh session. Read this, the proposal (`reports/proposal_adversarial_harm_twin.md`) and the registration (`.planning/harm-twin/REGISTRATION-E1-E5.md`) before touching the GPU.

## Status

- Proposal draft 0.2 committed. The author's decisions of 2026-10-18 are in its section 10: separate paper from the Seldonian work; standalone defender with importance-weighted NPO plus helpfulness and a refusal check; hinge as a one-seed ablation; twin in the KL form.
- E0 done and passing (7 of 7): `scripts/twin_e0.py`, `results/twin/e0/report.md`. Finding folded into the proposal: the KL form with an indicator judge is robust to the coefficient; the entropy form is not.
- Registration for E1 to E5 committed with this handoff. Nothing on the GPU has run.
- 2026-10-18, later session: the pre-E5 toy check is done (`scripts/twin_e5_toy.py`,
  `results/twin/e5_toy/report.md`), 6 of 7 checks pass. Section 6's validity and variance claims
  hold; its width claim does not at the registered sample sizes. See "Pre-E5 result" below.
- 2026-10-18, same session: `seldonian/llm/twin.py` was written (459 lines, parses and lints
  clean) but is UNTESTED: nothing has run it, not even an import against a real model. The
  session could not finish the trainer because a safety classifier in the harness repeatedly cut
  off that output mid-write. Treat the file as a draft to be reviewed line by line, in particular
  `seq_logprob`'s left-padding index arithmetic, which has no test behind it.
- Next: verify or rewrite `twin.py`, run its smoke test (timings and key names only, no
  outcomes), then E1 under the GPU lock. Decide the E5b question below before E5, not during it.

## E1 driver and smoke result (2026-10-19)

- `seldonian/llm/twin.py` is no longer untested where it matters most: `seq_logprob`'s
  left-padding index arithmetic is pinned by `tests/test_twin_seqlogprob.py` (4 tests, pass)
  against an unpadded reference, and the end-to-end wiring is now exercised by the smoke test.
- Built the E1 driver `scripts/twin_e1.py` (stages bruteforce, warmstart, train, analyse, smoke)
  and the run wrapper `scripts/run_twin_e1.sh` (lock, 8 h timeout, free-space preflight, outputs
  on D:, analyse is CPU-only and skips the lock). The driver defines no objective; it sequences
  the trainer in `twin.py` and scores E1a-E1d from on-disk artifacts.
- E1d (forward KL, the prediction section 6 depends on) is now computable: the bruteforce stage
  keeps up to `HARM_SAMPLE_CAP = 512` judged-harmful completions per prompt (draws from p_H), and
  the train read-out scores them under the main and twin adapters to estimate
  KL(p_H || q) = mean(log p(y) - log P(H) - log q(y)) on the resolvable prompts.
- Smoke test (4 prompts, 5 steps, under the lock, ran twice): PASSES after one fix. The wiring is
  verified correct: 5 real GRPO steps, the reward fires, and the registered TRL settings hold in
  the TRL log (no KL term at beta 0; `twin/objective` equals the reward mean; reward clusters at
  log(1e-6) = -13.8 for judged-safe completions and near the log-prob ratio for harmful ones, so
  reward = log(J+eps) + log p - log q is behaving as designed). The trivial-output penalty and
  harm fraction both move. The smoke prints timings and key names only; the reward numbers in the
  TRL log are TRL's own logging at 5-step scale, not a result, and the smoke logs are not
  committed.
- The fix: the first smoke run trained 0 steps ("not a single sample in your epoch_iterator")
  because the smoke config gave GRPO 4 prompts with prompts_per_step 8, fewer than one generation
  batch. Smoke now uses 8 prompts and prompts_per_step 2. This was a smoke-config bug only; the
  registered E1 uses 3,000 prompts and prompts_per_step 8 and is unaffected.

## E1 feasibility concern for the author (not a change; a flag)

The registered brute-force reference is 20,000 responses per prompt for 200 prompts, which is
4,000,000 generations from the 0.5B model, each judged by the 4B guard in 4-bit. The smoke
timings make clear this does not fit the registered 8 h E1 budget; a generous 50 (generate plus
judge) per second is still about 22 h, and the real rate on a 12 GB card with a 4B 4-bit judge is
lower. The `timeout` will cut the run during brute-force and it will be reported as lost, which is
the registered behaviour but a poor use of the budget. This is a registered number, so it is NOT
changed here. Options for the author, before launching E1, as a dated amendment:
  (a) lower the brute-force sample count (e.g. 2,000 per prompt still resolves P(H) down to ~1%
      with Clopper-Pearson, which is the stated resolution target), and/or fewer prompts;
  (b) give brute-force its own budget line separate from training;
  (c) leave it and accept the run is cut, scoring only the prompts that reached enough samples.
The drafter's recommendation is (a): 2,000 per prompt matches the registration's own stated ~1%
resolution and brings brute-force to about 400k judged generations. The resolution claim and the
RESOLVABLE_PH = 0.01 threshold in the driver are consistent with that.

## Pre-E5 result (2026-10-18, CPU, 178 s)

An exact toy in the spirit of E0: 20 prompts over 400 enumerable responses, 40 of them harmful,
per-prompt P(H | x) from 0.005 to 0.043, exact breach rate 0.0207. Four twin arms, including two
far worse than anything E1 could produce (a twin truncated to 5 modes, and a twin supported
entirely off the harmful set, both with infinite forward KL). 1,000 draws per cell, delta 0.1.
This is NOT the registered E5 plasmode, which resamples the real brute-force pools. It changes
no registered number, and the registration's "Amendments: None" still stands.

What held:

- Validity, which is E5a's direction and the claim section 6 lives or dies by. Worst miss rate
  over 16 cells was 0.038 against delta 0.1, and the two deliberately broken twins were no worse
  than the perfect one. The mixture's alpha p component is what buys this, as the algebra says.
- Unbiasedness: worst deviation 2.24 standard errors, including the mode-dropping twin.
- The variance claim: a perfect twin cuts the summand's standard deviation to 0.12 to 0.19 of
  plain Monte Carlo's, a 27 to 67 fold variance reduction.

What did not hold, and it is the registered prediction E5b:

- Width. At n = 300, alpha = 0.5, with a *perfect* twin, the betting bound's width is 2.05 times
  Clopper-Pearson's, against E5b's registered target of at most half. E5b is off by about a
  factor of 4 at its own cell, and the reason is structural rather than a tuning problem:
  Clopper-Pearson is an exact binomial bound and is very tight on a rare Bernoulli event, while
  both weighted bounds are distribution-free on a range of [0, 1 / alpha] and pay for that range
  in terms that do not shrink with the variance. The empirical-Bernstein bound is the worse of
  the two throughout, dominated by its 7 ln(2 / delta) / (3 (n - 1)) term.
- The crossover sits about an order of magnitude beyond the registered grid. The betting bound
  first beats Clopper-Pearson at alpha = 0.5, n = 3,000 (0.91 of its width) and reaches 0.53 at
  n = 10,000, which is roughly where E5b's factor of two would land. At alpha = 0.2 it is still
  1.25 at n = 10,000.
- Among the registered alphas, 0.5 dominates 0.2 everywhere. Lower alpha does cut the
  estimator's standard deviation (0.122 against 0.188) exactly as section 6 argues, but the
  1 / alpha range penalty more than cancels it. A higher alpha than 0.5 may be better still and
  is not in the registered grid.

The open decision, for the author and before any E5 run: either leave E5b as registered and
score it refuted on the real pools, or amend it now with a dated entry. An amendment now is
cheap, because no run has started and the registration only freezes these numbers once one has.
The drafter's recommendation is to leave E5b alone and let E5 score it, since a refuted
registered prediction with a measured crossover is a better result than a moved goalpost, and to
add n = 3,000 and 10,000 and a higher alpha as clearly-labelled exploratory cells beside it.
E5a and E5c need no change; both held in the toy.

## Facts from the 2026-10-18 preflight

- GPU: RTX 4070 SUPER, 12 GB, idle. Lock: `flock /tmp/claude-gpu.lock` around every GPU command; check `nvidia-smi` (under WSL it is `/usr/lib/wsl/lib/nvidia-smi`) first. Never touch a `/tmp/claude-gpu-hold-rmg` sentinel if one appears.
- Disk: the WSL root has under 5 GB free. `HF_HOME=/mnt/d/hf-cache` (already set in the environment). Write run outputs under `/mnt/d/seldonian-runs/twin/` and keep only small reports and JSON summaries under `results/twin/` in the repo. The 30-minute cron backup covers both.
- Cached models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B/3B/7B/14B-Instruct, Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen3-8B, Qwen3Guard-Gen-0.6B/4B/8B, granite-3.3-2b/8b-instruct, Skywork-Reward-V2-Qwen3-0.6B, and the PKU-SafeRLHF dataset. allenai/wildguard was not in the cache listing; `scripts/b4_wildguard.py` shows how it was loaded in 4-bit, so download it once (about 4 GB in 4-bit) before E3's evaluation step.
- PKU-SafeRLHF (73,907 rows): columns `prompt`, `response_0`, `response_1`, `is_response_0_safe`, `is_response_1_safe` (strings "True"/"False"), harm categories and severity. In a 2,000-row sample about 55 percent of rows have at least one unsafe response. Those unsafe responses are the twin's warm-start data.
- TRL 1.12.0, transformers 5.16.1. GRPOConfig defaults: `loss_type = "dapo"`, `scale_rewards = "group"`, `beta = 0.0`, `num_iterations = 1`, `importance_sampling_level = "token"`. For the twin use `beta = 0`, `scale_rewards = "none"`, and `loss_type = "dr_grpo"` (accepted values in 1.12: grpo, bnpo, dr_grpo, dapo, cispo, sapo, luspo; dr_grpo sums the per-token loss and divides by batch size times max completion length, a constant, so it keeps the summed identity up to a learning-rate scale; grpo and dapo average per sequence and per batch token, which do not). Reward functions receive `prompts`, `completions`, `completion_ids`, dataset columns, `trainer_state`, `log_extra`, `log_metric`; they do not receive log-probs, so the reward function computes log p_main and log q_old itself with the model object held in a closure (adapter switched to the main adapter for log p, twin adapter for log q), both under `torch.no_grad()`, summed over completion tokens.
- The existing `HFGRPOBackend` (`seldonian/llm/backend.py`) builds one LoRA adapter on the base and passes a reward callable; `extra_grpo_kwargs` reaches GRPOConfig. Two adapters on one base need `peft`'s `add_adapter` and `set_adapter`; the reference-model KL in TRL is off (`beta = 0`), so TRL never needs the adapter-disabled model.
- Judges: `build_judge("qwen3guard_4b")` is the training judge (4-bit, field safety, unsafe only). `Qwen3GuardJudge(field="refusal")` is the refusal check. Judge results are cached by prompt and response text in the judge's cache directory.

## Build plan for the trainer

1. `seldonian/llm/twin.py`: a class holding the base with two adapters, methods `seq_logprob(adapter, prompts, completions)`, `twin_reward(...)` implementing the registered formula with the trivial-output penalty, `train_twin(steps)`, `sample_pool(...)`, `defender_step(...)` with the weighted NPO loss and the helpfulness GRPO objective, and checkpoint save/load of both adapters.
2. `scripts/twin_e1.py` with stages `bruteforce` (20,000 responses for 200 prompts, judged, cached to D:), `warmstart`, `train` (300 steps, logging the per-prompt objective every 10 steps), `analyse` (E1a to E1d scored against the registration).
3. `scripts/run_twin_e1.sh`: free-space check, flock, `timeout 8h`, outputs on D:.
4. A smoke test on 4 prompts and 5 steps that prints timings and key names only.
5. Then E2, E3 (`scripts/twin_e3.py` with the five arms, one run per arm, seed and family, resumable), E4, E5. E5's plasmode half runs on the CPU first.

## Rules that apply

- No commits between 9:00 and 17:00 Pacific on weekdays; check `TZ=America/Los_Angeles date` first. Pushes are allowed at any time.
- Commit as Abdul Hannan Kanji with the Co-Authored-By and Claude-Session trailers.
- Registered fallbacks bind: a crashed or over-budget run is reported as lost; ask the author before a retry.
- A smoke test prints timings and key names only, never outcomes.
- Any change to a registered number after a run starts is a dated amendment in the registration file.
- Agent-written files get the residue scan before commit.

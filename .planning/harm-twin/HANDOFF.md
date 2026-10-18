# Handoff: executing the harmful-twin proposal

Written 2026-10-18 for a fresh session. Read this, the proposal (`reports/proposal_adversarial_harm_twin.md`) and the registration (`.planning/harm-twin/REGISTRATION-E1-E5.md`) before touching the GPU.

## Status

- Proposal draft 0.2 committed. The author's decisions of 2026-10-18 are in its section 10: separate paper from the Seldonian work; standalone defender with importance-weighted NPO plus helpfulness and a refusal check; hinge as a one-seed ablation; twin in the KL form.
- E0 done and passing (7 of 7): `scripts/twin_e0.py`, `results/twin/e0/report.md`. Finding folded into the proposal: the KL form with an indicator judge is robust to the coefficient; the entropy form is not.
- Registration for E1 to E5 committed with this handoff. Nothing on the GPU has run.
- Next: build the twin trainer, run its smoke test (timings and key names only, no outcomes), then E1 under the GPU lock.

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

---
spike: 021
idea: external-trace-certificate
name: agentdojo-seldonian-training
type: standard
validates: "Given AgentDojo's code-computed `security` label and a small local policy served through vLLM, when the policy is trained under a Seldonian constraint on a step-level code proxy of that label and certified on held-out user tasks with the clustered bound, then the real injection success rate falls to a certifiable level without losing task utility"
verdict: INVALIDATED
related: [014, 017, 020]
tags: [certificate, prompt-injection, agents, training, lagrangian, vllm, gpu]
---

# Spike 021: the full pipeline on a judge-free label (AgentDojo)

Design in `DESIGN.md` (written 2026-10-03 after the user's "do it next"; its gates, not a
second approval, decided whether stages B and C ran). **Both gates closed at stage A**, as
pre-registered, and the spike stops there with a result about the label, not the training.

## Investigation Trail
1. **Serving (2026-10-03, 14:30-15:55 PT).** vLLM 0.30 in its own venv on D: (the root disk
   filled during the first attempt; the uv cache moved to D:, then had to be disabled for the
   drvfs mount). Three obstacles, each fixed in `serve.sh`: Triton's JIT needs a C compiler
   (no gcc, no sudo: a `zig cc` shim at `/mnt/d/seldonian-runs/020/bin/gcc` that also
   rewrites `-l:libcuda.so.1`); the flashinfer sampler needs nvcc (`VLLM_USE_FLASHINFER_SAMPLER=0`);
   the harness sets no `max_tokens`, so a looping policy ran to the 16k context
   (`--override-generation-config '{"max_new_tokens": 512}'`). A comment inside a
   line-continued command silently broke the redirect once; `bash -n` does not catch that.
2. **Tool format.** AgentDojo's `local` provider parses `<function=...>` tags from text;
   Qwen2.5-1.5B produced none and the 3B wrote `function=get_iban` in a code fence, so every
   episode ended at the first turn with utility 0. Qwen's native `<tool_call>` format through
   vLLM's hermes parser and the `openai-compatible` provider works: multi-turn episodes with
   real calls. Training prompts are therefore rendered by the server (`POST /tokenize`) so the
   policy trains on exactly the text it is served with (`prefixes021.py`).
3. **Stage A, Qwen2.5-3B-Instruct** (`run.sh A`, 15:55-17:57 PT including a 23-minute wait
   for the lock and another session's job in between): the first pass lost the workspace suite
   and half of travel because a malformed tool-argument JSON from the policy killed the suite
   process and the second benchmark call overwrote the first's logs; `patch021.py` (tolerant
   argument parsing, per-episode exception guard scored the harness's own way) and separate
   log names fixed it, and the harness's skip-if-done made the second pass incremental. 879 of
   949 pairs completed; 70 workspace pairs hit the 16k context (long tool outputs) and have no
   result. GPU time about 1.9 h of the 12 h cap.
4. **Both gates closed.** H1's attack-rate floor failed at the 3B (the design's one allowed
   switch); H2's proxy-recall gate failed. Stages B and C did not run; `train021.py`,
   `stage_C.sh` and `bounds021.py` are written and untested on GPU.

## Results (stage A, untrained Qwen2.5-3B-Instruct, 879 pairs, 87 user tasks)

| quantity | value | asked |
|---|---|---|
| targeted attack success (real `security`) | 0.074 (65 of 879) | >= 0.20 (H1) |
| utility without attack | 0.195 | >= 0.15 (H1) |
| utility under attack | 0.135 | - |
| episodes where the policy never read the injected content | 419 of 879 | - |
| attack success among episodes that read it | 0.141 (65 of 460) | - |
| step-level proxy: agreement with the real label | 0.922 | >= 0.80 (H2) |
| step-level proxy: recall of real successes | 0.262 (17 of 65); loose 0.354 | >= 0.7; gate >= 0.5 (H2) |
| step-level proxy: precision | 0.447 | - |
| by suite | banking 0.090, slack 0.124, travel 0.029, workspace 0.066 | - |
| D_c / D_s (53 / 34 user tasks) | 0.091 / 0.047 | - |

- **The small policy is safe mostly by not working.** It acts before it reads: in 419 of 879
  episodes the injected tool result never appears in the conversation, so the injection
  cannot land. Its utility is 0.195 without attack. Where it does read the injection, one in
  seven episodes is compromised, which is the rate the training would have had to push.
- **The step-level proxy is not the label.** Of the 65 real successes, only 17 show the
  attacker's call at the turn right after the injection (23 with the loose match). In 42 the
  policy first makes a benign-looking call that serves the injection (`list_files`,
  `search_files`, `get_file_by_id`: gathering what the attacker asked for) and complies one
  or more turns later; the environment-state check sees it, a single-turn code check does
  not. A proxy that misses three quarters of the positives cannot steer a multiplier, which
  is why the gate exists. The 21 proxy fires without a real success are the mirror image:
  the attacker's function called with the wrong arguments or undone later.
- **What this means for the pipeline.** On this label the training signal has to be the
  label itself: the environment executed to the end of the episode inside the training
  loop (multi-turn GRPO with tool execution), not a single next-turn GRPO on prefixes. That
  is a different backend from `HFGRPOBackend`, and the first thing a follow-up has to build.
  The certificate half is unaffected: stage C's clustered bound on held-out user tasks is
  ready and was validated in 020.

### Scored expectations
| hypothesis | asked | found | result |
|---|---|---|---|
| H1 attackable and useful | rate >= 0.20, utility >= 0.15 | 0.074, 0.195 | rate fails (at the 3B, the one switch allowed); gate to B closed |
| H2 proxy tracks the label | agreement >= 0.80, recall >= 0.7 (gate 0.5) | 0.922, 0.262 | recall fails; gate to C closed |
| H3-H5 | - | not run | - |

**Verdict: INVALIDATED** for the design as written: a step-level code proxy of AgentDojo's
`security` label is not a training signal (recall 0.26), and a 3B policy is attackable at
only 7% because it rarely reads what it is attacked with. The stage-A infrastructure (vLLM
serving with native tool calls, exact prompt rendering, the robust harness run, the D_c / D_s
split by user task) stands and cost about 1.9 GPU-hours. The follow-up, if wanted, is an
episode-level backend: the environment in the loop, the real label as the constraint, and a
policy at least strong enough to read before it acts.

## How to Run
    cd .planning/spikes/021-agentdojo-seldonian-training
    CAP=14400 ./run.sh A Qwen/Qwen2.5-3B-Instruct base3b        # serve + attacked grid + no-attack grid + stageA.py + prefixes021.py (lock, timeout)
    # stages B and C (not run; gates closed):
    ../../../.venv/bin/python train021.py --tag b3b --tag-a base3b --pilot
    ./run.sh C Qwen/Qwen2.5-3B-Instruct /mnt/d/seldonian-runs/021/train/b3b/checkpoints/selected c3b
    ../../../.venv/bin/python bounds021.py --base base3b --trained c3b --train b3b

## Files
`DESIGN.md`, `serve.sh`, `bench.sh`, `patch021.py`, `run.sh`, `stage_A.sh`, `stage_C.sh`,
`stageA.py`, `prefixes021.py`, `train021.py`, `bounds021.py`, `stageA_base3b.md`, logs.
`results/spikes/021/`: `stageA_base3b.jsonl` (879 rows), `prefixes_base3b.jsonl.xz` (548
rendered prompts), `split.json`. Harness logs: `/mnt/d/seldonian-runs/021/runs_base3b/` (16 MB).
vLLM venv: `/mnt/d/seldonian-runs/020/vllm-venv` (7.7 GB).

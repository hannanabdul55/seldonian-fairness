#!/bin/bash
# Start vLLM (own venv) as the OpenAI-compatible server AgentDojo's `local` provider expects.
#   ./serve.sh <model> [--lora-modules name=path]   # blocks until /v1/models answers; prints the pid
MODEL=$1; shift
export HF_HOME=/mnt/d/hf-cache HF_HUB_CACHE=/mnt/d/hf-cache/hub HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
export CC=/mnt/d/seldonian-runs/020/bin/gcc PATH=/mnt/d/seldonian-runs/020/bin:$PATH   # zig cc shim for Triton (no system gcc, no sudo)
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer JIT needs nvcc, which this box lacks
VENV=/mnt/d/seldonian-runs/020/vllm-venv
# --enforce-eager: no torch.compile (Triton has no C compiler on this box; spike 014 hit the same)
mkdir -p /mnt/d/seldonian-runs/021/logs
nohup "$VENV/bin/vllm" serve "$MODEL" --port 8000 --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.88 --max-num-seqs 16 --enable-prefix-caching --seed 21 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --override-generation-config '{"max_new_tokens": 512}' "$@" \
  > /mnt/d/seldonian-runs/021/logs/vllm.log 2>&1 &
# (max_new_tokens: the harness sets no max_tokens, and a looping policy would otherwise run to the 16k context)
PID=$!
for i in $(seq 1 180); do
  if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then echo "$PID"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vllm died; see /mnt/d/seldonian-runs/021/logs/vllm.log" >&2; exit 1; fi
  sleep 5
done
echo "vllm did not come up in 15 min" >&2; kill "$PID"; exit 1

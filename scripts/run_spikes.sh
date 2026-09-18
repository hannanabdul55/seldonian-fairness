#!/usr/bin/env bash
# Spike queue (2026-09-18): worst-of-k sampling for the certified policy and the
# reference, then re-judging with a larger guard. Each job holds the shared GPU lock;
# the sampler resumes from its own output, so the queue can be relaunched.
set -u
cd "$(dirname "$0")/.."
LOCK=/tmp/claude-gpu.lock
export PYTORCH_CUDA_ALLOC_CONF=
job() {  # <name> <command...>
  local name=$1; shift
  echo "=== $(date '+%F %T') start $name"
  flock "$LOCK" -c "echo 'spikes pid=$$ $name' > $LOCK.info; $*"
  echo "=== $(date '+%F %T') end $name (rc $?)"
}
job wok_c0  uv run scripts/spike_worst_of_k.py sample --checkpoint auto --out results/spikes/worst_of_k/c0
job wok_ref uv run scripts/spike_worst_of_k.py sample --checkpoint none --out results/spikes/worst_of_k/ref
if [ -f scripts/spike_big_judge.py ]; then
  job big_judge uv run scripts/spike_big_judge.py judge
fi
echo "=== $(date '+%F %T') spikes complete"

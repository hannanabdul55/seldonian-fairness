#!/usr/bin/env bash
# Round 6 GPU queue, second half (reports/llm_round6_plan.md): Stages B2, B3, D, B4.
# Runs after scripts/run_round6.sh (B1 + C). Same settings: Stage A dual
# (lam0 5, floor 5, eta_down = eta), Clopper-Pearson, Round 5 trainer knobs.
# Holds the shared GPU lock per run; skips finished runs; safe to relaunch.
#
# Order: B2 (fixed-penalty frontier), B3 (marginal regime), D (DiscrimEval with the
# probability feature), B4 (ten-seed solution rate). B4 is last so the B1 gate
# (drift fixed?) can be read before it starts; touch results/llm_r6/B4_HOLD to skip it.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOCK=/tmp/claude-gpu.lock
S05=Qwen/Qwen2.5-0.5B-Instruct
DUAL="--lam0 5 --lam-floor 5 --lam-max 50 --eta 100"
BREV="--task brevity --n 3000 --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --word-cap 120 --long-margin 0.05"
DISC="--task discrim --n 3000 --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --decision-feature prob --attribute race --groups white Black"
REF5=results/llm_r5/v16/brevity   # Round 5 reference rates, seeds 0-2 (reference policy only)

run() {  # run <out> <task-args-var> <method> <seed> [extra args...]
  local out=$1 targs=$2 method=$3 seed=$4; shift 4
  local task; task=$(echo "$targs" | sed -n 's/.*--task \([a-z0-9]*\).*/\1/p')
  local dir="$out/$task/$method/seed$seed"
  if [ -f "$dir/result.json" ]; then echo "skip $out $task $method seed$seed"; return; fi
  mkdir -p "$out/logs"
  echo "=== $(date '+%F %T') start $out $task $method seed$seed $*"
  flock "$LOCK" bash -c "echo 'seldonian-fairness-c6 pid=$$ $out/$task/$method/seed$seed started $(date '+%T')' > $LOCK.info; \
    $PY scripts/run_llm_rl.py --method $method --seed $seed --model $S05 --out $out --quiet $targs $* > $out/logs/${task}_${method}_seed${seed}.log 2>&1"
  local rc=$?
  echo "=== $(date '+%F %T') end   $out $task $method seed$seed exit=$rc"
  if [ $rc -ne 0 ]; then echo "=== $(date '+%F %T') ABORT queue: run failed (see $out/logs/${task}_${method}_seed${seed}.log)"; exit $rc; fi
}
brev_ref() {  # brev_ref <out> <seed>: reuse a Round 5 reference-rate file when one exists
  mkdir -p "$1/brevity"
  cp --update=none "$REF5/reference_rates_seed$2.json" "$1/brevity/" 2>/dev/null || true
}

# C, penalty 4: run_round6.sh wrote both composite arms to the same directory, so its
# --lam 4 arm was skipped as "finished"; it runs here in its own directory with the
# Stage C reference rates (harm margin 0.045, adversarial group).
AB="--task ab --steps 150 --group-size 4 --predict-every 30 --predict-n 768 --steps-per-generation 4 --bound clopper_pearson --harm-margin 0.045 --harm-group adversarial"
for seed in 0 1 2; do
  out=results/llm_r6/c_l4; mkdir -p $out/ab
  cp --update=none results/llm_r6/c/ab/reference_rates_seed$seed.json $out/ab/ 2>/dev/null || true
  run $out "$AB" composite $seed --lam 4
done

# B1b: the always-on floor on the real task (synthetic: +9 pts solution rate at pressure 4
# for 3% reward; the armed floor left the first excursion unpenalised in B1)
out=results/llm_r6/b1a_v8; brev_ref $out 0
run $out "$BREV" seldonian_lag 0 --long-bonus 8 $DUAL --lam-floor-always

# B2: the fixed-penalty frontier at seed 0 (r5 has lam 16 at bonus 8, lam 16 / 32 at bonus 16)
for lam in 2 4 8 32; do
  out=results/llm_r6/b2_v8_l$lam; brev_ref $out 0
  run $out "$BREV" composite 0 --long-bonus 8 --lam $lam
done
for lam in 4 8; do
  out=results/llm_r6/b2_v16_l$lam; brev_ref $out 0
  run $out "$BREV" composite 0 --long-bonus 16 --lam $lam
done

# B3: the marginal regime, bonus 6 (between "shorter" at 4 and "breach" at 8)
out=results/llm_r6/b3_v6; brev_ref $out 0
run $out "$BREV" grpo          0 --long-bonus 6
run $out "$BREV" seldonian_lag 0 --long-bonus 6 $DUAL
run $out "$BREV" composite     0 --long-bonus 6 --lam 8

# D: DiscrimEval with P("yes") as the decision feature; reference first (does the paired
# interval collapse?), then the three trained arms under a bias bonus of 2 on white
out=results/llm_r6/d
run $out "$DISC" reference     0
run $out "$DISC" grpo          0 --bias-bonus 2
# bonus 2 did not bind (grpo parity 0.003 against a 0.05 threshold): a per-episode bonus
# below the within-group reward spread is absorbed, as on brevity. Trained arms at bonus 8.
out=results/llm_r6/d8; mkdir -p $out/discrim
cp --update=none results/llm_r6/d/discrim/reference_rates_seed0.json $out/discrim/ 2>/dev/null || true
cp -rn results/llm_r6/d/discrim/reference $out/discrim/ 2>/dev/null || true
run $out "$DISC" grpo          0 --bias-bonus 8
# bonus 8 on one group raised the yes rate of both groups to 0.997 (the policy does not
# condition on the race word): the pressure has to be differential to create a gap.
out=results/llm_r6/d8d; mkdir -p $out/discrim
cp --update=none results/llm_r6/d/discrim/reference_rates_seed0.json $out/discrim/ 2>/dev/null || true
cp -rn results/llm_r6/d/discrim/reference $out/discrim/ 2>/dev/null || true
run $out "$DISC" grpo          0 --bias-bonus 8 --bias-mode differential
run $out "$DISC" composite     0 --bias-bonus 8 --bias-mode differential --lam 4
run $out "$DISC" seldonian_lag 0 --bias-bonus 8 --bias-mode differential $DUAL

# B4: solution rate, seldonian_lag at bonus 8 with the always-on floor, seeds 1-9 (seed 0
# is B1b). Seeds 3-9 measure their own reference rates in-run (10 minutes each).
if [ -f results/llm_r6/B4_HOLD ]; then
  echo "=== $(date '+%F %T') B4 held (results/llm_r6/B4_HOLD exists)"
else
  out=results/llm_r6/b4_v8
  for seed in 1 2 3 4 5 6 7 8 9; do
    brev_ref $out $seed
    run $out "$BREV" seldonian_lag $seed --long-bonus 8 $DUAL --lam-floor-always
  done
fi
echo "=== $(date '+%F %T') ROUND6B DONE"

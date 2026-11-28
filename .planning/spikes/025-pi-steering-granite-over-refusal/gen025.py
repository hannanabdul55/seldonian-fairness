"""Spike 025: the PI steering rule on Granite over-refusal (GPU). DESIGN.md, approved 2026-11-28
with option (a): the same guard (Qwen3Guard-Gen-4B's refusal field) steers and certifies.

Spike 014's stage-2 recipe (``../014-pushed-label-stratification/gen014.py``) with one change:
``LagrangianReward`` is replaced by ``seldonian.llm.rewards.PIReward`` steered on a held-out
slice of 64 benign D_c prompts (one response each, judged every 5 steps, EMA a = 0.5). D_c, D_s,
the reward model, the judge, the GRPO settings and the safety test are 014's. No pool sampling
at checkpoints (014 needed it for its plasmode; 025 does not).

    ./run.sh --stage pilot                 # 20 steps, 100 pool prompts: step time and s0 -> c0
    ./run.sh --stage train --c0 <25 s0>    # seed 0, 200 steps, safety test on D_s
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "014-pushed-label-stratification"))
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
sys.path.insert(0, os.path.join(HERE, "..", "009-granite-transfer"))
import gen014  # noqa: E402

OUT = os.path.join(REPO, "results", "spikes", "025")
gen014.RUN_DIR = "/mnt/d/seldonian-runs/025"
MARGIN, DELTA = gen014.MARGIN, gen014.DELTA
SLICE_N, SLICE_EVERY, SLICE_EMA = 64, 5, 0.5
T, ETA_I, C0_PER_SD = 0.02, 10.0, 25.0
STEP_STOP_S = 90.0


class Recorder:
    """Wraps the base reward and keeps (prompt, score) so the pilot can measure s0."""

    def __init__(self, base):
        self.base, self.name, self.calls = base, base.name, []

    def __call__(self, prompts, responses, groups=None, references=None):
        out = np.asarray(self.base(prompts, responses, groups, references), dtype=float)
        self.calls.append((list(prompts), out.tolist()))
        return out


def pooled_within_prompt_sd(calls):
    var = []
    for prompts, scores in calls:
        by = {}
        for p, s in zip(prompts, scores):
            by.setdefault(p, []).append(s)
        var += [np.var(v, ddof=1) for v in by.values() if len(v) > 1]
    return float(np.sqrt(np.mean(var))) if var else float("nan")


def build(a, backend, pool, c0):
    from seldonian.llm.data import make_record
    from seldonian.llm.judges import build_judge
    from seldonian.llm.policy import Constraint, SeldonianLLMPolicy, effective_n
    from seldonian.llm.rewards import PIReward, SequenceClassifierReward
    ref_rate, n_ref = gen014.reference_rate()
    refusal = build_judge("qwen3guard_refusal", model_name="Qwen/Qwen3Guard-Gen-4B", quant4=True)
    c = Constraint("refusal", refusal, threshold=ref_rate + MARGIN, group="benign",
                   bound="clopper_pearson")
    recs = gen014.training_records(pool, a.n_adv, a.n_ben + SLICE_N)
    ben = [r for r in recs if r["group"] == "benign"]
    d_c = [r for r in recs if r["group"] == "adversarial"] + ben[:a.n_ben]   # 014's D_c
    slice_ = ben[a.n_ben:a.n_ben + SLICE_N]
    assert len(slice_) == SLICE_N and not {r["prompt"] for r in slice_} & {r["prompt"] for r in d_c}
    d_s = [dict(make_record(it["prompt"], "benign", "c1"), i=it["i"]) for it in pool]
    n_ben_pred = int(round(a.predict_n * a.n_ben / len(d_c)))
    base = Recorder(SequenceClassifierReward(gen014.RM))
    reward = PIReward(base, refusal, threshold=c.threshold, n=effective_n(n_ben_pred, len(d_s)),
                      delta=DELTA, group="benign", c0=c0, T=T, eta_i=ETA_I, ema=SLICE_EMA,
                      steer_on="slice")
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c], delta=DELTA,
                                predict_every=a.predict_every, predict_n=a.predict_n, seed=a.seed,
                                verbose=True)
    print(f"reference refusal {ref_rate:.4f} ({n_ref} prompts) -> threshold {c.threshold:.4f}; "
          f"D_c {len(d_c)}, slice {len(slice_)}, D_s {len(d_s)}, U at n={reward.n}, c0={c0}",
          flush=True)
    return policy, c, reward, refusal, slice_, base


def run(a, steps, tag, c0):
    os.makedirs(OUT, exist_ok=True)
    pool = gen014.pool_c1()
    if a.limit:
        pool = pool[:a.limit]
    backend = gen014.make_backend(steps, a.gen_batch, a.seed, tag)
    policy, c, reward, judge, slice_, base = build(a, backend, pool, c0)
    sp = [r["prompt"] for r in slice_]
    log, times, t0 = [], [], time.time()

    def read_slice(step):
        resp = backend.generate(sp, max_new_tokens=128)
        v = np.asarray(judge(sp, resp), dtype=float)
        rate = reward.observe(v)
        log.append(dict(step=step, slice_rate=float(v.mean()), rbar=rate))
        print(f"step {step}: steering slice refusal {v.mean():.3f}, rbar {rate:.3f}", flush=True)

    orig_train = backend.train

    def train_with_slice(records, rw, on_step):
        read_slice(0)                       # the reference policy's slice rate starts rbar
        last = [time.time()]

        def hook(step):
            on_step(step)
            now = time.time()
            times.append(now - last[0])
            last[0] = now
            if step % SLICE_EVERY == 0:
                read_slice(step)
                last[0] = time.time()       # slice time is not training step time
        return orig_train(records, rw, hook)

    backend.train = train_with_slice
    solution = policy.fit(seldonian=True)
    rep = policy.safety_report
    hist = [dict(step=h.step, feasible=bool(h.feasible), rates=h.rates, upper=h.upper,
                 reward=h.reward) for h in policy.history]
    out = dict(steps=steps, c0=c0, threshold=c.threshold, n_U=reward.n,
               selected=policy.selected, solution=solution is not None,
               safety_test=dict(g=rep.g, rates=rep.rates, upper=rep.upper, n=rep.n,
                                passed=rep.passed, reward=rep.reward),
               history=hist, pi_history=reward.history, slice=log, step_seconds=times,
               s0_first5=pooled_within_prompt_sd(base.calls[:5]),
               train_log=backend.train_log, seconds=time.time() - t0)
    json.dump(out, open(os.path.join(OUT, f"run_{tag}.json"), "w"), indent=1, default=float)
    recs, responses, _ = policy._safety_episodes
    with open(os.path.join(OUT, f"safety_{tag}.jsonl"), "w") as fh:
        for r, s in zip(recs, responses):
            fh.write(json.dumps(dict(i=r["i"], response=s)) + "\n")
    print(f"done: solution {solution is not None}, safety {rep.rates} upper {rep.upper} "
          f"<= {c.threshold:.3f}: {rep.passed}; {time.time() - t0:.0f}s", flush=True)
    return out


def stage_pilot(a):
    a.limit = 100
    out = run(a, 20, "pilot", c0=C0_PER_SD * 1.0)      # provisional c0; the pilot measures s0
    late = out["step_seconds"][10:]
    s0 = out["s0_first5"]
    verdict = dict(s0=s0, c0=C0_PER_SD * s0, median_step_s=float(np.median(out["step_seconds"])),
                   max_step_after_10=float(max(late)) if late else None,
                   stop=bool(late and np.median(late) > STEP_STOP_S))
    json.dump(verdict, open(os.path.join(OUT, "pilot_verdict.json"), "w"), indent=1)
    print("PILOT", json.dumps(verdict), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "train"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--c0", type=float)
    ap.add_argument("--n-adv", type=int, default=600)
    ap.add_argument("--n-ben", type=int, default=600)
    ap.add_argument("--predict-every", type=int, default=25)
    ap.add_argument("--predict-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "pilot":
        stage_pilot(a)
    else:
        if a.c0 is None:
            sys.exit("--c0 is required: 25 x the pilot's s0 (results/spikes/025/pilot_verdict.json)")
        run(a, a.steps, f"s{a.seed}", a.c0)


if __name__ == "__main__":
    main()

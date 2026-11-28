"""Spike 026: Lagrangian vs PI steering on Granite over-refusal with candidate data exchangeable
with the safety set (GPU). DESIGN.md (DRAFT until the user approves it).

Spike 025 found the predicted test, run on OR-Bench `ab` prompts, underestimated the safety-set
rate on 013's C1 pool (XSTest + OR-Bench) in every run. Here the C1 pool itself is split per seed,
stratified by source, into three disjoint parts:

- T (124 benign): trained on, with 125 adversarial `ab` prompts (the 50/50 mix of 014);
- H (126 benign): never trained on; every predicted test runs on all of H (one response each),
  and PI's steering slice is 64 prompts of H read every 5 steps;
- D_s (250 benign): the safety test, touched once.

Everything else is 014's stage-2 recipe; the two arms differ only in the reward.

    ./run.sh --arm lagrangian --seed 0
    ./run.sh --arm pi --seed 0
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

OUT = os.path.join(REPO, "results", "spikes", "026")
gen014.RUN_DIR = "/mnt/d/seldonian-runs/026"
MARGIN, DELTA = gen014.MARGIN, gen014.DELTA
N_T, N_H = 125, 125
SLICE_N, SLICE_EVERY, SLICE_EMA = 64, 5, 0.5
C0, T, ETA_I = 38.526, 0.02, 10.0          # c0 = 25 x s0 from spike 025's pilot


def split_pool(pool, seed):
    """Per-source random split of the C1 pool into T, H and D_s (124 / 126 / 250)."""
    rng = np.random.default_rng([seed, 26])
    T_, H_, S_ = [], [], []
    for src in sorted({it["meta"] for it in pool}):
        items = [it for it in pool if it["meta"] == src]
        idx = rng.permutation(len(items))
        a, b = len(items) // 4, len(items) // 2
        T_ += [items[i] for i in idx[:a]]
        H_ += [items[i] for i in idx[a:b]]
        S_ += [items[i] for i in idx[b:]]
    return T_, H_, S_


def build(a, backend, pool):
    from seldonian.llm.data import make_record
    from seldonian.llm.judges import build_judge
    from seldonian.llm.policy import Constraint, SeldonianLLMPolicy, effective_n
    from seldonian.llm.rewards import LagrangianReward, PIReward, SequenceClassifierReward
    ref_rate, n_ref = gen014.reference_rate()
    refusal = build_judge("qwen3guard_refusal", model_name="Qwen/Qwen3Guard-Gen-4B", quant4=True)
    c = Constraint("refusal", refusal, threshold=ref_rate + MARGIN, group="benign",
                   bound="clopper_pearson")
    T_, H_, S_ = split_pool(pool, a.seed)
    rec = lambda it, src: dict(make_record(it["prompt"], "benign", src), i=it["i"])  # noqa: E731
    adv = [r for r in gen014.training_records(pool, N_T, 0) if r["group"] == "adversarial"][:N_T]
    d_c = adv + [rec(it, "c1") for it in T_]
    held = [rec(it, "c1") for it in H_]
    d_s = [rec(it, "c1") for it in S_]
    assert not ({r["prompt"] for r in d_c} & {r["prompt"] for r in held + d_s}
                or {r["prompt"] for r in held} & {r["prompt"] for r in d_s})
    base = SequenceClassifierReward(gen014.RM)
    n_U = effective_n(len(held), len(d_s))
    if a.arm == "lagrangian":
        reward = LagrangianReward(base, [(refusal, "benign")], names=["refusal"], lam0=5.0,
                                  eta=100.0, lam_max=50.0, lam_floor=5.0)
    else:
        reward = PIReward(base, refusal, threshold=c.threshold, n=n_U, delta=DELTA,
                          group="benign", c0=C0, T=T, eta_i=ETA_I, ema=SLICE_EMA,
                          steer_on="slice")

    class HeldOutPolicy(SeldonianLLMPolicy):
        """Every predicted test runs on all of H (never trained on, exchangeable with D_s)."""

        def _candidate_subset(self, n):
            return list(held)

    policy = HeldOutPolicy(backend, d_c, d_s, reward=reward, constraints=[c], delta=DELTA,
                           predict_every=a.predict_every, predict_n=len(held), seed=a.seed,
                           verbose=True)
    print(f"arm {a.arm} seed {a.seed}: reference {ref_rate:.4f} -> threshold {c.threshold:.4f}; "
          f"D_c {len(d_c)} ({len(T_)} benign), H {len(held)}, D_s {len(d_s)}, n_U {n_U}", flush=True)
    return policy, c, reward, refusal, held


def run(a):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.arm}_s{a.seed}"
    backend = gen014.make_backend(a.steps, a.gen_batch, a.seed, tag)
    policy, c, reward, judge, held = build(a, backend, gen014.pool_c1())
    rng = np.random.default_rng([a.seed, 261])
    sp = [held[i]["prompt"] for i in rng.choice(len(held), SLICE_N, replace=False)]
    log, times, t0 = [], [], time.time()

    def read_slice(step):
        resp = backend.generate(sp, max_new_tokens=128)
        v = np.asarray(judge(sp, resp), dtype=float)
        rate = reward.observe(v)
        log.append(dict(step=step, slice_rate=float(v.mean()), rbar=rate))
        print(f"step {step}: steering slice refusal {v.mean():.3f}, rbar {rate:.3f}", flush=True)

    orig_train = backend.train

    def train_hooked(records, rw, on_step):
        if a.arm == "pi":
            read_slice(0)
        last = [time.time()]

        def hook(step):
            on_step(step)
            now = time.time()
            times.append(now - last[0])
            last[0] = now
            if a.arm == "pi" and step % SLICE_EVERY == 0:
                read_slice(step)
                last[0] = time.time()
        return orig_train(records, rw, hook)

    backend.train = train_hooked
    solution = policy.fit(seldonian=True)
    rep = policy.safety_report
    hist = [dict(step=h.step, feasible=bool(h.feasible), rates=h.rates, upper=h.upper,
                 reward=h.reward, lambdas=getattr(h, "lambdas", None)) for h in policy.history]
    out = dict(arm=a.arm, seed=a.seed, steps=a.steps, threshold=c.threshold,
               selected=policy.selected, solution=solution is not None,
               safety_test=dict(g=rep.g, rates=rep.rates, upper=rep.upper, n=rep.n,
                                passed=rep.passed, reward=rep.reward),
               history=hist, pi_history=getattr(reward, "history", None), slice=log,
               step_seconds=times, train_log=backend.train_log, seconds=time.time() - t0)
    json.dump(out, open(os.path.join(OUT, f"run_{tag}.json"), "w"), indent=1, default=float)
    recs, responses, _ = policy._safety_episodes
    with open(os.path.join(OUT, f"safety_{tag}.jsonl"), "w") as fh:
        for r, s in zip(recs, responses):
            fh.write(json.dumps(dict(i=r["i"], response=s)) + "\n")
    print(f"done: {tag} solution {solution is not None}, safety {rep.rates} upper {rep.upper} "
          f"<= {c.threshold:.3f}: {rep.passed}; {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lagrangian", "pi"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--predict-every", type=int, default=25)
    ap.add_argument("--gen-batch", type=int, default=64)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

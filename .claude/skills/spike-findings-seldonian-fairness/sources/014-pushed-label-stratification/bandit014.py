"""Spike 014 stage 1: the stratification gain when the Lagrangian pushes the label (bandit).

Spike 013's in-loop harness (``013/inloop.py``: the real ``SeldonianLLMPolicy`` with
Lagrangian GRPO on ``HeteroEnv``, reference-rate strata, ``b1w`` bounds) with the knobs
DESIGN.md section 4 names, each defaulting to 013's value:

- ``pressure``  the reward's weight on the violating action (013: 1.0);
- ``eta``       the dual step on a predicted violation (013: 100);
- ``margin``    threshold = reference rate + margin (013: +0.03; negative pushes the rate
                below the reference);
- ``steps``     training steps (013: 200);
- ``method``    ``seldonian_lag`` (013) or ``grpo`` (no constraint: the side-effect control).

Rules: ``random``, ``strat_ref`` (8 equal rank strata of an 8-sample reference rate),
``placebo`` (strata of a permuted reference rate). Every bound is computed at delta 0.05 and
0.1 from the same safety labels. Per run: the pool and population truths, ICC_ref, ICC_cand,
rho, the rate moved, feasibility, and every bound.

    OMP_NUM_THREADS=1 ../../../.venv/bin/python bandit014.py --seeds 300 --out bandit.json
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "013-stratified-safety-set"))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))

import inloop as IL  # noqa: E402
from heteroenv import HeteroEnv  # noqa: E402
from tdlab import (DEFAULTS, Constraint, LagrangianReward, SeldonianLLMPolicy,  # noqa: E402
                   SyntheticJudge, SyntheticReward, TDBackend)

H, K_REF = 8, 8
DELTAS = (0.05, 0.1)
C_H = IL.C_H


def run(seed, env, rule, pressure=1.0, eta=100.0, margin=0.03, steps=200,
        method="seldonian_lag"):
    t0 = time.time()
    cfg = dict(DEFAULTS, pressure=pressure, eta=eta, margin=margin, steps=steps)
    env_key = env
    u_scale, shared = IL.ENVS[env_key]
    env = HeteroEnv(cfg["population"], shared=shared, d=cfg["d"], pressure=pressure,
                    seed=seed, u_scale=u_scale)
    records = env.records(cfg["n"], seed=seed)
    idx = env.indices(records)
    N = len(idx)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = TDBackend(env, max_steps=steps, group_size=cfg["group_size"],
                        prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                        beta=cfg["beta"], seed=seed + 2)
    ref = backend.params
    crng = np.random.default_rng([seed, 7])
    _, v = env.sample(ref, np.repeat(idx, K_REF), crng)
    r_ref = v.reshape(N, K_REF).mean(axis=1)
    srng = np.random.default_rng([seed, 12])
    if rule == "random":
        strata, m = None, np.zeros(N, dtype=bool)
        m[srng.permutation(N)[:int(round(IL.TEST_SIZE * N))]] = True
    elif rule == "strat_ref":
        strata = IL.quantile_strata(r_ref, H, srng)
        m = IL.blocked(strata, srng)
    elif rule == "placebo":
        strata = IL.quantile_strata(srng.permutation(r_ref), H, srng)
        m = IL.blocked(strata, srng)
    else:
        raise ValueError(rule)
    d_c = [records[i] for i in np.flatnonzero(~m)]
    d_s = [records[i] for i in np.flatnonzero(m)]
    ref_rate = float(env.true_rate(ref))
    threshold = ref_rate + margin
    c = Constraint("harm", judge, threshold=threshold, group=None, bound=cfg["bound"])
    if method == "seldonian_lag":
        reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=cfg["lam0"],
                                  eta=eta, lam_max=cfg["lam_max"])
    else:
        reward = base
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                delta=DELTAS[1], predict_every=cfg["predict_every"],
                                predict_n=cfg["predict_n"], seed=seed)
    if method == "grpo":
        policy.fit(seldonian=False)
        solution = None
        policy._safetyTest()                     # the same one-shot draw on D_s, unconstrained
    else:
        solution = policy.fit(seldonian=True) is not None
    params = backend.params
    recs, responses, _ = policy._safety_episodes
    labels = np.asarray(judge([r["prompt"] for r in recs], responses), dtype=float)
    pos = {int(i): k for k, i in enumerate(idx)}
    s_rows = np.array([pos[int(i)] for i in env.indices(recs)])
    if strata is not None:
        W = np.array([np.mean(strata == h) for h in range(strata.max() + 1)])
        st = strata[s_rows]
    else:
        W, st = None, None
    b = {}
    for d in DELTAS:
        b.update({f"{k}@{d}": v for k, v in IL.bounds_for(labels, st, W, d, N).items()})
    p_cand = np.sum(env.probs(params, idx) * env.p_v[idx], axis=1)
    p_ref = np.sum(env.probs(ref, idx) * env.p_v[idx], axis=1)
    mu = p_cand.mean()
    icc_ref = p_ref.var() / (p_ref.mean() * (1 - p_ref.mean()))
    lam = getattr(reward, "lambdas", {}).get("harm") if method == "seldonian_lag" else None
    return dict(
        seed=seed, env=env_key, rule=rule, pressure=pressure, eta=eta, margin=margin,
        steps=steps, method=method, threshold=float(threshold), ref_rate=ref_rate,
        predicted_feasible=bool(any(h.feasible for h in policy.history)) if policy.history else None,
        solution=solution, truth_pool=float(mu), truth_pop=float(env.true_rate(params)),
        icc_cand=float(p_cand.var() / (mu * (1 - mu))), icc_ref=float(icc_ref),
        rho=float(np.corrcoef(p_ref, p_cand)[0, 1]), lam_final=None if lam is None else float(lam),
        n_s=int(m.sum()), N=N, seconds=time.time() - t0, **b)


def _job(a):
    return run(**a)


def cells(exploratory=True):
    """DESIGN.md section 4: the core factorial, the control, the placebo, the exploratory knobs."""
    out = []
    envs = ("icc26", "icc55", "icc75")
    for env, p, mg in itertools.product(envs, (0.5, 1.0, 2.0, 4.0), (0.03, -0.03, -0.06)):
        for rule in ("random", "strat_ref"):
            out.append(dict(env=env, rule=rule, pressure=p, eta=400.0, margin=mg))
    for env, p in itertools.product(envs, (0.5, 1.0, 2.0, 4.0)):           # side-effect control
        for rule in ("random", "strat_ref"):
            out.append(dict(env=env, rule=rule, pressure=p, method="grpo"))
    for env, p, mg in itertools.product(("icc55", "icc75"), (2.0, 4.0), (-0.03, -0.06)):
        out.append(dict(env=env, rule="placebo", pressure=p, eta=400.0, margin=mg))
    if exploratory:
        for env, p in itertools.product(("icc55", "icc75"), (1.0, 4.0)):
            for rule in ("random", "strat_ref"):
                out.append(dict(env=env, rule=rule, pressure=p, eta=100.0, margin=-0.03))
                out.append(dict(env=env, rule=rule, pressure=p, eta=400.0, margin=-0.03, steps=400))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 6 cells, no pool")
    a = ap.parse_args()
    cs = cells()
    if a.smoke:
        cs, a.seeds = cs[:4] + [c for c in cs if c.get("method") == "grpo"][:1] + \
            [c for c in cs if c["rule"] == "placebo"][:1], 2
    jobs = [dict(c, seed=s) for c in cs for s in range(a.seeds)]
    print(f"{len(cs)} cells x {a.seeds} seeds = {len(jobs)} runs", flush=True)
    t0 = time.time()
    rows = []
    if a.smoke:
        for j in jobs:
            rows.append(run(**j))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                   if k in ("env", "rule", "pressure", "margin", "method", "ref_rate", "truth_pool",
                            "icc_cand", "icc_ref", "rho", "solution", "seconds")}, flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(_job, jobs, chunksize=8)):
                rows.append(r)
                if (i + 1) % 1000 == 0:
                    el = time.time() - t0
                    print(f"{i + 1}/{len(jobs)} {el:.0f}s eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s",
                          flush=True)
                    json.dump(rows, open(a.out, "w"))
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s -> {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

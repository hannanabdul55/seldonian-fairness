"""Spike 013 stage 0b: plasmode resampling on the bandit, where the truth is exact.

For each environment and seed: train one candidate with the real pipeline (random split,
Lagrangian GRPO), then resample thousands of safety sets from its pool with ``plasmode``.
Candidates: the trained policy, and the reference itself (rho = 1, positive control). The
same resampler runs on real generations in stage 3; here it is checked against exact truth
and the pre-flight prediction G (hypothesis H3).

    ../../../.venv/bin/python bandit_plasmode.py --seeds 10 --reps 1000 --out bandit_plasmode.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))

from tdlab import (DEFAULTS, Constraint, LagrangianReward, SeldonianLLMPolicy,  # noqa: E402
                   SyntheticJudge, SyntheticReward, TDBackend, split_prompts)

import plasmode as PM  # noqa: E402
from heteroenv import HeteroEnv  # noqa: E402
from inloop import ENVS, K_REF  # noqa: E402


def train(seed, env_key):
    cfg = dict(DEFAULTS)
    u_scale, shared = ENVS[env_key]
    env = HeteroEnv(cfg["population"], shared=shared, d=cfg["d"], pressure=cfg["pressure"],
                    seed=seed, u_scale=u_scale)
    records = env.records(cfg["n"], seed=seed)
    d_c, d_s = split_prompts(records, test_size=0.4, seed=seed)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    backend = TDBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                        prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                        beta=cfg["beta"], seed=seed + 2)
    ref = backend.params
    c = Constraint("harm", judge, threshold=env.true_rate(ref) + cfg["margin"], group=None)
    reward = LagrangianReward(SyntheticReward(env, seed=seed + 1), [(judge, None)],
                              names=["harm"], lam0=cfg["lam0"], eta=cfg["eta"],
                              lam_max=cfg["lam_max"])
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                delta=cfg["delta"], predict_every=cfg["predict_every"],
                                predict_n=cfg["predict_n"], seed=seed)
    policy.fit(seldonian=True)
    return env, env.indices(records), ref, backend.params


def job(args):
    seed, env_key, reps = args
    t0 = time.time()
    env, idx, ref, params = train(seed, env_key)
    crng = np.random.default_rng([seed, 7])
    _, v = env.sample(ref, np.repeat(idx, K_REF), crng)
    cov = v.reshape(len(idx), K_REF).astype(float)
    out = []
    for cand, prm in (("trained", params), ("reference", ref)):
        p = np.sum(env.probs(prm, idx) * env.p_v[idx], axis=1)

        def draw(ids, rng, p=p):
            return (rng.random(len(ids)) < p[ids]).astype(float)

        pool = dict(cov=cov, draw=draw, p_truth=p, meta=env.group_idx[idx])
        pre = {H: PM.preflight(cov, p, k=k, H=H) for H in (2, 4, 8) for k in (8,)}
        pre_k = {k: PM.preflight(cov, p, k=k, H=4) for k in (1, 2, 4, 8)}
        rows = PM.run_cells(pool, PM.standard_cells(), reps, seed=seed)
        for r in rows:
            g = PM.preflight(cov, p, k=r["k"], H=r["H"]) if r["arm"] == "S2" else pre_k[8]
            out.append(dict(r, seed=seed, env=env_key, cand=cand, **{f"pf_{k}": v for k, v in g.items()}))
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(s, e, a.reps) for e in ENVS for s in range(a.seeds)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, dt in ex.map(job, jobs):
            rows.extend(r)
    json.dump(rows, open(a.out, "w"))
    print(f"{len(jobs)} candidates x2, {len(rows)} rows in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()

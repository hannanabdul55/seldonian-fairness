"""Spike 013 stage 0a: in-loop validity of reference-rate stratified safety sets.

The real SeldonianLLMPolicy (Lagrangian GRPO, spike 001's tdlab backend) trains on D_c of
each split, on ``HeteroEnv`` at four heterogeneity levels (ICC_ref 0.05-0.75). The safety
episodes the policy draws are then scored with every bound, pooled and stratified, against
two targets: the pool's own rate and the population the pool was drawn from (two-phase).

Rules: ``random``; ``strat_ref`` (quantile strata of an 8-sample reference rate, H = 4);
``strat_meta`` (the prompt's group, 2 strata); ``placebo`` (strata of a permuted reference
rate). Bounds: the project's ``ttest``; ``b1`` (stratified Wald-t) and ``b1w`` (stratified
Wilson-type) from ``stratbounds.py``, each also at one stratum (pooled).

    ../../../.venv/bin/python inloop.py --seeds 500 --out inloop.json
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
                   SyntheticJudge, SyntheticReward, TDBackend)
from seldonian.bounds import ttest_bounds  # noqa: E402

import stratbounds as SB  # noqa: E402
from heteroenv import HeteroEnv  # noqa: E402

TEST_SIZE = 0.4
K_REF = 8
H = 4
ENVS = {"icc05": (0.5, 0.0), "icc26": (1.0, 0.5), "icc55": (2.0, 0.8), "icc75": (3.0, 1.0)}
RULES = ["random", "strat_ref", "strat_meta", "placebo"]
C_H = {1: 0.0, 2: 2 / np.pi, 3: 0.81, 4: 0.88, 5: 0.91, 8: 0.96}


def quantile_strata(x, H, rng):
    """H near-equal strata by rank of x (ties broken at random, so shares are exact)."""
    order = np.lexsort((rng.random(len(x)), x))
    out = np.empty(len(x), dtype=int)
    out[order] = np.arange(len(x)) * H // len(x)
    return out


def blocked(cells, rng):
    m = np.zeros(len(cells), dtype=bool)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        m[rng.permutation(idx)[:int(round(TEST_SIZE * len(idx)))]] = True
    return m


def bounds_for(labels, strata, W, delta, N):
    """All bounds on the safety labels: pooled at one stratum, stratified with W."""
    s_all, n_all = labels.sum(), len(labels)
    out = {"est_pool": float(labels.mean()),
           "ttest": float(ttest_bounds(labels, delta).upper),
           "b1_pooled": SB.b1(s_all, n_all, 1.0, delta),
           "b1w_pooled": SB.b1w(s_all, n_all, 1.0, delta)}
    if strata is not None:
        hs = np.arange(len(W))
        s = np.array([labels[strata == h].sum() for h in hs])
        n = np.array([(strata == h).sum() for h in hs])
        out["est_strat"] = SB.estimate(s, n, W)
        for k, fn in (("b1", SB.b1), ("b1w", SB.b1w)):
            out[f"{k}_strat_pool"] = fn(s, n, W, delta)
            out[f"{k}_strat_pop"] = fn(s, n, W, delta, N=N)
    return out


def run(seed, env_key, rule, margin=0.03):
    t0 = time.time()
    cfg = dict(DEFAULTS, margin=margin)
    u_scale, shared = ENVS[env_key]
    env = HeteroEnv(cfg["population"], shared=shared, d=cfg["d"], pressure=cfg["pressure"],
                    seed=seed, u_scale=u_scale)
    records = env.records(cfg["n"], seed=seed)
    idx = env.indices(records)
    N = len(idx)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = TDBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                        prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                        beta=cfg["beta"], seed=seed + 2)
    ref = backend.params
    # covariate: K_REF separate reference samples per prompt, before any split
    crng = np.random.default_rng([seed, 7])
    _, v = env.sample(ref, np.repeat(idx, K_REF), crng)
    r_ref = v.reshape(N, K_REF).mean(axis=1)
    srng = np.random.default_rng([seed, 12])
    if rule == "random":
        strata, m = None, np.zeros(N, dtype=bool)
        m[srng.permutation(N)[:int(round(TEST_SIZE * N))]] = True
    elif rule == "strat_ref":
        strata = quantile_strata(r_ref, H, srng)
        m = blocked(strata, srng)
    elif rule == "placebo":
        strata = quantile_strata(srng.permutation(r_ref), H, srng)
        m = blocked(strata, srng)
    elif rule == "strat_meta":
        strata = env.group_idx[idx]
        m = blocked(strata, srng)
    else:
        raise ValueError(rule)
    d_c = [records[i] for i in np.flatnonzero(~m)]
    d_s = [records[i] for i in np.flatnonzero(m)]
    threshold = env.true_rate(ref) + margin
    c = Constraint("harm", judge, threshold=threshold, group=None, bound=cfg["bound"])
    reward = LagrangianReward(base, [(judge, None)], names=["harm"], lam0=cfg["lam0"],
                              eta=cfg["eta"], lam_max=cfg["lam_max"])
    policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                delta=cfg["delta"], predict_every=cfg["predict_every"],
                                predict_n=cfg["predict_n"], seed=seed)
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
    b = bounds_for(labels, st, W, cfg["delta"], N)
    # ground truth and the pre-flight ingredients (exact, over the pool)
    p_cand = np.sum(env.probs(params, idx) * env.p_v[idx], axis=1)
    p_ref = np.sum(env.probs(ref, idx) * env.p_v[idx], axis=1)
    mu = p_cand.mean()
    icc_ref = p_ref.var() / (p_ref.mean() * (1 - p_ref.mean()))
    return dict(
        seed=seed, env=env_key, rule=rule, threshold=float(threshold),
        predicted_feasible=bool(any(h.feasible for h in policy.history)),
        solution_ttest=bool(solution), truth_pool=float(mu), truth_pop=float(env.true_rate(params)),
        icc_cand=float(p_cand.var() / (mu * (1 - mu))), icc_ref=float(icc_ref),
        rho=float(np.corrcoef(p_ref, p_cand)[0, 1]), n_s=int(m.sum()), N=N, **b,
        seconds=time.time() - t0)


def _job(a):
    return run(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(s, e, r) for e in a.envs.split(",") for r in a.rules.split(",")
            for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()

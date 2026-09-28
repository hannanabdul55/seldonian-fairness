"""Spike 012, part B: split rules for an LLM-shaped Seldonian run on the synthetic bandit.

The pipeline is ``tdlab.run`` (spike 001: SeldonianLLMPolicy, Lagrangian GRPO, t-test bound,
threshold = reference rate + margin) with the candidate/safety split swapped. The covariates
are the ones an LLM run has before training: the prompt's group (metadata) and the reference
policy's per-prompt violation rate, estimated from ``REF_SAMPLES`` separate samples.

The safety test's episodes are also scored with a post-stratified bound (strata = group x
reference-rate quintile, weights = the strata's shares of the whole pool), to see whether a
bound that knows the strata turns the balance into power.

    ../../../.venv/bin/python banditsplit.py --seeds 1000 --out bandit.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import chi2, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "001-grpo-advantage-vs-td"))

import tdlab  # noqa: E402
from tdlab import (DEFAULTS, LagrangianReward, SeldonianLLMPolicy, Constraint,  # noqa: E402
                   SyntheticEnv, SyntheticJudge, SyntheticReward, TDBackend, split_prompts)

TEST_SIZE = 0.4
REF_SAMPLES = 4
N_STRATA_Q = 5


def ref_rates(env, ref, idx, rng):
    """(exact, estimated from REF_SAMPLES samples) reference violation rate per prompt."""
    exact = np.sum(env.probs(ref, idx) * env.p_v[idx], axis=1)
    reps = np.repeat(idx, REF_SAMPLES)
    _, v = env.sample(ref, reps, rng)
    return exact, v.reshape(len(idx), REF_SAMPLES).mean(axis=1)


def strata_of(group, ref_est):
    """group x quintile of the estimated reference rate (quintiles within each group)."""
    out = np.zeros(len(group), dtype=int)
    for g in np.unique(group):
        m = group == g
        q = np.quantile(ref_est[m], np.linspace(0, 1, N_STRATA_Q + 1)[1:-1])
        out[m] = g * N_STRATA_Q + np.searchsorted(q, ref_est[m], side="right")
    return out


def _perm_mask(n, rng):
    m = np.zeros(n, dtype=bool)
    m[rng.permutation(n)[:int(round(TEST_SIZE * n))]] = True
    return m


def _blocked(cells, rng):
    m = np.zeros(len(cells), dtype=bool)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        m[rng.permutation(idx)[:int(round(TEST_SIZE * len(idx)))]] = True
    return m


def split_mask(rule, cov, rng):
    """Safety mask over the pool for one split rule; returns (mask, tries)."""
    n = len(cov["group"])
    if rule == "random":
        return _perm_mask(n, rng), 1
    if rule == "strat_group":
        return _blocked(cov["group"], rng), 1
    if rule == "strat_ref":
        return _blocked(cov["strata"], rng), 1
    if rule == "maha_ref":
        Z = np.column_stack([cov["group"], cov["ref_est"]]).astype(float)
        Z = Z - Z.mean(axis=0)
        ns = int(round(TEST_SIZE * n))
        ci = np.linalg.pinv(np.cov(Z, rowvar=False) * (1 / ns + 1 / (n - ns)))
        thr = chi2.ppf(0.1, 2)
        for tries in range(1, 5001):
            m = _perm_mask(n, rng)
            d = Z[m].mean(axis=0) - Z[~m].mean(axis=0)
            if d @ ci @ d <= thr:
                return m, tries
        return m, tries
    if rule == "alg1_ref":
        # the 2020 Algorithm 1 with theta_s = the reference policy: best of 30 splits on
        # |mean one-shot reference label on D_c - on D_s|
        lab = cov["ref_label"]
        best, best_d = None, np.inf
        for _ in range(30):
            m = _perm_mask(n, rng)
            d = abs(lab[m].mean() - lab[~m].mean())
            if d < best_d:
                best, best_d = m, d
        return best, 30
    raise ValueError(rule)


RULES = ["random", "strat_group", "strat_ref", "maha_ref", "alg1_ref"]


def poststrat_upper(labels, strata, weights, delta):
    """Post-stratified mean + t-quantile * stratified se (pool shares as weights)."""
    est, var, dof = 0.0, 0.0, 0
    for h, w in weights.items():
        x = labels[strata == h]
        if len(x) < 2:
            return np.inf
        est += w * x.mean()
        var += w * w * x.var(ddof=1) / len(x)
        dof += len(x) - 1
    return est + np.sqrt(var) * tdist.ppf(1 - delta, dof)


def run(seed, rule, margin=0.03, u_scale=0.5, ref_exact=False):
    """``u_scale`` sets how much the violation rate varies across prompts (env default 0.5);
    ``ref_exact`` balances on the exact reference rate instead of the 4-sample estimate."""
    t0 = time.time()
    cfg = dict(DEFAULTS, margin=margin)
    env = SyntheticEnv(n_contexts=cfg["population"], d=cfg["d"], pressure=cfg["pressure"],
                       seed=seed, u_scale=u_scale)
    records = env.records(cfg["n"], seed=seed)
    idx = env.indices(records)
    judge = SyntheticJudge(1.0, 1.0, seed=seed)
    base = SyntheticReward(env, seed=seed + 1)
    backend = TDBackend(env, max_steps=cfg["steps"], group_size=cfg["group_size"],
                        prompts_per_step=cfg["prompts_per_step"], lr=cfg["lr"],
                        beta=cfg["beta"], seed=seed + 2)
    ref = backend.params
    # covariates fixed before the split (common to every rule for this seed)
    crng = np.random.default_rng([seed, 7])
    exact, est = ref_rates(env, ref, idx, crng)
    _, one = env.sample(ref, idx, crng)
    group = env.group_idx[idx]
    if ref_exact:
        est = exact
    cov = dict(group=group, ref_est=est, ref_label=one.astype(float),
               strata=strata_of(group, est))
    m, tries = split_mask(rule, cov, np.random.default_rng([seed, 12]))
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
    true_rate = env.true_rate(params)
    rep = policy.safety_report
    # post-stratified bound on the same safety episodes
    recs, responses, _ = policy._safety_episodes
    labels = np.asarray(judge([r["prompt"] for r in recs], responses), dtype=float)
    s_idx = env.indices(recs)
    pos = {int(i): k for k, i in enumerate(idx)}
    st = cov["strata"][[pos[int(i)] for i in s_idx]]
    weights = {h: float(np.mean(cov["strata"] == h)) for h in np.unique(cov["strata"])}
    ps_upper = poststrat_upper(labels, st, weights, cfg["delta"])
    pred = [h for h in policy.history if h.feasible]
    return dict(
        seed=seed, rule=rule, margin=margin, u_scale=u_scale, ref_exact=ref_exact, tries=tries, threshold=float(threshold),
        solution=bool(solution), predicted_feasible=bool(pred),
        true_rate=float(true_rate), violates=bool(true_rate > threshold),
        miss=bool(solution and true_rate > threshold),
        rate_s=float(rep.rates["harm"]), upper_s=float(rep.upper["harm"]),
        err_s=float(rep.rates["harm"] - true_rate),
        comp_err=float(env.true_rate(params, d_s) - true_rate),
        ref_imbalance=float(exact[m].mean() - exact[~m].mean()),
        ps_upper=float(ps_upper), ps_pass=bool(ps_upper <= threshold),
        ps_miss=bool(ps_upper <= threshold and true_rate > threshold),
        ps_uncovered=bool(true_rate > ps_upper), uncovered=bool(true_rate > rep.upper["harm"]),
        true_reward=float(env.true_reward(params)), seconds=time.time() - t0)


def _job(a):
    return run(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--u-scale", type=float, default=0.5)
    ap.add_argument("--ref-exact", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(s, r, a.margin, a.u_scale, a.ref_exact) for r in a.rules.split(",")
            for s in range(a.seed0, a.seed0 + a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=10))
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()

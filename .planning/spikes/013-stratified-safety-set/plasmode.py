"""Plasmode resampling for spike 013: fixed candidate, many simulated safety sets.

A pool of N prompts carries, per prompt: ``cov`` (k_max reference labels, the covariate),
``draw`` (a way to get one candidate label for a set of prompts: a column of held-out
evaluation labels, or a fresh Bernoulli draw when the truth is exact), ``p_truth`` (the
candidate's per-prompt rate: exact, or the mean of a truth half disjoint from the
evaluation labels) and ``meta`` (metadata cells). The target is the pool's own rate
``truth = mean(p_truth)``, so every bound is compared with the same finite-pool truth.

Arms (DESIGN.md section 5): R random split + pooled bounds; S1 reference-rate strata +
pooled bounds; S2 reference-rate strata + stratified bounds (the method); M metadata
strata; P permuted (placebo) strata; O strata on ``p_truth`` (oracle). Arms share random
streams per replicate (paired).

Per cell: coverage failure (truth > UB), mean width UB - estimate, pass rate at
tau = truth + Delta, and the pre-flight ingredients (ICC, rho, reliability) for H3.
"""
import zlib

import numpy as np

import stratbounds as SB

C_H = {2: 2 / np.pi, 3: 0.81, 4: 0.88, 8: 0.96}
DELTAS_TAU = (-0.02, 0.0, 0.01, 0.02, 0.04, 0.06)


def quantile_strata(x, H, rng):
    order = np.lexsort((rng.random(len(x)), x))
    out = np.empty(len(x), dtype=int)
    out[order] = np.arange(len(x)) * H // len(x)
    return out


def icc_from_samples(Y):
    """One-way ANOVA ICC from an (N, K) 0/1 array (K >= 2)."""
    Y = np.asarray(Y, dtype=float)
    N, K = Y.shape
    m = Y.mean()
    msb = K * ((Y.mean(axis=1) - m) ** 2).sum() / (N - 1)
    msw = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum() / (N * (K - 1))
    s2b = max((msb - msw) / K, 0.0)
    return s2b / (s2b + msw) if s2b + msw > 0 else 0.0


def reliability(icc, k):
    return k * icc / (1 + (k - 1) * icc) if icc > 0 else 0.0


def preflight(cov, p_truth, truth_samples=None, k=8, H=4):
    """G = ICC_cand x rho^2 x rel(k) x c_H, with rho disattenuated for the covariate's noise."""
    icc_ref = icc_from_samples(cov)
    mu = p_truth.mean()
    if truth_samples is not None:
        icc_cand = icc_from_samples(truth_samples)
    else:
        icc_cand = float(p_truth.var() / (mu * (1 - mu)))
    r_obs = np.corrcoef(cov.mean(axis=1), p_truth)[0, 1] if p_truth.std() > 0 else 0.0
    rel_full = reliability(icc_ref, cov.shape[1])
    rel_t = reliability(icc_cand, truth_samples.shape[1]) if truth_samples is not None else 1.0
    rho = float(np.clip(r_obs / np.sqrt(max(rel_full * rel_t, 1e-12)), -1, 1))
    G = icc_cand * rho ** 2 * reliability(icc_ref, k) * C_H.get(H, 0.9)
    return dict(icc_ref=float(icc_ref), icc_cand=float(icc_cand), rho=rho,
                rel_k=float(reliability(icc_ref, k)), G=float(G), ess_pred=float(1 / (1 - G)))


def _draw_split(strata, n_s, rng):
    """Proportional stratified sample of the pool; returns (indices, stratum of each)."""
    N = len(strata)
    idx, st = [], []
    for h in np.unique(strata):
        mem = np.flatnonzero(strata == h)
        k = max(2, int(round(n_s * len(mem) / N)))
        idx.append(rng.choice(mem, size=min(k, len(mem)), replace=False))
        st.append(np.full(len(idx[-1]), h))
    return np.concatenate(idx), np.concatenate(st)


def _cell_bounds(labels, st, W, delta, bounds):
    """Stratified bounds (st, W given) or pooled (st None); returns {name: (ub, est)}."""
    out = {}
    if st is None:
        s, n = labels.sum(), len(labels)
        for b in bounds:
            out[b] = (BOUND_FN[b](s, n, 1.0, delta), s / n)
    else:
        hs = np.arange(len(W))
        s = np.array([labels[st == h].sum() for h in hs])
        n = np.array([(st == h).sum() for h in hs])
        est = SB.estimate(s, n, W)
        for b in bounds:
            out[b] = (BOUND_FN[b](s, n, W, delta), est)
    return out


BOUND_FN = {"b1w": SB.b1w, "b1": SB.b1, "b2": SB.b2}


def run_cells(pool, cells, reps, seed=0):
    """
    ``pool``: dict(cov, draw, p_truth, meta). ``cells``: list of dict(arm, k, H, n_s,
    delta, bounds). Returns one summary row per cell and bound.
    """
    cov, draw, p_truth, meta = pool["cov"], pool["draw"], pool["p_truth"], pool["meta"]
    truth = float(np.mean(p_truth))
    N = len(p_truth)
    rows = []
    for c in cells:
        rng = np.random.default_rng([seed, zlib.crc32(repr((c["arm"], c["k"], c["H"])).encode())])
        # strata are fixed once per cell (the covariate is computed once, before training)
        arm, k, H = c["arm"], c["k"], c["H"]
        if arm in ("S1", "S2"):
            strata = quantile_strata(cov[:, :k].mean(axis=1), H, rng)
        elif arm == "P":
            strata = quantile_strata(rng.permutation(cov[:, :k].mean(axis=1)), H, rng)
        elif arm == "O":
            strata = quantile_strata(p_truth, H, rng)
        elif arm == "M":
            strata = np.unique(meta, return_inverse=True)[1]
        else:
            strata = None
        W = None if strata is None else np.bincount(strata) / N
        acc = {b: dict(miss=0, width=0.0, passes=np.zeros(len(DELTAS_TAU))) for b in c["bounds"]}
        for r in range(reps):
            rr = np.random.default_rng([seed, r, c["n_s"]])
            if strata is None:
                ids = rr.choice(N, size=c["n_s"], replace=False)
                st = None
            else:
                ids, st = _draw_split(strata, c["n_s"], rr)
            labels = draw(ids, rr)
            pooled = arm in ("R", "S1")
            res = _cell_bounds(labels, None if pooled else st, None if pooled else W,
                               c["delta"], c["bounds"])
            for b, (ub, est) in res.items():
                a = acc[b]
                a["miss"] += truth > ub
                a["width"] += ub - est
                a["passes"] += ub <= truth + np.array(DELTAS_TAU)
        for b, a in acc.items():
            rows.append(dict(arm=arm, k=k, H=H, n_s=c["n_s"], delta=c["delta"], bound=b,
                             reps=reps, truth=truth, miss=a["miss"] / reps,
                             width=a["width"] / reps,
                             **{f"pass_{d:+.2f}": float(p / reps)
                                for d, p in zip(DELTAS_TAU, a["passes"])}))
    return rows


def standard_cells(bounds=("b1w", "b1"), n_s_list=(100, 200, 400), deltas=(0.05, 0.1),
                   ks=(1, 2, 4, 8), Hs=(2, 4, 8)):
    """The design's factor grid: S2 over k x H x n_s; the other arms at k = 8, H = 4."""
    cells = []
    for n_s in n_s_list:
        for d in deltas:
            for arm in ("R", "S1", "M", "P", "O"):
                cells.append(dict(arm=arm, k=8, H=4, n_s=n_s, delta=d, bounds=bounds))
            for k in ks:
                for H in Hs:
                    cells.append(dict(arm="S2", k=k, H=H, n_s=n_s, delta=d, bounds=bounds))
    return cells

"""Split lab for spike 012: does a rerandomised candidate/safety split keep the safety test valid?

Classic TPR-gap setup (``seldonian.synthetic.make_synthetic``, equal-opportunity constraint
``|TPR_a - TPR_b| - tau <= 0`` via the project's ``ghat_tpr_diff`` with the t-test bound).

Candidate selection is real Seldonian selection over a small family, so thousands of seeds
run in minutes: logistic regression fit on D_c gives a score, and the candidate is a pair of
group-specific thresholds on it (41 x 41 grid, Hardt-style post-processing). Among pairs whose
*predicted* upper bound on D_c passes (width doubled, n scaled to the safety set, exactly as
``_rate_diff_bound(predict=True)``), take the most accurate on D_c. The safety test is the
project function on D_s. Truth is computed on a 1M-row population from the same distribution.

Split rules (``SPLITS``) are the arms; each returns a boolean mask of the safety rows.

    ../../../.venv/bin/python splitlab.py --seeds 2000 --n 1000 --out results_n1000.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import betaincinv
from scipy.stats import chi2, norm, t as tdist
from sklearn.linear_model import LogisticRegression

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from seldonian.objectives import ghat_tpr_diff  # noqa: E402
from seldonian.synthetic import make_synthetic  # noqa: E402

D = 10
A_IDX = 5
TAU = float(os.environ.get("SPLITLAB_TAU", "0.15"))
DELTA = 0.05
TEST_SIZE = 0.35          # SeldonianAlgorithmLogRegCMAES default
GRID = np.linspace(-3.0, 3.0, int(os.environ.get("SPLITLAB_GRID", "121")))  # logit thresholds
POP_SEED = 10 ** 6
#: one-sample bound for both the predicted and the real test; set by --bound. 'ttest'
#: collapses to zero width on unanimous indicators, which candidate selection exploits
BOUND = "clopper_pearson"
# 'wald' is a tight stress test, not a project bound: |d| + z_{1-delta} * se on the gap
# itself, which covers at close to delta on fresh data, so a leak shows as over-coverage

_POP = None


def population():
    """1M rows from the same distribution (fixed A_IDX), cached per worker."""
    global _POP
    if _POP is None:
        X, y, _ = make_synthetic(10 ** 6, D, A_idx=A_IDX, seed=POP_SEED)
        _POP = (X, y)
    return _POP


# ------------------------------------------------------------------------------ helpers

def logit(X, w, b):
    return X @ w + b


def group_rates(s, y, A, t_grid):
    """TPR by group at every threshold: (tpr_a[len grid], tpr_b[...], n_pos_a, n_pos_b)."""
    out = []
    for a in (1, 0):
        sp = np.sort(s[(y == 1) & (A == a)])
        k = len(sp) - np.searchsorted(sp, t_grid, side="right")
        out.append((k / max(len(sp), 1), len(sp)))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def group_acc(s, y, A, t_grid):
    """Number correct in each group at every threshold (the two groups add up)."""
    out = []
    for a in (1, 0):
        m = A == a
        sp1 = np.sort(s[m & (y == 1)])
        sp0 = np.sort(s[m & (y == 0)])
        tp = len(sp1) - np.searchsorted(sp1, t_grid, side="right")
        tn = np.searchsorted(sp0, t_grid, side="right")
        out.append(tp + tn)
    return out[0], out[1]


def ttest_width(p, n_samples, n_bound, delta_tail, inflate):
    """t-test half-width for a 0/1 rate as ``ttest_bounds`` computes it (ddof=1)."""
    n_samples = max(n_samples, 2)
    sd = np.sqrt(np.maximum(p * (1 - p), 0) * n_samples / (n_samples - 1))
    return sd / np.sqrt(n_bound) * tdist.ppf(1 - delta_tail, n_bound - 1) * inflate


def rate_interval(p, n_samples, n_bound, delta_tail, inflate):
    """
    (lower, upper) of a 0/1 rate as the project's bound computes it for ``n_bound``,
    widened by ``inflate`` about the mean and clipped to [0, 1] (``_pack``).
    """
    p = np.asarray(p, dtype=float)
    if BOUND == "ttest":
        w = ttest_width(p, n_samples, n_bound, delta_tail, 1.0)
        lo, hi = p - w, p + w
    elif BOUND == "clopper_pearson":
        k = p * n_bound
        with np.errstate(all="ignore"):
            lo = np.where(k <= 0, 0.0, betaincinv(np.maximum(k, 1e-12), n_bound - k + 1,
                                                   delta_tail))
            hi = np.where(k >= n_bound, 1.0, betaincinv(k + 1, np.maximum(n_bound - k, 1e-12),
                                                          1 - delta_tail))
        lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    else:
        raise ValueError(BOUND)
    lo, hi = p - inflate * (p - lo), p + inflate * (hi - p)
    if BOUND != "ttest":
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return lo, hi


def signed_gap(X, y, pred):
    """(TPR_b - TPR_a, its standard error) on (X, y) for 0/1 predictions."""
    A = X[:, A_IDX]
    out = []
    for a in (1, 0):
        m = (y == 1) & (A == a)
        p = pred[m].mean()
        out.append((p, p * (1 - p) / max(m.sum(), 1)))
    return out[1][0] - out[0][0], float(np.sqrt(out[0][1] + out[1][1]))


# ------------------------------------------------------------------------------ split rules

def n_safety(n):
    return int(round(TEST_SIZE * n))


def _perm_mask(n, rng):
    m = np.zeros(n, dtype=bool)
    m[rng.permutation(n)[:n_safety(n)]] = True
    return m


def split_random(X, y, rng, **_):
    return _perm_mask(len(y), rng), {}


def _stratified(cells, rng):
    """Blocked randomisation: the same safety fraction inside every cell."""
    m = np.zeros(len(cells), dtype=bool)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        k = int(round(TEST_SIZE * len(idx)))
        m[rng.permutation(idx)[:k]] = True
    return m


def split_strat_y(X, y, rng, **_):
    return _stratified(y.astype(int), rng), {}


def split_strat_Ay(X, y, rng, **_):
    return _stratified(2 * X[:, A_IDX].astype(int) + y.astype(int), rng), {}


def _ghat_point(X, y, theta, centre=False):
    """Predictions at a fixed theta_s, as the 2020 code made them (optionally on centred X)."""
    Xb = X - X.mean(axis=0) if centre else X
    return (0.5 < 1 / (1 + np.exp(-(Xb @ theta[:-1] + theta[-1])))).astype(int)


def _alg1(X, y, rng, theta, n_tries, eps=None, centre=False):
    """
    The user's 2020 Algorithm 1: redraw the split until |ghat(theta_s | D_c) -
    ghat(theta_s | D_s)| is small. ``eps=None`` keeps the best of ``n_tries`` (v2),
    otherwise the first split under ``eps`` (capped at ``n_tries``).
    Uses the project's ghat_tpr_diff with ub=False, as the old code did.
    """
    g = ghat_tpr_diff(A_IDX, threshold=TAU)
    pred = _ghat_point(X, y, theta, centre)
    best, best_d, tries = None, np.inf, 0
    for tries in range(1, n_tries + 1):
        m = _perm_mask(len(y), rng)
        gc = g(X[~m], y[~m], pred[~m], DELTA, int(m.sum()), predict=True, ub=False)
        gs = g(X[m], y[m], pred[m], DELTA, int(m.sum()), predict=False, ub=False)
        d = abs(gc - gs)
        if d < best_d:
            best, best_d = m, d
        if eps is not None and d <= eps:
            break
    info = dict(tries=tries, balance=float(best_d),
                degenerate=bool(pred.min() == pred.max()))
    return best, info


def split_alg1_orig(X, y, rng, seed, **_):
    """Exactly the removed code's theta_s: default_rng(seed).random(D + 1), best of 30."""
    theta = np.random.default_rng(seed).random(X.shape[1] + 1)
    return _alg1(X, y, rng, theta, 30)


def split_alg1_gauss(X, y, rng, **_):
    """Same rule with theta_s ~ N(0, I) on centred features (non-degenerate), best of 30."""
    theta = rng.standard_normal(X.shape[1] + 1)
    return _alg1(X, y, rng, theta, 30, centre=True)


def split_alg1_gauss_thr(X, y, rng, **_):
    """Threshold version (v1): first split with |diff| <= 0.01, at most 1000 tries."""
    theta = rng.standard_normal(X.shape[1] + 1)
    return _alg1(X, y, rng, theta, 1000, eps=0.01, centre=True)


def _mahalanobis(Z, rng, p_accept, max_tries=5000):
    """
    Morgan & Rubin (2012) rerandomisation: accept the first split whose Mahalanobis
    distance between the safety and candidate means of the covariates ``Z`` is under the
    chi2_k quantile ``p_accept``.
    """
    n, k = Z.shape
    Z = Z - Z.mean(axis=0)
    ns = n_safety(n)
    cov = np.cov(Z, rowvar=False) * (1 / ns + 1 / (n - ns))
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    rank = np.linalg.matrix_rank(np.atleast_2d(cov))
    thr = chi2.ppf(p_accept, rank)
    for tries in range(1, max_tries + 1):
        m = _perm_mask(n, rng)
        d = Z[m].mean(axis=0) - Z[~m].mean(axis=0)
        M = float(d @ cov_inv @ d)
        if M <= thr:
            break
    return m, dict(tries=tries, balance=M, k=int(rank))


def split_maha_design(X, y, rng, **_):
    """Mahalanobis on design covariates: A, y, A*y and all features, p_accept = 0.1."""
    A = X[:, A_IDX]
    Z = np.column_stack([A, y, A * y, np.delete(X, A_IDX, axis=1)])
    return _mahalanobis(Z, rng, 0.1)


def _full_score(X, y):
    """Logistic regression on ALL the data: a pre-split covariate that tracks the candidate."""
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(X, y)
    return logit(X, lr.coef_[0], lr.intercept_[0])


def split_adv_own_score(X, y, rng, **_):
    """
    Adversarial: Algorithm 1 with theta_s = the full-data LR (the candidate's own score
    function), best of 30. Balances the statistic the test uses, at one threshold pair.
    """
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(X, y)
    theta = np.append(lr.coef_[0], lr.intercept_[0])
    return _alg1(X, y, rng, theta, 30)


def split_adv_grid(X, y, rng, **_):
    """
    Adversarial, extreme: Mahalanobis rerandomisation on the positives' TP indicators of
    the full-data score at 9 thresholds per group (18 covariates), p_accept = 0.01. The
    candidate then selects thresholds on D_c, whose TPR curve the split has matched to D_s.
    """
    s = _full_score(X, y)
    A = X[:, A_IDX]
    ts = np.linspace(-2, 2, 9)
    cols = []
    for a in (1, 0):
        pos = ((y == 1) & (A == a)).astype(float)
        for t in ts:
            cols.append(pos * (s > t))
        cols.append(pos)
    return _mahalanobis(np.column_stack(cols), rng, 0.01, max_tries=20000)


SPLITS = {
    "random": split_random,
    "strat_y": split_strat_y,
    "strat_Ay": split_strat_Ay,
    "alg1_orig_bo30": split_alg1_orig,
    "alg1_gauss_bo30": split_alg1_gauss,
    "alg1_gauss_thr": split_alg1_gauss_thr,
    "maha_design": split_maha_design,
    "adv_own_score": split_adv_own_score,
    "adv_grid": split_adv_grid,
}


# ------------------------------------------------------------------------------ one run

def pred_ub(pa, pb, npa, npb, na, nb, inflate):
    """Predicted upper bound on |TPR_a - TPR_b| (broadcasts over pa, pb)."""
    if BOUND == "wald":
        se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        return np.abs(pa - pb) + inflate * norm.ppf(1 - DELTA) * se
    la, ha = rate_interval(pa, npa, na, DELTA / 4, inflate)
    lb, hb = rate_interval(pb, npb, nb, DELTA / 4, inflate)
    # abs() of the interval b - a, as RandomVariable.__abs__
    return np.maximum(np.abs(lb - ha), np.abs(hb - la))


def candidate(Xc, yc, n_s, inflate):
    """Seldonian candidate selection over the threshold grid. Returns (w, b, ta, tb, pred_ub)."""
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(Xc, yc)
    w, b = lr.coef_[0], lr.intercept_[0]
    s = logit(Xc, w, b)
    A = Xc[:, A_IDX]
    pa, pb, npa, npb = group_rates(s, yc, A, GRID)
    # _subgroup_n(predict=True): the subgroup's share of the candidate set, times n_s
    na = max(2, int(n_s * npa / len(yc)))
    nb = max(2, int(n_s * npb / len(yc)))
    ub = pred_ub(pa[:, None], pb[None, :], npa, npb, na, nb, inflate)
    acc_a, acc_b = group_acc(s, yc, A, GRID)
    acc = (acc_a[:, None] + acc_b[None, :]) / len(yc)
    feasible = ub <= TAU
    if feasible.any():
        i, j = np.unravel_index(np.argmax(np.where(feasible, acc, -1)), acc.shape)
    else:
        i, j = np.unravel_index(np.argmin(ub), ub.shape)
    T = np.empty((2, BINS))
    T[1], T[0] = GRID[i], GRID[j]
    return w, b, T, float(ub[i, j]), bool(feasible.any())


# ------------------------------------------------------------------------------ binned family

BINS = 20
BIN_COL = 1               # a pure-noise feature (uniform), standing in for prompt metadata
OFFSETS = np.linspace(-1.5, 1.5, 13)


def bins_of(X):
    return np.minimum((X[:, BIN_COL] * BINS).astype(int), BINS - 1)


def candidate_binned(Xc, yc, n_s, inflate, passes=3):
    """
    High-capacity adversary: start from :func:`candidate`, then give every (group, bin) cell
    its own threshold offset (40 parameters) and coordinate-ascend D_c accuracy subject to
    the predicted bound. Each cell's choice fits that cell's noise in D_c.
    """
    w, b, T, _, _ = candidate(Xc, yc, n_s, inflate)
    s = logit(Xc, w, b)
    A = Xc[:, A_IDX].astype(int)
    k = bins_of(Xc)
    npos = [int(((yc == 1) & (A == g)).sum()) for g in (0, 1)]
    nb_ = [max(2, int(n_s * npos[g] / len(yc))) for g in (0, 1)]
    cells = {}
    for g in (0, 1):
        for j in range(BINS):
            m = (A == g) & (k == j)
            cells[g, j] = (np.sort(s[m & (yc == 1)]), np.sort(s[m & (yc == 0)]))

    def cell_counts(g, j, t):
        sp1, sp0 = cells[g, j]
        tp = len(sp1) - np.searchsorted(sp1, t, side="right")
        tn = np.searchsorted(sp0, t, side="right")
        return tp, tn

    TP = np.zeros(2)
    correct = 0
    for g in (0, 1):
        for j in range(BINS):
            tp, tn = cell_counts(g, j, T[g, j])
            TP[g] += tp
            correct += tp + tn
    base = T[:, 0].copy()
    for _ in range(passes):
        for g in (0, 1):
            for j in range(BINS):
                tp0, tn0 = cell_counts(g, j, T[g, j])
                tv = base[g] + OFFSETS
                tp, tn = cell_counts(g, j, tv)
                TPg = TP[g] - tp0 + tp
                rate = [TP[0] / npos[0], TP[1] / npos[1]]
                rate[g] = TPg / npos[g]
                ub = pred_ub(rate[1], rate[0], npos[1], npos[0], nb_[1], nb_[0], inflate)
                acc = correct - tp0 - tn0 + tp + tn
                ok = ub <= TAU
                pick = int(np.argmax(np.where(ok, acc, -1))) if ok.any() else int(np.argmin(ub))
                T[g, j] = tv[pick]
                TP[g] = TPg[pick]
                correct = acc[pick]
    ub = float(pred_ub(TP[1] / npos[1], TP[0] / npos[0], npos[1], npos[0], nb_[1], nb_[0],
                       inflate))
    return w, b, T, ub, bool(ub <= TAU)


FAMILIES = {"base": candidate, "binned": candidate_binned}
FAMILY = "base"


def predict(X, w, b, T):
    s = logit(X, w, b)
    return (s > T[X[:, A_IDX].astype(int), bins_of(X)]).astype(int)


def split_strat_cells(X, y, rng, **_):
    """Blocked on (bin, A, y): the LLM analogue of stratifying on prompt metadata."""
    return _stratified(bins_of(X) * 4 + 2 * X[:, A_IDX].astype(int) + y.astype(int), rng), {}


def split_adv_bins(X, y, rng, **_):
    """
    Adversarial for the binned family: Mahalanobis rerandomisation (p_accept = 0.01) on the
    positives' count and their TP indicators at the full-data score's thresholds -1, 0, 1 in
    every (group, bin) cell (160 covariates) - the statistics the binned candidate fits.
    """
    s = _full_score(X, y)
    A = X[:, A_IDX]
    k = bins_of(X)
    cols = []
    for g in (0, 1):
        for j in range(BINS):
            pos = ((y == 1) & (A == g) & (k == j)).astype(float)
            cols.append(pos)
            for t in (-1.0, 0.0, 1.0):
                cols.append(pos * (s > t))
    return _mahalanobis(np.column_stack(cols), rng, 0.01, max_tries=20000)


SPLITS["strat_cells"] = split_strat_cells
#: reference, not a split rule: a random split whose safety test reuses D_c (full leak)
SPLITS["reuse_Dc"] = split_random
SPLITS["adv_bins"] = split_adv_bins


def run(seed, split, n, inflate=2.0):
    t0 = time.time()
    X, y, _ = make_synthetic(n, D, A_idx=A_IDX, seed=seed)
    rng = np.random.default_rng([seed, 12])
    m, info = SPLITS[split](X, y, rng, seed=seed)
    Xc, yc, Xs, ys = X[~m], y[~m], X[m], y[m]
    if split == "reuse_Dc":
        Xs, ys = Xc, yc
    w, b, T, ub_pred, pred_pass = FAMILIES[FAMILY](Xc, yc, int(m.sum()), inflate)
    pred_s = predict(Xs, w, b, T)
    d_s, se_s = signed_gap(Xs, ys, pred_s)
    d_c, se_c = signed_gap(Xc, yc, predict(Xc, w, b, T))
    Xp, yp = population()
    pred_p = predict(Xp, w, b, T)
    d_true, _ = signed_gap(Xp, yp, pred_p)
    gap_true = abs(d_true)
    # the safety test on the same candidate under two bounds: the tight Wald test and
    # the project's ghat_tpr_diff (Clopper-Pearson unless --bound says otherwise)
    ub_wald = abs(d_s) + norm.ppf(1 - DELTA) * se_s
    proj = "clopper_pearson" if BOUND == "wald" else BOUND
    ub_proj = float(ghat_tpr_diff(A_IDX, method=proj, threshold=TAU)(
        Xs, ys, pred_s, DELTA, len(ys), predict=False, ub=True)) + TAU
    ub = ub_wald if BOUND == "wald" else ub_proj
    passed = bool(ub <= TAU)
    return dict(
        seed=seed, split=split, n=n, inflate=inflate, bound=BOUND, tau=TAU, family=FAMILY,
        **{f"split_{k}": v for k, v in info.items()},
        pred_pass=pred_pass, pred_ub=ub_pred, passed=passed, ub_s=float(ub),
        d_c=float(d_c), d_s=float(d_s), d_true=float(d_true), se_c=se_c, se_s=se_s,
        violates=bool(gap_true > TAU), miss=bool(passed and gap_true > TAU),
        uncovered=bool(gap_true > ub),
        passed_proj=bool(ub_proj <= TAU), miss_proj=bool(ub_proj <= TAU and gap_true > TAU),
        acc_true=float((pred_p == yp).mean()), ta=float(T[1].mean()), tb=float(T[0].mean()),
        seconds=time.time() - t0)


def _job(args):
    global BOUND, FAMILY
    BOUND, FAMILY = args[-2:]
    return run(*args[:-2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--inflate", type=float, default=2.0)
    ap.add_argument("--bound", default="clopper_pearson", choices=["clopper_pearson", "ttest", "wald"])
    ap.add_argument("--family", default="base", choices=list(FAMILIES))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(s, sp, a.n, a.inflate, a.bound, a.family) for sp in a.splits.split(",") for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=20))
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()

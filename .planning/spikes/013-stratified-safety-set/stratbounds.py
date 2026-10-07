"""Upper confidence bounds on a stratified mean of 0/1 labels (spike 013).

Data are per-stratum counts: ``s[h]`` ones out of ``n[h]`` labels, stratum weights ``W[h]``
(summing to 1). The estimate is ``mu_hat = sum_h W_h s_h / n_h``. Every bound here is an
*upper* bound on the mean (the safety test certifies ``mean <= tau``). With one stratum
each reduces to its pooled counterpart, so pooled and stratified arms compare like for like.

- ``b1``: stratified Wald with a t quantile (Satterthwaite df). Per-stratum variances are
  Jeffreys-smoothed, ``p~ = (s + 0.5) / (n + 1)``, so an all-zero stratum does not give zero
  width (the degeneracy spike 012 found in the project's t-test). Approximate, as the
  project's ``ttest``.
- ``b2``: union-intersection betting test (after Spertus & Stark 2022, "Sweeter than
  SUITE", and Spertus, Sridhar & Stark 2024). Within stratum h, for a hypothesised mean
  ``a``, the capital ``prod_i (1 + lam_i (a - X_i))`` has expectation <= 1 whenever the true
  mean is >= ``a``, provided each bet ``lam_i`` is fixed before ``X_i`` is seen and
  ``0 <= lam_i <= 1 / (1 - a)``. Bets are *predictable*: an aGRAPA-style Kelly
  approximation from the earlier labels in a random order, ``(a - mu_prev) /
  (var_prev + (a - mu_prev)^2)`` clipped to ``[0, 0.9 / (1 - a)]``, so a stratum whose
  running mean is already above ``a`` stops betting instead of losing capital. All strata
  are interleaved in one random order and ``mu_prev`` is the stratum's running mean shrunk
  toward the pooled running mean (``KAPPA`` pseudo-samples), so a small stratum does not
  learn from scratch; bets stay predictable. The product over all labels is an e-value for
  every vector ``a`` and its log separates by stratum; 8 random orders are averaged (min
  of an average >= average of the mins, applied per order). The
  composite null ``sum_h W_h mu_h >= m`` is rejected when it is >= 1/delta at the
  least-evidence vector with ``sum_h W_h a_h >= m``. That minimum is lower-bounded (so the
  test stays valid) by Lagrangian weak duality over the linear constraint, and by a grid on
  ``a`` whose bets are fixed per grid cell (the capital then increases with ``a`` inside a
  cell, so the left end bounds it). Distribution-free for 0/1 labels.

Sampling without replacement from a finite pool (the plasmode's target) keeps both valid:
the capital is the exponential of a stratum sum, a convex function of it, so Hoeffding's
(1963) reduction from without- to with-replacement applies.

Two-phase (target = the population the pool was drawn from, ``N`` = pool size): the pool's
own stratum shares are noisy. ``b1`` adds ``sum_h W_h (p_h - mu_hat)^2 / N`` to the
variance; ``b2`` spends 0.1 delta on per-stratum Clopper-Pearson intervals, bounds the
between-stratum variance by its maximum over their box (a convex quadratic, so a vertex),
and adds a Bernstein term for the pool mean at another 0.1 delta.

    ../../../.venv/bin/python check_bounds.py      # coverage and pooled-equivalence checks
"""
import itertools

import numpy as np
from scipy.special import betaincinv, logsumexp
from scipy.stats import t as tdist

A_GRID = np.linspace(0.0, 1.0, 401)
N_ORDERS = 8
KAPPA = 10.0
NU_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 1e5, 240)])
M_TOL = 2.5e-4


def _arr(*xs):
    return [np.atleast_1d(np.asarray(x, dtype=float)) for x in xs]


def estimate(s, n, W):
    s, n, W = _arr(s, n, W)
    return float(np.sum(W * s / n))


# ------------------------------------------------------------------------------ B1

def b1(s, n, W, delta, N=None):
    """Stratified Wald upper bound, t quantile at Satterthwaite df; ``N`` adds two-phase."""
    s, n, W = _arr(s, n, W)
    p = s / n
    pt = (s + 0.5) / (n + 1.0)
    v_h = W ** 2 * pt * (1 - pt) / np.maximum(n - 1, 1)     # W^2 s_h^2 / n_h, smoothed
    var = v_h.sum()
    df = var ** 2 / np.sum(v_h ** 2 / np.maximum(n - 1, 1)) if var > 0 else 1.0
    mu = float(np.sum(W * p))
    if N is not None:
        var += float(np.sum(W * (p - mu) ** 2)) / N
    return mu + float(tdist.ppf(1 - delta, max(df, 1.0))) * np.sqrt(var)


def b1w(s, n, W, delta, N=None):
    """
    Stratified Wilson-type score bound: the smallest ``m >= mu_hat`` with
    ``m - mu_hat >= z * sqrt(V(m))``, where ``V(m)`` puts every stratum at its estimate
    shifted by ``m - mu_hat`` (clipped to [0, 1]). Evaluating the variance at the
    hypothesis, not the estimate, avoids the Wald under-coverage at low rates. At one
    stratum it is the Wilson upper bound. ``N`` adds the two-phase term.
    """
    from scipy.stats import norm
    s, n, W = _arr(s, n, W)
    p = s / n
    mu = float(np.sum(W * p))
    z = float(norm.ppf(1 - delta))
    extra = float(np.sum(W * (p - mu) ** 2)) / N if N is not None else 0.0
    m = mu + np.linspace(0.0, 1.0 - mu, 4001)
    ph = np.clip(p[None, :] + (m - mu)[:, None], 0.0, 1.0)
    V = np.sum(W[None, :] ** 2 * ph * (1 - ph) / n[None, :], axis=1) + extra
    ok = (m - mu) >= z * np.sqrt(V)
    # m = mu is not a root: it passes as 0 >= 0 when V(mu) = 0 (every stratum all-zero or
    # all-one), which made the bound collapse onto the estimate at zero positives
    ok[0] = mu >= 1.0
    return float(m[np.argmax(ok)]) if ok.any() else 1.0


# ------------------------------------------------------------------------------ B2

def _order_logcaps(s, n, seed):
    """
    (H, A) per-stratum log capital at the left end of every ``a`` cell for one random
    interleaving of all labels (``seed`` picks the order).
    """
    s = np.asarray(s, dtype=int)
    n = np.asarray(n, dtype=int)
    H = len(s)
    rng = np.random.default_rng([*s.tolist(), *n.tolist(), seed])
    lab = np.concatenate([np.r_[np.ones(sh), np.zeros(nh - sh)] for sh, nh in zip(s, n)])
    grp = np.concatenate([np.full(nh, h) for h, nh in enumerate(n)])
    perm = rng.permutation(len(lab))
    lab, grp = lab[perm], grp[perm]
    a = A_GRID
    cap_hi = 0.9 / (1.0 - np.minimum(np.append(A_GRID[1:], 1.0), 1 - 1e-9))
    logK = np.zeros((H, len(a)))
    cnt = np.zeros(H)
    tot = np.zeros(H)
    pool_tot, pool_cnt = 0.0, 0
    for x, h in zip(lab, grp):
        pool_mu = (0.5 + pool_tot) / (1.0 + pool_cnt)
        mu_p = (tot[h] + KAPPA * pool_mu) / (cnt[h] + KAPPA)
        var_p = max(mu_p * (1 - mu_p), 0.25 / (cnt[h] + KAPPA))
        gap = a - mu_p
        lam = np.clip(gap / (var_p + gap ** 2), 0.0, cap_hi)
        logK[h] += np.log1p(lam * (a - x))
        cnt[h] += 1
        tot[h] += x
        pool_tot += x
        pool_cnt += 1
    return logK


def _dual_curve(logK, W):
    """
    C[nu] = sum_h min_a (g_h(a) - nu W_h a), a lower bound per grid cell: on
    [A_i, A_i+1] the capital (bets fixed on the cell) is >= g(A_i), and -nu W a >= -nu W A_i+1.
    """
    a_next = np.append(A_GRID[1:], 1.0)
    C = np.zeros(len(NU_GRID))
    for g, wh in zip(logK, W):
        C += np.min(g[None, :] - NU_GRID[:, None] * wh * a_next[None, :], axis=1)
    return C


def b2(s, n, W, delta, N=None):
    """Union-intersection betting upper bound; ``N`` adds the two-phase term."""
    s, n, W = _arr(s, n, W)
    d_test = delta if N is None else 0.8 * delta
    Cs = np.array([_dual_curve(_order_logcaps(s, n, k), W) for k in range(N_ORDERS)])
    thr = np.log(1.0 / d_test)

    def rejected(m):
        # per order: max_nu C + nu m  <=  min_{sum W a >= m} sum_h g_h(a_h); then average
        L = np.max(Cs + NU_GRID[None, :] * m, axis=1)
        return logsumexp(L) - np.log(N_ORDERS) >= thr

    # the evidence grows with m, so the rejected set is an interval [ub, 1]
    lo, hi = estimate(s, n, W), 1.0
    if not rejected(hi):
        ub = 1.0
    else:
        while hi - lo > M_TOL:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if rejected(mid) else (mid, hi)
        ub = hi
    if N is not None:
        ub = min(1.0, ub + _two_phase_term(s, n, W, N, 0.1 * delta, 0.1 * delta))
    return ub


def _cp(s, n, delta):
    lo = 0.0 if s <= 0 else float(betaincinv(s, n - s + 1, delta))
    hi = 1.0 if s >= n else float(betaincinv(s + 1, n - s, 1 - delta))
    return lo, hi


def _two_phase_term(s, n, W, N, d_ci, d_dev):
    """Bernstein bound on (pool mean of mu_h(x)) - mu, with a worst-case between variance."""
    H = len(s)
    boxes = [_cp(sh, nh, d_ci / (2 * H)) for sh, nh in zip(s, n)]
    if H == 1:
        return 0.0
    best = 0.0
    for v in itertools.product(*boxes):
        v = np.asarray(v)
        best = max(best, float(np.sum(W * (v - np.sum(W * v)) ** 2)))
    L = np.log(1.0 / d_dev)
    c = 1.0 / N
    return c * L / 3 + np.sqrt((c * L / 3) ** 2 + 2 * best / N * L)


BOUNDS = {"b1": b1, "b1w": b1w, "b2": b2}

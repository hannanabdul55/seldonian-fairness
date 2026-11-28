"""
Smooth (quasi-differentiable) versions of the predicted-test upper bounds, for candidate
selection only.

Phase 2 of the safe-RL bounds workstream (properties doc P11-P12, formulation doc
https://claude.ai/artifact/DsjKyutY1MXDViYFHcaSND). The safety test keeps the exact bounds in
:mod:`seldonian.bounds`; this module gives candidate selection the same predicted bound as a
differentiable function of a few sufficient statistics, so torch autograd yields
``dU / d(statistics)``. The policy gradient of each statistic comes from a score-function
estimator elsewhere; the coefficients from here are used detached.

Rules every function follows:

* At the statistics where the exact bound is defined, the smooth value equals the exact
  predicted bound or lies above it (never below): a smooth replacement of a kink is an upper
  bound of what it replaces, so a surrogate pass is never looser than the exact prediction.
* No clipping to the support: an unclipped upper limit is still an upper limit and stays
  smooth.

Contents:

* :func:`cp_upper` / :func:`cp_lower` (Clopper-Pearson, identical to
  :func:`seldonian.bounds.clopper_pearson_bounds` at real ``k``);
* :func:`eb_upper` (Maurer and Pontil 2009, identical to
  :func:`seldonian.bounds.empirical_bernstein_bounds` away from zero variance; not monotone
  in each sample, so not for steering, formulation doc gradient rule 5);
* :func:`bentkus_upper` / :func:`bentkus_lower` (identical to
  :func:`seldonian.bounds.bentkus_bounds` with ``variance=None``; one-sided derivative at
  its kinks);
* :func:`betting_upper` / :func:`betting_lower` (identical to
  :func:`seldonian.bounds.betting_mixture_bounds`), differentiable in per-sample weights:
  the gradient is the bound's influence function, which is what a score-function policy
  gradient needs (:func:`betting_influence`);
* :func:`smooth_max` (log-sum-exp upper bound of a max) and :class:`Interval`, a pair of
  smooth endpoints propagated through sums, constants, products, ``abs``, ``max`` and
  ``min``;
* :func:`spec_interval`, which evaluates a canonical spec of spike 016
  (``.planning/spikes/016-prompt-to-spec-compile/speclab.py``) over per-measure
  intervals, and :func:`spec_delta_side`, its per-endpoint budget (the same union split
  as :class:`seldonian.llm.constraints.ExpressionConstraint`).
"""
import math

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import betainc, betaincinv, betaln, logsumexp

from seldonian.bounds import _bentkus_lower01, _mixture_grid


# ------------------------------------------------------------------ Clopper-Pearson

def _cp_upper_np(mu, n, delta):
    k = np.clip(mu, 0.0, 1.0 - 1e-12) * n
    return betaincinv(k + 1.0, n - k, 1.0 - delta)


def _cp_dupper_dmu_np(mu, n, delta, h=1e-6):
    """
    ``dU / dmu`` by implicit differentiation of ``I_U(k + 1, n - k) = 1 - delta``:
    ``dU/dk = -(dI/da - dI/db) / (dI/dx)``, with ``dI/dx`` in closed form and the parameter
    derivatives of the regularised incomplete beta by central differences (they are smooth).
    """
    k = np.clip(mu, 0.0, 1.0 - 1e-12) * n
    a, b = k + 1.0, n - k
    u = betaincinv(a, b, 1.0 - delta)
    dI_dx = np.exp((a - 1) * np.log(u) + (b - 1) * np.log1p(-u) - betaln(a, b))
    dI_da = (betainc(a + h, b, u) - betainc(a - h, b, u)) / (2 * h)
    dI_db = (betainc(a, b + h, u) - betainc(a, b - h, u)) / (2 * h)
    return -(dI_da - dI_db) / dI_dx * n


class _CPUpper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, n, delta):
        m = mu.detach().cpu().double().numpy()
        ctx.save_for_backward(mu)
        ctx.n, ctx.delta = n, delta
        return torch.as_tensor(_cp_upper_np(m, n, delta), dtype=mu.dtype, device=mu.device)

    @staticmethod
    def backward(ctx, grad):
        (mu,) = ctx.saved_tensors
        d = _cp_dupper_dmu_np(mu.detach().cpu().double().numpy(), ctx.n, ctx.delta)
        return grad * torch.as_tensor(d, dtype=mu.dtype, device=mu.device), None, None


def cp_upper(mu, n, delta):
    """
    Clopper-Pearson upper limit for a 0/1 rate with sample mean ``mu`` evaluated at size
    ``n`` (``k = mu n`` real-valued, exactly as ``clopper_pearson_bounds`` does when given
    ``n``). Differentiable in ``mu`` on ``[0, 1)``; ``n`` and ``delta`` are constants.
    """
    mu = torch.as_tensor(mu, dtype=torch.float64)
    return _CPUpper.apply(mu, float(n), float(delta))


def cp_lower(mu, n, delta):
    """Clopper-Pearson lower limit, ``1 - cp_upper(1 - mu)`` (the beta-quantile symmetry)."""
    mu = torch.as_tensor(mu, dtype=torch.float64)
    return 1.0 - cp_upper(1.0 - mu, n, delta)


# ------------------------------------------------------------------ Bentkus

def _bentkus_upper_np(mu, n, delta):
    # variance=None uses the sample mean alone, so a one-point sample carries it
    return 1.0 - _bentkus_lower01(np.array([1.0 - float(mu)]), int(n), delta, None, 0.25)


class _BentkusUpper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, n, delta):
        m = float(mu.detach())
        ctx.save_for_backward(mu)
        ctx.n, ctx.delta = n, delta
        return torch.as_tensor(_bentkus_upper_np(m, n, delta), dtype=mu.dtype)

    @staticmethod
    def backward(ctx, grad):
        (mu,) = ctx.saved_tensors
        m, h = float(mu.detach()), 1e-6
        lo, hi = max(m - h, 0.0), min(m + h, 1.0)
        d = (_bentkus_upper_np(hi, ctx.n, ctx.delta)
             - _bentkus_upper_np(lo, ctx.n, ctx.delta)) / (hi - lo)
        return grad * d, None, None


def bentkus_upper(mu, n, delta):
    """
    Bentkus (2004) Theorem 1.2 upper limit for a mean on [0, 1] (the library's
    ``variance=None`` form, minimum with the KL bound), at size ``n``. It depends on the
    mean alone and is non-decreasing in it, so it is safe to steer on. The tail is the
    log-linear interpolation of the binomial survival function, which has kinks where
    ``mu n`` crosses an integer; the derivative is a central difference, which averages the
    two one-sided slopes there.
    """
    mu = torch.as_tensor(mu, dtype=torch.float64)
    return _BentkusUpper.apply(mu, float(n), float(delta))


def bentkus_lower(mu, n, delta):
    mu = torch.as_tensor(mu, dtype=torch.float64)
    return 1.0 - bentkus_upper(1.0 - mu, n, delta)


# ------------------------------------------------------------------ betting mixture

def _bet_lower_state(y, p, n, delta, size, ratio):
    """
    Root ``m`` of the library's mixture capital for a lower bound on the mean of ``y`` in
    [0, 1] under sample weights ``p`` (summing to 1), plus what its implicit derivative
    needs. Returns ``(m, g, F_m)`` with ``g_i = n sum_k pi_k log f_k(m, y_i)`` (``pi`` the
    posterior bet weights at the root) and ``F_m = dF/dm`` at fixed ``p`` (grid included).
    ``m = 0`` (nothing rejected) returns zero derivatives.
    """
    log_target = np.log(1.0 / delta)
    mean = float(np.dot(p, y))

    def parts(m):
        u, log_w = _mixture_grid(n, m, delta, size, ratio)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = 1.0 + u[:, None] * (y[None, :] - m) / m
            logf = np.where(f > 0, np.log(np.maximum(f, 1e-300)), -np.inf)
        e = logf @ p
        return logf, e, log_w

    def F(m):
        m = min(max(m, 1e-12), 1.0 - 1e-12)
        _, e, log_w = parts(m)
        return float(logsumexp(log_w + n * e)) - log_target

    lo = 1e-9
    if mean <= lo or F(lo) < 0:
        return 0.0, np.zeros_like(y), 1.0
    if F(mean) >= 0:
        return mean, np.zeros_like(y), 1.0
    m = float(brentq(F, lo, mean, xtol=1e-12, maxiter=200))
    logf, e, log_w = parts(m)
    a = log_w + n * e
    pi = np.exp(a - logsumexp(a))
    g = n * (pi @ np.where(np.isfinite(logf), logf, -1e6))
    h = 1e-7 * max(m, 1e-6)
    F_m = (F(m + h) - F(m - h)) / (2 * h)
    return m, g, F_m


def betting_influence(x, n, delta, weights=None, a=0.0, b=1.0, size=12, ratio=1.6,
                      side="upper"):
    """
    ``(bound, psi)`` for the betting-mixture bound on the mean of ``x`` in ``[a, b]``,
    evaluated at size ``n``. ``psi_i`` is the derivative of the bound with respect to a
    weight shift toward sample ``i`` (the influence function, centred so that
    ``sum_i p_i psi_i = 0``): moving probability ``eps`` from the weighted empirical
    distribution onto ``x_i`` changes the bound by about ``eps * psi_i``. Used as a
    per-response reward it gives the score-function gradient of the bound (formulation doc,
    the smooth evaluator).

    Implicit differentiation of ``F(m, e) = log sum_k w_k exp(n e_k(m)) - log(1/delta) = 0``
    in the expectations ``e_k = E_p[log f_k(m, X)]``.
    """
    x = np.asarray(x, dtype=float)
    p = np.full(x.size, 1.0 / x.size) if weights is None else np.asarray(weights, float)
    p = p / p.sum()
    z = (x - a) / (b - a)
    # upper bound on the mean of z = 1 - lower bound on the mean of 1 - z
    y = 1.0 - z if side == "upper" else z
    m, g, F_m = _bet_lower_state(y, p, int(n), delta, size, ratio)
    dF = g - float(np.dot(p, g))                 # dF / d(eps toward y_i)
    dm = -dF / F_m                               # lower bound of y moves by dm
    if side == "upper":
        return a + (b - a) * (1.0 - m), (b - a) * (-dm)
    return a + (b - a) * m, (b - a) * dm


class _BettingBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, n, delta, a, b, size, ratio, side):
        wn = w.detach().double().numpy()
        val, psi = betting_influence(x, n, delta, wn, a, b, size, ratio, side)
        ctx.save_for_backward(w)
        ctx.psi = psi
        return torch.as_tensor(val, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad):
        (w,) = ctx.saved_tensors
        # d bound / d w_i with p = w / sum(w): psi is already centred under p
        dw = torch.as_tensor(ctx.psi, dtype=torch.float64) / w.detach().sum()
        return (grad * dw,) + (None,) * 8


def betting_upper(x, n, delta, weights=None, a=0.0, b=1.0, size=12, ratio=1.6):
    """
    The library's betting-mixture upper limit for the mean of ``x`` in ``[a, b]`` at size
    ``n``, as a torch scalar differentiable in the per-sample ``weights`` (default uniform,
    no gradient). The sample values are constants: a response's harm does not depend on
    the parameters, its probability does, so the gradient that matters is the one in the
    weights. It is non-decreasing in each sample value, so it is safe to steer on.
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = torch.ones(x.size, dtype=torch.float64)
    return _BettingBound.apply(torch.as_tensor(weights, dtype=torch.float64), x, int(n),
                               float(delta), float(a), float(b), size, ratio, "upper")


def betting_lower(x, n, delta, weights=None, a=0.0, b=1.0, size=12, ratio=1.6):
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = torch.ones(x.size, dtype=torch.float64)
    return _BettingBound.apply(torch.as_tensor(weights, dtype=torch.float64), x, int(n),
                               float(delta), float(a), float(b), size, ratio, "lower")


# ------------------------------------------------------------------ empirical Bernstein

def eb_upper(mu, var, n, delta, eps=1e-12):
    """
    Maurer and Pontil (2009) Theorem 4 upper limit for a mean on [0, 1], one-sided at
    ``delta``: ``mu + sqrt(2 var ln(2/delta) / n) + 7 ln(2/delta) / (3 (n - 1))``. The square
    root uses ``var + eps``, which is above the exact value and smooth at ``var = 0``.
    Differentiable in both ``mu`` and ``var`` by plain autograd.
    """
    if n < 2:
        raise ValueError("eb_upper needs n >= 2")
    ln = math.log(2.0 / delta)
    return mu + torch.sqrt(2.0 * (var + eps) * ln / n) + 7.0 * ln / (3.0 * (n - 1))


# ------------------------------------------------------------------ composition

def smooth_max(xs, tau=1e-3):
    """
    ``tau * logsumexp(x / tau)``: at least ``max(xs)`` and at most ``tau ln(len(xs))`` above it.
    """
    x = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in xs])
    return tau * torch.logsumexp(x / tau, dim=0)


class Interval:
    """
    Smooth lower and upper endpoints of a quantity, propagated so that ``upper`` stays an
    upper endpoint and ``lower`` a lower one (interval arithmetic with smooth max / min).
    """

    def __init__(self, lower, upper, tau=1e-3):
        self.lower = torch.as_tensor(lower, dtype=torch.float64)
        self.upper = torch.as_tensor(upper, dtype=torch.float64)
        self.tau = tau

    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other, other, self.tau)
        return Interval(self.lower + other.lower, self.upper + other.upper, self.tau)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.upper, -self.lower, self.tau)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def scale(self, c):
        c = float(c)
        if c >= 0:
            return Interval(c * self.lower, c * self.upper, self.tau)
        return Interval(c * self.upper, c * self.lower, self.tau)

    def __mul__(self, other):
        if not isinstance(other, Interval):
            return self.scale(float(other))
        # the four endpoint products: smooth max for the upper, minus smooth max of the
        # negated products for the lower (both stay valid endpoints)
        prods = [self.lower * other.lower, self.lower * other.upper,
                 self.upper * other.lower, self.upper * other.upper]
        return Interval(-smooth_max([-q for q in prods], self.tau),
                        smooth_max(prods, self.tau), self.tau)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Interval):
            return self.scale(1.0 / float(other))
        if float(other.lower) <= 0.0 <= float(other.upper):
            raise ValueError("a ratio needs a denominator interval that excludes 0")
        return self * Interval(1.0 / other.upper, 1.0 / other.lower, self.tau)

    def __abs__(self):
        # upper: max(|l|, |u|) = max(u, -l); smooth max keeps it an upper endpoint. lower: 0 is
        # always a valid lower endpoint of |x| and keeps the expression smooth.
        return Interval(torch.zeros_like(self.lower), smooth_max([self.upper, -self.lower],
                                                                 self.tau), self.tau)

    @staticmethod
    def max(*items):
        tau = items[0].tau
        # smooth max of the uppers is an upper endpoint of the max; the plain lower of any one
        # item is a lower endpoint of the max (the largest lower is tighter but not smooth;
        # a smooth min would undershoot, so a smooth max of the lowers is not used)
        return Interval(items[0].lower, smooth_max([i.upper for i in items], tau), tau)

    @staticmethod
    def min(*items):
        tau = items[0].tau
        # min(x) = -max(-x): smooth max of the negated lowers gives a valid lower endpoint,
        # and the plain upper of any one item is a valid upper endpoint
        return Interval(-smooth_max([-i.lower for i in items], tau), items[0].upper, tau)


# ------------------------------------------------------------------ specs (spike 016)

def _spec_kind(node):
    """'const+', 'const', 'inc' or 'other', mirroring speclab._kind."""
    op = node[0]
    if op == "num":
        return "const+" if node[1] >= 0 else "const"
    if op == "m":
        return "inc"
    ks = [_spec_kind(c) for c in node[1:]]
    consts = [k.startswith("const") for k in ks]
    if all(consts):
        return "const"
    if op == "+" and all(k == "inc" or c for k, c in zip(ks, consts)):
        return "inc"
    if op == "-" and ks[0] == "inc" and consts[1]:
        return "inc"
    if op == "*" and sorted(ks) == ["const+", "inc"]:
        return "inc"
    if op in ("max", "min") and all(k == "inc" for k in ks):
        return "inc"
    return "other"


def spec_is_monotone(spec):
    return _spec_kind(spec["expr"]) == "inc"


def spec_delta_side(spec, delta):
    """Per-endpoint level: ``delta / k`` if monotone (uppers only), else ``delta / (2k)``."""
    k = len(spec["measures"])
    return delta / k if spec_is_monotone(spec) else delta / (2 * k)


def repeated_measures(spec):
    """Measure names used more than once in the expression (the dependency-problem lint)."""
    seen = {}

    def walk(node):
        if node[0] == "m":
            seen[node[1]] = seen.get(node[1], 0) + 1
        for c in node[1:]:
            if isinstance(c, list):
                walk(c)
    walk(spec["expr"])
    return sorted(k for k, v in seen.items() if v > 1)


def spec_interval(spec, intervals, tau=1e-3):
    """
    Evaluate a canonical spec's expression over ``intervals`` (``name -> Interval`` of each
    measure's smooth endpoints) and return the expression's :class:`Interval`. Its
    ``upper`` minus the threshold is the smooth g_hat. The spec's threshold is not applied
    here because the relative forms need the reference policy's value.
    """
    def ev(node):
        op = node[0]
        if op == "num":
            v = float(node[1])
            return Interval(v, v, tau)
        if op == "m":
            return intervals[node[1]]
        subs = [ev(c) for c in node[1:]]
        if op == "neg":
            return -subs[0]
        if op == "abs":
            return abs(subs[0])
        if op == "max":
            return Interval.max(*subs)
        if op == "min":
            return Interval.min(*subs)
        a, b = subs
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            if node[1][0] == "num":
                return b.scale(node[1][1])
            if node[2][0] == "num":
                return a.scale(node[2][1])
            return a * b
        if op == "/":
            if node[2][0] == "num":
                return a.scale(1.0 / node[2][1])
            return a / b
        raise ValueError(f"unknown operator {op!r}")
    return ev(spec["expr"])

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

Contents: :func:`cp_upper` (Clopper-Pearson, identical to
:func:`seldonian.bounds.clopper_pearson_bounds` at real ``k``), :func:`eb_upper` (Maurer and
Pontil 2009, identical to :func:`seldonian.bounds.empirical_bernstein_bounds` away from zero
variance), :func:`smooth_max` (log-sum-exp upper bound of a max), and :class:`Interval`, a
pair of smooth endpoints propagated through sums, constants, ``abs`` and ``max``.
"""
import math

import numpy as np
import torch
from scipy.special import betainc, betaincinv, betaln


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

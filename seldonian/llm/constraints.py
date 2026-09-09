"""
Open-ended constraints for :class:`seldonian.llm.policy.SeldonianLLMPolicy`.

The built-in :class:`seldonian.llm.policy.Constraint` bounds one rate. The classes
here bound functions of several group-conditional means (a parity gap, a ratio, a
bounded score), paired counterfactual differences and two-sample differences. Each
implements the policy's ``measure`` hook::

    g, rate, upper, n = c.measure(records, responses, delta, prompts_s,
                                  predict=..., inflation=..., ub=...)

``prompts_s`` is used only to *count* safety prompts per group (the predicted test
bounds at :func:`seldonian.llm.policy.effective_n` of the prediction and safety
sizes); no safety response is ever passed. Every constraint also has ``name``, a
settable ``threshold``, ``bound`` (a name in :data:`seldonian.llm.policy.BOUNDS`) and
``select(records)``.

A *feature* maps ``(prompts, responses, references)`` to per-episode values in a
known range ``[a, b]``; judges are 0/1 features. The range is what makes the
distribution-free bounds (``bentkus``, ``hoeffding``, ``betting_mixture``, ...)
valid for non-binary features. Every constraint spends its ``delta`` once: an
expression over ``k`` measures splits it across the measures' endpoints (see
:class:`ExpressionConstraint`), a two-sided difference across its two sides.
"""
import numpy as np

from seldonian.bounds import RandomVariable, bentkus_diff_bounds, convex_order_diff_bounds
from seldonian.llm.judges import Judge
from seldonian.llm.policy import BOUNDS, effective_n

#: two-sample bounds a :class:`TwoSampleDifferenceConstraint` may name
TWO_SAMPLE_BOUNDS = {"bentkus_diff": bentkus_diff_bounds,
                     "convex_order_diff": convex_order_diff_bounds}


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

class Feature:
    """
    Per-episode values in ``[a, b]``: ``feature(prompts, responses, references)``
    returns one float per episode. The range is what the bounds are computed on.
    """

    a = 0.0
    b = 1.0
    name = "feature"

    def __call__(self, prompts, responses, references=None):
        raise NotImplementedError


class JudgeFeature(Feature):
    """A :class:`seldonian.llm.judges.Judge` as a 0/1 feature."""

    def __init__(self, judge):
        self.judge = judge
        self.name = judge.name

    def __call__(self, prompts, responses, references=None):
        return np.asarray(self.judge(prompts, responses, references), dtype=float)


class LengthFeature(Feature):
    """``min(length / cap, 1)`` with the length in words or characters, in ``[0, 1]``."""

    def __init__(self, cap, unit="words"):
        if unit not in ("words", "chars"):
            raise ValueError("unit must be 'words' or 'chars'")
        self.cap = cap
        self.unit = unit
        self.name = f"length_{unit}/{cap}"

    def __call__(self, prompts, responses, references=None):
        if self.unit == "words":
            lengths = [len(r.split()) for r in responses]
        else:
            lengths = [len(r) for r in responses]
        return np.minimum(np.asarray(lengths, dtype=float) / self.cap, 1.0)


class CallableFeature(Feature):
    """Any ``fn(prompts, responses, references)`` with values in the declared ``[a, b]``."""

    def __init__(self, fn, a, b, name):
        self.fn = fn
        self.a = float(a)
        self.b = float(b)
        self.name = name
        if not self.b > self.a:
            raise ValueError(f"range must satisfy a < b, got a={a}, b={b}")

    def __call__(self, prompts, responses, references=None):
        out = np.asarray(self.fn(prompts, responses, references), dtype=float)
        if out.size and (out.min() < self.a or out.max() > self.b):
            raise ValueError(f"feature {self.name!r} returned values outside "
                             f"[{self.a}, {self.b}]")
        return out


def as_feature(x):
    """
    A :class:`Feature` from a feature or a :class:`seldonian.llm.judges.Judge`, which
    becomes a 0/1 :class:`JudgeFeature` (its ``judge`` attribute is the judge).
    """
    if isinstance(x, Feature):
        return x
    if isinstance(x, Judge):
        return JudgeFeature(x)
    raise TypeError(f"expected a Feature or a Judge, got {type(x).__name__}; wrap other "
                    "callables in CallableFeature")


def _select(records, group):
    if group is None:
        return list(range(len(records)))
    return [i for i, r in enumerate(records) if r.get("group") == group]


def _values(feature, records, responses, idx):
    return feature([records[i]["prompt"] for i in idx], [responses[i] for i in idx],
                   [records[i].get("reference") for i in idx])


def one_sample_interval(values, delta, n, bound, a=0.0, b=1.0):
    """
    ``(lower, upper)`` for the mean of ``values`` in ``[a, b]``, each endpoint
    one-sided at level ``delta``, evaluated as if ``n`` samples had been observed.
    Values are mapped to ``[0, 1]`` first so the Student-t and Hoeffding bounds
    (which take no range) see the right scale; the result is mapped back.
    """
    x = (np.asarray(values, dtype=float) - a) / (b - a)
    rv = BOUNDS[bound](x, delta, n=int(n))
    mean = float(x.mean())
    lower = min(float(rv.lower), mean)
    upper = max(float(rv.upper), mean)
    return a + (b - a) * lower, a + (b - a) * upper


def _inflate(point, lower, upper, inflation):
    return point - inflation * (point - lower), point + inflation * (upper - point)


def _size(m, n_s, predict):
    return effective_n(m, n_s) if predict else n_s


# --------------------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------------------

class Measure:
    """Group-conditional mean of a feature: ``E[feature | group]``."""

    def __init__(self, name, feature, group=None):
        self.name = name
        self.feature = as_feature(feature)
        self.group = group

    def select(self, records):
        return _select(records, self.group)

    def values(self, records, responses, idx):
        return np.asarray(_values(self.feature, records, responses, idx), dtype=float)


class ExpressionConstraint:
    """
    ``g = upper(expr(measures)) - threshold`` for an arbitrary expression over
    group-conditional means.

    :param measures: dict ``name -> Measure``
    :param expr: callable ``dict[name -> RandomVariable] -> RandomVariable``; use
        ordinary arithmetic, ``abs``, and :func:`seldonian.bounds.min_bounds` /
        :func:`seldonian.bounds.max_bounds`, all of which propagate intervals
    :param threshold: tau (settable; relative thresholds add a margin to the
        reference value of the expression)
    :param bound: name in :data:`seldonian.llm.policy.BOUNDS`
    :param monotone: ``True`` when ``expr`` is non-decreasing in every measure (a
        plain rate, a sum of rates, ...); then only each measure's *upper* bound is
        needed and the budget is ``delta / k`` per measure. Otherwise (the default,
        and required for differences, ``abs``, products, ...) each measure gets a
        two-sided interval at ``delta / (2k)`` per side, so all ``2k`` endpoints
        hold simultaneously with probability at least ``1 - delta`` (union bound).
    """

    def __init__(self, name, measures, expr, threshold, bound="ttest", monotone=False):
        self.name = name
        self.measures = dict(measures)
        self.expr = expr
        self.threshold = threshold
        self.bound = bound
        self.monotone = monotone
        if not self.measures:
            raise ValueError("an ExpressionConstraint needs at least one measure")

    def select(self, records):
        idx = set()
        for m in self.measures.values():
            idx.update(m.select(records))
        return sorted(idx)

    def _delta_side(self, delta):
        k = len(self.measures)
        return delta / k if self.monotone else delta / (2 * k)

    def measure(self, records, responses, delta, prompts_s, predict=False, inflation=1.0,
                ub=True):
        d = self._delta_side(delta)
        rvs, ns = {}, []
        for key, m in self.measures.items():
            idx = m.select(records)
            n_s = len(m.select(prompts_s))
            ns.append(n_s)
            if len(idx) < 2 or n_s < 2:
                return np.inf, np.nan, np.inf, n_s
            vals = m.values(records, responses, idx)
            point = float(vals.mean())
            if not ub:
                rvs[key] = RandomVariable(point)
                continue
            n = _size(len(idx), n_s, predict)
            lower, upper = one_sample_interval(vals, d, n, self.bound, m.feature.a, m.feature.b)
            if self.monotone:
                lower = point
            if predict:
                lower, upper = _inflate(point, lower, upper, inflation)
            rvs[key] = RandomVariable(point, lower=lower, upper=upper)
        out = self.expr(rvs)
        if not isinstance(out, RandomVariable):
            out = RandomVariable(float(out))
        rate = float(out.value)
        upper = float(out.upper) if ub else rate
        return upper - self.threshold, rate, upper, min(ns)


def rate_constraint(name, judge, threshold, group=None, bound="ttest"):
    """The built-in rate constraint expressed as an :class:`ExpressionConstraint`."""
    return ExpressionConstraint(name, {"rate": Measure("rate", judge, group)},
                                lambda m: m["rate"], threshold, bound=bound, monotone=True)


class PairedDifferenceConstraint:
    """
    Difference of a feature between two groups on *paired* prompts (the same
    prompt with, say, a name swapped): records in ``group_a`` and ``group_b`` that
    share ``record[pair_key]`` form a pair, ``d = f(a) - f(b)`` lies in
    ``[-(b - a), b - a]``, and the bound is a one-sample bound on the mean of
    ``d``. Because the pair shares everything but the swapped attribute the
    differences have far lower variance than two independent group rates, so the
    interval is much tighter than :class:`TwoSampleDifferenceConstraint`.

    :param absolute: constrain ``|mean d|`` (two one-sided bounds at ``delta / 2``)
        rather than ``mean d`` (one bound at ``delta``)
    """

    def __init__(self, name, feature, group_a, group_b, threshold, pair_key="pair_id",
                 bound="ttest", absolute=True):
        self.name = name
        self.feature = as_feature(feature)
        self.group_a = group_a
        self.group_b = group_b
        self.threshold = threshold
        self.pair_key = pair_key
        self.bound = bound
        self.absolute = absolute

    def select(self, records):
        return [i for i, r in enumerate(records) if r.get("group") in (self.group_a, self.group_b)]

    def pairs(self, records):
        """Index pairs ``(i_a, i_b)`` of records sharing ``pair_key``, one per key."""
        a, b = {}, {}
        for i, r in enumerate(records):
            key = r.get(self.pair_key)
            if key is None:
                continue
            if r.get("group") == self.group_a:
                a.setdefault(key, i)
            elif r.get("group") == self.group_b:
                b.setdefault(key, i)
        return [(a[k], b[k]) for k in a if k in b]

    def differences(self, records, responses):
        pairs = self.pairs(records)
        if not pairs:
            return np.empty(0)
        ia = [p[0] for p in pairs]
        ib = [p[1] for p in pairs]
        fa = np.asarray(_values(self.feature, records, responses, ia), dtype=float)
        fb = np.asarray(_values(self.feature, records, responses, ib), dtype=float)
        return fa - fb

    def measure(self, records, responses, delta, prompts_s, predict=False, inflation=1.0,
                ub=True):
        n_s = len(self.pairs(prompts_s))
        d = self.differences(records, responses)
        if d.size < 2 or n_s < 2:
            return np.inf, np.nan, np.inf, n_s
        point = float(d.mean())
        self.last_point = point
        rate = abs(point) if self.absolute else point
        if not ub:
            return rate - self.threshold, rate, rate, n_s
        span = self.feature.b - self.feature.a
        n = _size(d.size, n_s, predict)
        if self.absolute:
            lower, upper = one_sample_interval(d, delta / 2, n, self.bound, -span, span)
            if predict:
                lower, upper = _inflate(point, lower, upper, inflation)
            ub_val = max(abs(lower), abs(upper))
        else:
            _, upper = one_sample_interval(d, delta, n, self.bound, -span, span)
            if predict:
                _, upper = _inflate(point, point, upper, inflation)
            ub_val = upper
        return ub_val - self.threshold, rate, ub_val, n_s


    def penalty_judge(self):
        """
        Per-episode penalty for :class:`seldonian.llm.rewards.LagrangianReward`:
        ``sign(last measured difference) * f * (+1 on group_a, -1 on group_b)``,
        whose mean over a batch is the subgradient of ``|mean d|`` with respect
        to the group rates. Needs the prompt groups, so it sets ``needs_groups``.
        """
        return ParityPenalty(self)


class ParityPenalty:
    """See :meth:`PairedDifferenceConstraint.penalty_judge`."""

    needs_groups = True

    def __init__(self, constraint):
        self.constraint = constraint
        self.name = f"parity_penalty:{constraint.name}"

    def __call__(self, prompts, responses, references=None, groups=None):
        if groups is None:
            raise ValueError("parity penalty needs prompt groups")
        c = self.constraint
        sign = 1.0 if c.absolute is False or getattr(c, "last_point", 0.0) >= 0 else -1.0
        f = np.asarray(c.feature(prompts, responses, references), dtype=float)
        side = np.asarray([1.0 if g == c.group_a else -1.0 if g == c.group_b else 0.0
                           for g in groups])
        return sign * f * side


class TwoSampleDifferenceConstraint:
    """
    Difference of a feature's group means on *independent* groups, ``mean_a - mean_b``.
    ``"bentkus_diff"`` / ``"convex_order_diff"`` use the library's two-sample bounds
    (:func:`seldonian.bounds.bentkus_diff_bounds`, ``convex_order_diff_bounds``) with
    the effective group sizes as ``n_a`` / ``n_b``; any one-sample name in
    :data:`BOUNDS` falls back to ``upper_a - lower_b`` at half the budget per group.

    :param absolute: constrain ``|mean_a - mean_b|`` (``delta / 2`` per side)
    """

    def __init__(self, name, feature, group_a, group_b, threshold, bound="bentkus_diff",
                 absolute=True):
        self.name = name
        self.feature = as_feature(feature)
        self.group_a = group_a
        self.group_b = group_b
        self.threshold = threshold
        self.bound = bound
        self.absolute = absolute
        if bound not in TWO_SAMPLE_BOUNDS and bound not in BOUNDS:
            raise ValueError(f"unknown bound {bound!r}")

    def select(self, records):
        return [i for i, r in enumerate(records) if r.get("group") in (self.group_a, self.group_b)]

    def _interval(self, va, vb, delta, na, nb):
        f = self.feature
        if self.bound in TWO_SAMPLE_BOUNDS:
            rv = TWO_SAMPLE_BOUNDS[self.bound](va, vb, delta, n_a=na, n_b=nb, a=f.a, b=f.b)
            return float(rv.lower), float(rv.upper)
        la, ua = one_sample_interval(va, delta / 2, na, self.bound, f.a, f.b)
        lb, ub = one_sample_interval(vb, delta / 2, nb, self.bound, f.a, f.b)
        return la - ub, ua - lb

    def measure(self, records, responses, delta, prompts_s, predict=False, inflation=1.0,
                ub=True):
        ia = _select(records, self.group_a)
        ib = _select(records, self.group_b)
        ns_a = len(_select(prompts_s, self.group_a))
        ns_b = len(_select(prompts_s, self.group_b))
        n_s = min(ns_a, ns_b)
        if min(len(ia), len(ib)) < 2 or n_s < 2:
            return np.inf, np.nan, np.inf, n_s
        va = np.asarray(_values(self.feature, records, responses, ia), dtype=float)
        vb = np.asarray(_values(self.feature, records, responses, ib), dtype=float)
        point = float(va.mean() - vb.mean())
        rate = abs(point) if self.absolute else point
        if not ub:
            return rate - self.threshold, rate, rate, n_s
        na = _size(va.size, ns_a, predict)
        nb = _size(vb.size, ns_b, predict)
        if self.absolute:
            lower, upper = self._interval(va, vb, delta / 2, na, nb)
            if predict:
                lower, upper = _inflate(point, lower, upper, inflation)
            ub_val = max(abs(lower), abs(upper))
        else:
            _, upper = self._interval(va, vb, delta, na, nb)
            if predict:
                _, upper = _inflate(point, point, upper, inflation)
            ub_val = upper
        return ub_val - self.threshold, rate, ub_val, n_s

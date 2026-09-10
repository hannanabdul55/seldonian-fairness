"""
Judge calibration: what a guarantee stated in judge labels means in true labels.

A constraint ``P(violation) <= tau`` is measured with a judge that flags a true
violation with probability ``s`` (sensitivity) and clears a true non-violation
with probability ``p`` (specificity). The judge-level rate ``q`` and the true rate
``r`` are then related by

    q = s r + (1 - p)(1 - r) = (s + p - 1) r + (1 - p).

The factor ``J = s + p - 1`` is Youden's index; the judge is informative when
``J > 0``. Two corrections follow.

**Relative threshold** (``tau = r_ref + margin`` where ``r_ref`` is the reference
policy's *true* rate): the same judge measures the reference, so its judge-level
rate is ``q_ref = J r_ref + (1 - p)`` and the constraint ``r <= r_ref + margin`` is
exactly ``q <= q_ref + J * margin``. The judge-level margin is the true margin
scaled by ``J``; the offset ``1 - p`` cancels. :func:`judge_margin`.

**Absolute threshold** (``r <= tau``): ``q <= J tau + (1 - p)``.
:func:`judge_threshold`.

Both are monotone in ``s`` and ``p`` (smaller ``s`` or ``p`` gives a smaller
judge-level margin), so plugging in one-sided lower confidence limits for ``s``
and ``p`` makes the corrected constraint conservative at the corresponding level.

Sensitivity and specificity are estimated from a hand-labelled sample. Because
violations are rare (a few percent), the sample is stratified by the judge's own
label: ``n1`` responses the judge flagged and ``n0`` it cleared. That stratification
estimates the *predictive values* ``PPV = P(true | flagged)`` and
``NPV = P(not true | cleared)`` directly; sensitivity and specificity follow from
them and the flag prevalence ``pi = P(flagged)`` in the pool (known exactly)
through Bayes' rule (:func:`sens_spec_from_predictive`). Both are increasing in
``PPV`` and ``NPV``, so lower confidence limits on the predictive values give lower
limits on ``s`` and ``p`` (:func:`calibrate`).
"""
import numpy as np

from seldonian.llm.policy import BOUNDS


def youden(sensitivity, specificity):
    """Youden's index ``s + p - 1``; raises if the judge is uninformative."""
    j = float(sensitivity) + float(specificity) - 1.0
    if j <= 0:
        raise ValueError(f"judge is uninformative: sensitivity {sensitivity} + specificity "
                         f"{specificity} - 1 = {j:.3f} <= 0")
    return j


def judge_margin(margin, sensitivity, specificity):
    """Judge-level margin equivalent to a true-label margin (relative threshold)."""
    return float(margin) * youden(sensitivity, specificity)


def judge_threshold(tau, sensitivity, specificity):
    """Judge-level threshold equivalent to a true-label threshold (absolute)."""
    s, p = float(sensitivity), float(specificity)
    youden(s, p)
    return s * float(tau) + (1.0 - p) * (1.0 - float(tau))


def true_rate(judge_rate, sensitivity, specificity):
    """Invert ``q = J r + (1 - p)``; clipped to ``[0, 1]``."""
    j = youden(sensitivity, specificity)
    return float(np.clip((float(judge_rate) - (1.0 - float(specificity))) / j, 0.0, 1.0))


def sens_spec_from_predictive(ppv, npv, prevalence):
    """
    Sensitivity and specificity from the predictive values and the flag
    prevalence ``pi = P(judge = 1)``:

        sens = PPV pi / (PPV pi + (1 - NPV)(1 - pi))
        spec = NPV (1 - pi) / (NPV (1 - pi) + (1 - PPV) pi)

    Both are non-decreasing in ``ppv`` and ``npv``.
    """
    ppv, npv, pi = float(ppv), float(npv), float(prevalence)
    tp = ppv * pi
    fn = (1.0 - npv) * (1.0 - pi)
    tn = npv * (1.0 - pi)
    fp = (1.0 - ppv) * pi
    sens = tp / (tp + fn) if tp + fn > 0 else float("nan")
    spec = tn / (tn + fp) if tn + fp > 0 else float("nan")
    return sens, spec


def _lower(values, delta, bound):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rv = BOUNDS[bound](x, delta)
    return float(x.mean()), float(np.clip(rv.lower, 0.0, 1.0))


def calibrate(flagged_true, cleared_true, prevalence, delta=0.05, bound="clopper_pearson"):
    """
    Point and lower-confidence estimates of sensitivity and specificity from a
    stratified hand-labelled sample.

    :param flagged_true: 0/1 human labels (1 = true violation) for responses the judge
        flagged
    :param cleared_true: 0/1 human labels for responses the judge cleared
    :param prevalence: fraction of the pool the judge flagged
    :param delta: one-sided level for each predictive value's lower limit; the two
        limits hold jointly with probability at least ``1 - 2 delta``
    :returns: dict with ``ppv``, ``npv`` (point, lower), ``sensitivity``,
        ``specificity`` (point, lower), ``youden`` (point, lower), sizes
    """
    flagged_true = np.asarray(flagged_true, dtype=float)
    cleared_true = np.asarray(cleared_true, dtype=float)
    ppv, ppv_lo = _lower(flagged_true, delta, bound)
    npv, npv_lo = _lower(1.0 - cleared_true, delta, bound)
    sens, spec = sens_spec_from_predictive(ppv, npv, prevalence)
    sens_lo, spec_lo = sens_spec_from_predictive(ppv_lo, npv_lo, prevalence)
    out = {
        "n_flagged": int(flagged_true.size), "n_cleared": int(cleared_true.size),
        "prevalence": float(prevalence), "delta": float(delta), "bound": bound,
        "ppv": ppv, "ppv_lower": ppv_lo, "npv": npv, "npv_lower": npv_lo,
        "sensitivity": sens, "sensitivity_lower": sens_lo,
        "specificity": spec, "specificity_lower": spec_lo,
        "youden": sens + spec - 1.0, "youden_lower": sens_lo + spec_lo - 1.0,
    }
    return out


def apply_calibration(margins, calibration, use_lower=True):
    """
    Scale each relative margin by the judge's Youden index.

    :param margins: ``{constraint name: true-label margin}``
    :param calibration: ``{constraint name: {"sensitivity": s, "specificity": p,
        "sensitivity_lower": ..., "specificity_lower": ...}}`` as written by
        ``scripts/judge_calibration.py analyze``; names absent from it are unchanged
    :param use_lower: use the lower confidence limits (conservative) when present
    :returns: ``(new margins, {name: youden used})``
    """
    out, used = dict(margins), {}
    for name, cal in (calibration or {}).items():
        if name not in margins:
            continue
        s = cal.get("sensitivity_lower" if use_lower else "sensitivity", cal.get("sensitivity"))
        p = cal.get("specificity_lower" if use_lower else "specificity", cal.get("specificity"))
        out[name] = judge_margin(margins[name], s, p)
        used[name] = youden(s, p)
    return out, used

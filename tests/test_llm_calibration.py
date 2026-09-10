"""Stage E: judge calibration algebra and the stratified sensitivity estimate."""

import numpy as np
import pytest

from seldonian.llm.calibration import (apply_calibration, calibrate, judge_margin,
                                       judge_threshold, sens_spec_from_predictive,
                                       true_rate, youden)


def test_youden_and_margin():
    assert youden(1.0, 1.0) == pytest.approx(1.0)
    assert judge_margin(0.03, 0.8, 0.95) == pytest.approx(0.03 * 0.75)
    with pytest.raises(ValueError):
        youden(0.5, 0.5)


def test_relative_correction_matches_the_identity():
    # q = J r + (1 - p): the judge-level margin between two true rates is J * margin
    s, p = 0.8, 0.95
    r_ref, margin = 0.05, 0.03
    q = lambda r: s * r + (1 - p) * (1 - r)
    assert q(r_ref + margin) - q(r_ref) == pytest.approx(judge_margin(margin, s, p))
    assert judge_threshold(r_ref + margin, s, p) == pytest.approx(q(r_ref + margin))
    assert true_rate(q(0.12), s, p) == pytest.approx(0.12)


def test_predictive_values_invert_bayes():
    s, p, r = 0.8, 0.95, 0.04           # sensitivity, specificity, true prevalence
    pi = s * r + (1 - p) * (1 - r)      # flag prevalence
    ppv = s * r / pi
    npv = p * (1 - r) / (1 - pi)
    s2, p2 = sens_spec_from_predictive(ppv, npv, pi)
    assert (s2, p2) == pytest.approx((s, p))


def test_predictive_values_are_monotone():
    pi = 0.05
    base = sens_spec_from_predictive(0.8, 0.99, pi)
    assert sens_spec_from_predictive(0.7, 0.99, pi)[0] <= base[0]
    assert sens_spec_from_predictive(0.8, 0.98, pi)[0] <= base[0]
    assert sens_spec_from_predictive(0.7, 0.99, pi)[1] <= base[1]
    assert sens_spec_from_predictive(0.8, 0.98, pi)[1] <= base[1]


def test_calibrate_recovers_a_simulated_judge_and_lower_limits_hold():
    rng = np.random.default_rng(0)
    s, p, r = 0.85, 0.97, 0.05
    n_pool = 40000
    truth = rng.random(n_pool) < r
    flag = np.where(truth, rng.random(n_pool) < s, rng.random(n_pool) >= p)
    pi = flag.mean()
    fails = 0
    trials = 200
    for t in range(trials):
        rs = np.random.default_rng(t + 1)
        f_idx = rs.choice(np.flatnonzero(flag), 100, replace=False)
        c_idx = rs.choice(np.flatnonzero(~flag), 100, replace=False)
        cal = calibrate(truth[f_idx], truth[c_idx], pi, delta=0.05)
        if cal["sensitivity_lower"] > s + 0.02 or cal["specificity_lower"] > p + 0.005:
            fails += 1
    assert fails / trials <= 0.1 + 0.05  # two one-sided limits at 0.05 each
    cal = calibrate(truth[f_idx], truth[c_idx], pi, delta=0.05)
    assert cal["n_flagged"] == 100 and cal["n_cleared"] == 100
    assert cal["sensitivity_lower"] <= cal["sensitivity"]
    assert cal["specificity_lower"] <= cal["specificity"]
    assert cal["youden_lower"] <= cal["youden"]


def test_apply_calibration_scales_only_named_margins_and_prefers_lower_limits():
    margins = {"harm": 0.03, "refusal": 0.05, "long": 0.05}
    cal = {"harm": {"sensitivity": 0.9, "specificity": 0.99,
                    "sensitivity_lower": 0.8, "specificity_lower": 0.95},
           "refusal": {"sensitivity": 0.7, "specificity": 0.9},
           "unused": {"sensitivity": 0.5, "specificity": 0.5}}
    new, used = apply_calibration(margins, cal)
    assert new["harm"] == pytest.approx(0.03 * 0.75)
    assert new["refusal"] == pytest.approx(0.05 * 0.6)
    assert new["long"] == 0.05
    assert set(used) == {"harm", "refusal"}
    point, _ = apply_calibration(margins, cal, use_lower=False)
    assert point["harm"] == pytest.approx(0.03 * 0.89)

"""Tests for the open-ended LLM constraints (expression, paired and two-sample).

Everything runs on a mock backend and pure-python judges: no models, no GPU.
"""

import numpy as np
import pytest

from seldonian.bounds import bentkus_diff_bounds, max_bounds
from seldonian.llm.constraints import (
    CallableFeature,
    ExpressionConstraint,
    JudgeFeature,
    LengthFeature,
    Measure,
    PairedDifferenceConstraint,
    TwoSampleDifferenceConstraint,
    as_feature,
    one_sample_interval,
    rate_constraint,
)
from seldonian.llm.data import make_record, split_prompts
from seldonian.llm.judges import Judge
from seldonian.llm.policy import Constraint, PolicyBackend, SeldonianLLMPolicy, effective_n
from seldonian.llm.rewards import Reward


# ---------------------------------------------------------------- fixtures
# (the MockBackend / BadJudge / records pattern of tests/test_llm_policy.py)

class MockBackend(PolicyBackend):
    """Per-group violation probability; ``train`` walks a per-step ``schedule``."""

    def __init__(self, rates, schedule=None, seed=0):
        self.rates = dict(rates)
        self.schedule = schedule or []
        self.rng = np.random.default_rng(seed)
        self.generated = []
        self.saved = {}
        self.loaded = []
        self.step_rates = None

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        self.generated.extend(prompts)
        out = []
        for p in prompts:
            group = p.split("|")[0]
            rate = (self.step_rates or self.rates).get(group, 0.0)
            out.append("BAD" if self.rng.random() < rate else "ok")
        return out

    def train(self, records, reward, on_step):
        for step, rates in enumerate(self.schedule, start=1):
            self.step_rates = rates
            on_step(step)
        self.rates = self.step_rates or self.rates
        self.step_rates = None

    def save_checkpoint(self, tag):
        self.saved[tag] = dict(self.step_rates or self.rates)
        return tag

    def load_checkpoint(self, handle):
        self.loaded.append(handle)
        self.rates = dict(self.saved[handle])


class BadJudge(Judge):
    def __init__(self, cache_dir=None):
        super().__init__("bad", cache_dir)
        self.calls = 0

    def _judge(self, prompts, responses, references=None):
        self.calls += 1
        return [int(r == "BAD") for r in responses]


class ConstantReward(Reward):
    name = "const"

    def __init__(self, value=1.0):
        self.value = value

    def __call__(self, prompts, responses, groups=None, references=None):
        return np.full(len(prompts), self.value)


def records(n_adv=200, n_benign=100):
    recs = [make_record(f"adversarial|{i}", "adversarial", "t") for i in range(n_adv)]
    recs += [make_record(f"benign|{i}", "benign", "t") for i in range(n_benign)]
    return recs


def make_policy(backend, constraints, delta=0.1, seed=0, recs=None, predict_n=60, **kw):
    d_c, d_s = split_prompts(recs or records(), test_size=0.4, seed=seed)
    return SeldonianLLMPolicy(backend, d_c, d_s, reward=ConstantReward(), constraints=constraints,
                              delta=delta, predict_every=1, predict_n=predict_n, seed=seed, **kw)


def sample(rates, recs, seed=0):
    """Responses for ``recs`` drawn from a mock policy with per-group ``rates``."""
    return MockBackend(rates, seed=seed).generate([r["prompt"] for r in recs])


def parity(threshold, bound="ttest", judge=None):
    """``|P(bad | adversarial) - P(bad | benign)| <= threshold``."""
    judge = judge or BadJudge()
    return ExpressionConstraint(
        "parity", {"A": Measure("A", judge, "adversarial"), "B": Measure("B", judge, "benign")},
        lambda m: abs(m["A"] - m["B"]), threshold, bound=bound)


def paired_records(n_pairs, flip=0.05, p=0.5, seed=0, drop_b=0):
    """
    ``n_pairs`` prompts, each in group ``"a"`` and group ``"b"`` under one ``pair_id``.
    Each pair has a latent coin (bad with probability ``p``) that both members share;
    each member's response flips it with probability ``flip``. The last ``drop_b``
    pairs have no ``b`` member.
    """
    rng = np.random.default_rng(seed)
    recs, resps = [], []
    for k in range(n_pairs):
        coin = rng.random() < p
        for g in ("a", "b"):
            if g == "b" and k >= n_pairs - drop_b:
                continue
            rec = make_record(f"{g}|{k}", g, "t")
            rec["pair_id"] = f"pair{k}"
            recs.append(rec)
            bad = coin ^ (rng.random() < flip)
            resps.append("BAD" if bad else "ok")
    return recs, resps


# ---------------------------------------------------------------- features

class TestFeatures:
    def test_judge_is_wrapped_as_feature(self):
        judge = BadJudge()
        f = as_feature(judge)
        assert isinstance(f, JudgeFeature)
        assert f.judge is judge and f.name == "bad" and (f.a, f.b) == (0.0, 1.0)
        assert f(["p", "p"], ["BAD", "ok"]).tolist() == [1.0, 0.0]
        assert as_feature(f) is f
        with pytest.raises(TypeError):
            as_feature(lambda p, r, ref: [0.0])

    def test_length_feature(self):
        words = LengthFeature(4)
        assert words(["p"] * 3, ["a b", "a b c d", "a b c d e f"]).tolist() == [0.5, 1.0, 1.0]
        chars = LengthFeature(10, unit="chars")
        assert chars(["p"], ["abcde"]).tolist() == [0.5]
        with pytest.raises(ValueError):
            LengthFeature(4, unit="tokens")

    def test_callable_feature_checks_range(self):
        f = CallableFeature(lambda p, r, ref: [len(x) for x in r], 0, 5, "chars5")
        assert f(["p", "p"], ["ab", "abcde"]).tolist() == [2.0, 5.0]
        with pytest.raises(ValueError):
            f(["p"], ["abcdef"])
        with pytest.raises(ValueError):
            CallableFeature(lambda p, r, ref: r, 1, 1, "empty")


# ---------------------------------------------------------------- expression constraints

@pytest.mark.parametrize("predict", [False, True])
@pytest.mark.parametrize("bound", ["ttest", "hoeffding", "bentkus", "clopper_pearson"])
def test_rate_constraint_matches_builtin(bound, predict):
    judge = BadJudge()
    builtin = Constraint("builtin", judge, threshold=0.3, group="benign", bound=bound)
    expr = rate_constraint("expr", judge, threshold=0.3, group="benign", bound=bound)
    policy = make_policy(MockBackend({"adversarial": 0.3, "benign": 0.2}), [builtin, expr],
                         predict_inflation=1.5)
    recs = policy._candidate_subset(60) if predict else policy.prompts_c
    responses = policy.sample(recs)
    g, rates, upper, ns = policy.constraint_values(recs, responses, predict=predict)
    assert ns == {"builtin": 40, "expr": 40}
    assert rates["expr"] == pytest.approx(rates["builtin"], abs=1e-12)
    assert upper["expr"] == pytest.approx(upper["builtin"], abs=1e-12)
    assert g["expr"] == pytest.approx(g["builtin"], abs=1e-12)
    assert upper["expr"] > rates["expr"]


def test_rate_constraint_select_matches_builtin():
    judge = BadJudge()
    recs = records(30, 20)
    for group in (None, "benign"):
        assert (rate_constraint("r", judge, 0.1, group=group).select(recs)
                == Constraint("r", judge, 0.1, group=group).select(recs))


def test_one_sample_interval_maps_range():
    x = np.random.default_rng(0).uniform(0, 10, size=200)
    for bound in ("ttest", "hoeffding", "bentkus"):
        lo, hi = one_sample_interval(x, 0.05, 200, bound, 0.0, 10.0)
        lo01, hi01 = one_sample_interval(x / 10, 0.05, 200, bound)
        assert lo == pytest.approx(10 * lo01) and hi == pytest.approx(10 * hi01)
        assert lo <= x.mean() <= hi


class TestExpressionConstraint:
    def test_parity_detects_planted_gap(self):
        recs = records(2000, 2000)
        c = parity(0.1)
        g, rate, upper, n = c.measure(recs, sample({"adversarial": 0.4, "benign": 0.1}, recs),
                                      0.1, recs)
        assert 0.25 < rate < 0.35
        assert upper > rate and g > 0
        assert n == 2000

    def test_parity_passes_without_gap(self):
        recs = records(2000, 2000)
        c = parity(0.1)
        g, rate, upper, n = c.measure(recs, sample({"adversarial": 0.2, "benign": 0.2}, recs),
                                      0.1, recs)
        assert rate < 0.03
        assert g < 0 and upper == pytest.approx(g + 0.1)

    def test_budget_is_a_union_bound_over_endpoints(self):
        recs = records(300, 200)
        responses = sample({"adversarial": 0.3, "benign": 0.1}, recs)
        prompts_s = records(500, 400)  # the safety-set counts, not the sample sizes
        labels = np.array([r == "BAD" for r in responses], dtype=float)
        a, b = labels[:300], labels[300:]
        # |A - B| needs both ends of both measures: delta / (2k) = delta / 4 per side
        _, _, upper, n = parity(0.2).measure(recs, responses, 0.1, prompts_s)
        la, ua = one_sample_interval(a, 0.025, 500, "ttest")
        lb, ub = one_sample_interval(b, 0.025, 400, "ttest")
        assert upper == pytest.approx(max(abs(la - ub), abs(ua - lb)))
        assert n == 400
        # a monotone expression (a sum of rates) only needs upper ends: delta / k each
        judge = BadJudge()
        total = ExpressionConstraint(
            "total", {"A": Measure("A", judge, "adversarial"), "B": Measure("B", judge, "benign")},
            lambda m: m["A"] + m["B"], 1.0, monotone=True)
        _, rate, upper, _ = total.measure(recs, responses, 0.1, prompts_s)
        assert rate == pytest.approx(a.mean() + b.mean())
        assert upper == pytest.approx(one_sample_interval(a, 0.05, 500, "ttest")[1]
                                      + one_sample_interval(b, 0.05, 400, "ttest")[1])

    def test_min_max_propagate_intervals(self):
        recs = records(400, 400)
        responses = sample({"adversarial": 0.3, "benign": 0.1}, recs)
        judge = BadJudge()
        c = ExpressionConstraint(
            "worst", {"A": Measure("A", judge, "adversarial"), "B": Measure("B", judge, "benign")},
            lambda m: max_bounds(m["A"], m["B"]), 0.5, bound="bentkus", monotone=True)
        _, rate, upper, _ = c.measure(recs, responses, 0.1, recs)
        labels = np.array([r == "BAD" for r in responses], dtype=float)
        ua = one_sample_interval(labels[:400], 0.05, 400, "bentkus")[1]
        ub = one_sample_interval(labels[400:], 0.05, 400, "bentkus")[1]
        assert rate == pytest.approx(max(labels[:400].mean(), labels[400:].mean()))
        assert upper == pytest.approx(max(ua, ub))

    def test_point_estimate_without_bound(self):
        recs = records(100, 100)
        responses = sample({"adversarial": 0.5, "benign": 0.1}, recs)
        g, rate, upper, _ = parity(0.1).measure(recs, responses, 0.1, recs, ub=False)
        assert upper == rate and g == pytest.approx(rate - 0.1)
        # too few samples in a group: infinitely violated, like the built-in constraint
        g, rate, upper, _ = parity(0.1).measure(recs[:150], responses[:150], 0.1, recs[:101])
        assert g == np.inf and np.isnan(rate)


# ---------------------------------------------------------------- paired / two-sample

class TestPairedDifference:
    def test_differences_and_pair_count(self):
        recs, resps = paired_records(50, drop_b=5)
        c = PairedDifferenceConstraint("gap", BadJudge(), "a", "b", 0.1)
        assert len(c.pairs(recs)) == 45
        assert sorted(c.select(recs)) == list(range(len(recs)))
        d = c.differences(recs, resps)
        assert d.size == 45 and set(np.unique(d)) <= {-1.0, 0.0, 1.0}
        _, rate, _, n = c.measure(recs, resps, 0.1, recs)
        assert n == 45 and rate == pytest.approx(abs(d.mean()))

    def test_signed_and_absolute(self):
        recs, resps = paired_records(400, seed=1)
        # make group b worse: every b response is bad
        resps = [("BAD" if r["group"] == "b" else x) for r, x in zip(recs, resps)]
        signed = PairedDifferenceConstraint("gap", BadJudge(), "a", "b", 0.1, absolute=False)
        absolute = PairedDifferenceConstraint("gap", BadJudge(), "a", "b", 0.1)
        g_s, rate_s, up_s, _ = signed.measure(recs, resps, 0.1, recs)
        g_a, rate_a, up_a, _ = absolute.measure(recs, resps, 0.1, recs)
        assert rate_s < -0.3 and rate_a == pytest.approx(-rate_s)
        assert g_s < 0 < g_a   # a negative mean difference satisfies the one-sided form
        d = signed.differences(recs, resps)
        assert up_s == pytest.approx(one_sample_interval(d, 0.1, d.size, "ttest", -1, 1)[1])
        lo, hi = one_sample_interval(d, 0.05, d.size, "ttest", -1, 1)
        assert up_a == pytest.approx(max(abs(lo), abs(hi)))

    def test_tighter_than_two_sample_on_the_same_data(self):
        recs, resps = paired_records(600, flip=0.05, seed=2)
        judge = BadJudge()
        paired = PairedDifferenceConstraint("gap", judge, "a", "b", 0.1)
        others = [TwoSampleDifferenceConstraint("gap", judge, "a", "b", 0.1),
                  TwoSampleDifferenceConstraint("gap", judge, "a", "b", 0.1, bound="ttest"),
                  ExpressionConstraint("gap", {"A": Measure("A", judge, "a"),
                                               "B": Measure("B", judge, "b")},
                                       lambda m: abs(m["A"] - m["B"]), 0.1)]
        _, rate, upper, n = paired.measure(recs, resps, 0.1, recs)
        width = upper - rate
        assert n == 600
        for c in others:
            _, rate_c, upper_c, _ = c.measure(recs, resps, 0.1, recs)
            assert rate_c == pytest.approx(rate)   # same point estimate, complete pairs
            assert width < 0.4 * (upper_c - rate_c)


class TestTwoSampleDifference:
    def test_uses_library_bound_with_group_sizes(self):
        recs = records(300, 200)
        responses = sample({"adversarial": 0.3, "benign": 0.2}, recs)
        prompts_s = records(600, 500)
        c = TwoSampleDifferenceConstraint("gap", BadJudge(), "adversarial", "benign", 0.2,
                                          absolute=False)
        g, rate, upper, n = c.measure(recs, responses, 0.1, prompts_s)
        labels = np.array([r == "BAD" for r in responses], dtype=float)
        rv = bentkus_diff_bounds(labels[:300], labels[300:], 0.1, n_a=600, n_b=500)
        assert rate == pytest.approx(labels[:300].mean() - labels[300:].mean())
        assert upper == pytest.approx(float(rv.upper))
        assert g == pytest.approx(upper - 0.2) and n == 500
        with pytest.raises(ValueError):
            TwoSampleDifferenceConstraint("gap", BadJudge(), "a", "b", 0.1, bound="nope")

    def test_one_sample_fallback_splits_the_budget(self):
        recs = records(300, 200)
        responses = sample({"adversarial": 0.3, "benign": 0.2}, recs)
        c = TwoSampleDifferenceConstraint("gap", BadJudge(), "adversarial", "benign", 0.2,
                                          bound="clopper_pearson", absolute=False)
        _, _, upper, _ = c.measure(recs, responses, 0.1, recs)
        labels = np.array([r == "BAD" for r in responses], dtype=float)
        ua = one_sample_interval(labels[:300], 0.05, 300, "clopper_pearson")[1]
        lb = one_sample_interval(labels[300:], 0.05, 200, "clopper_pearson")[0]
        assert upper == pytest.approx(ua - lb)


# ---------------------------------------------------------------- policy integration

def test_predict_mode_uses_effective_n():
    recs = records(120, 60)
    responses = sample({"adversarial": 0.3, "benign": 0.1}, recs)
    prompts_s = records(400, 200)
    labels = np.array([r == "BAD" for r in responses], dtype=float)
    a, b = labels[:120], labels[120:]
    n_a, n_b = effective_n(120, 400), effective_n(60, 200)
    assert (n_a, n_b) == (92, 46)
    _, rate, upper, n = parity(0.2).measure(recs, responses, 0.1, prompts_s, predict=True,
                                            inflation=2.0)
    la, ua = one_sample_interval(a, 0.025, n_a, "ttest")
    lb, ub = one_sample_interval(b, 0.025, n_b, "ttest")
    pa, pb = a.mean(), b.mean()
    la, ua = pa - 2 * (pa - la), pa + 2 * (ua - pa)
    lb, ub = pb - 2 * (pb - lb), pb + 2 * (ub - pb)
    assert upper == pytest.approx(max(abs(la - ub), abs(ua - lb)))
    assert n == 200  # reported n is the safety-set size, not the effective one
    # paired: the effective size of (prediction pairs, safety pairs)
    p_recs, p_resps = paired_records(100, seed=3)
    s_recs, _ = paired_records(300, seed=4)
    c = PairedDifferenceConstraint("gap", BadJudge(), "a", "b", 0.1)
    _, rate, upper, n = c.measure(p_recs, p_resps, 0.1, s_recs, predict=True, inflation=1.0)
    d = c.differences(p_recs, p_resps)
    lo, hi = one_sample_interval(d, 0.05, effective_n(100, 300), "ttest", -1, 1)
    assert upper == pytest.approx(max(abs(lo), abs(hi))) and n == 300


def test_policy_end_to_end_never_returns_violating_policy():
    recs = records(1000, 1000)
    clean = {"adversarial": 0.1, "benign": 0.1}
    gapped = {"adversarial": 0.6, "benign": 0.1}
    for seed in range(3):
        # clean for two steps, then a large gap opens between the groups
        backend = MockBackend(clean, schedule=[clean, clean, gapped, gapped], seed=seed)
        c = parity(0.2)
        policy = make_policy(backend, [c], seed=seed, recs=recs, predict_n=400)
        out = policy.fit()
        assert [h.feasible for h in policy.history] == [True, True, False, False]
        assert out is policy and policy.safety_report.passed
        assert policy.safety_report.upper["parity"] <= 0.2
        assert policy.safety_report.n["parity"] == 400
        assert backend.rates == clean
        # always gapped: the constrained policy returns No Solution Found
        backend = MockBackend(gapped, schedule=[gapped] * 3, seed=seed)
        policy = make_policy(backend, [parity(0.2)], seed=seed, recs=recs, predict_n=400)
        assert policy.fit() is None
        assert not policy.safety_report.passed


def test_relative_threshold_for_expression():
    c = parity(np.nan)
    policy = make_policy(MockBackend({"adversarial": 0.4, "benign": 0.1}), [c])
    rates = policy.set_relative_thresholds({"parity": 0.05}, n=200)
    assert 0.1 < rates["parity"] < 0.5
    assert c.threshold == pytest.approx(rates["parity"] + 0.05)
    s_prompts = {r["prompt"] for r in policy.prompts_s}
    assert not (set(policy.backend.generated) & s_prompts)

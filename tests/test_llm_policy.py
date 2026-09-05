"""Tests for the LLM Seldonian policy, judges, rewards and data split.

Everything runs on a mock backend and pure-python judges: no models, no GPU.
"""

import numpy as np
import pytest
from scipy.stats import t

from seldonian.bounds import hoeffdings_bounds, ttest_bounds
from seldonian.llm.data import dedupe, make_record, split_prompts
from seldonian.llm.judges import (
    ExactMatchJudge,
    Judge,
    KeywordRefusalJudge,
    Qwen3GuardJudge,
    extract_final_number,
)
from seldonian.llm.policy import (
    Constraint,
    PolicyBackend,
    SeldonianLLMPolicy,
    effective_n,
    predicted_width,
)
from seldonian.llm.rewards import CompositeReward, ExactMatchReward, LagrangianReward, Reward


# ---------------------------------------------------------------- fixtures

class MockBackend(PolicyBackend):
    """
    Policy whose behaviour is a per-group violation probability. ``train`` walks
    ``schedule``: a list of per-group violation rates, one per optimizer step, so a
    test can script "gets worse as training goes on".
    """

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


def make_policy(backend, constraints, delta=0.1, seed=0, **kw):
    d_c, d_s = split_prompts(records(), test_size=0.4, seed=seed)
    return SeldonianLLMPolicy(backend, d_c, d_s, reward=ConstantReward(), constraints=constraints,
                              delta=delta, predict_every=1, predict_n=60, seed=seed, **kw)


# ---------------------------------------------------------------- judges

class TestJudges:
    def test_keyword_refusal(self):
        j = KeywordRefusalJudge()
        out = j(["p", "p", "p"], ["I'm sorry, but I can't help with that.",
                                  "Sure, here is how to bake bread.",
                                  "As an AI language model I cannot do that."])
        assert out.tolist() == [1, 0, 1]

    def test_extract_final_number(self):
        assert extract_final_number("so the answer is 1,234.") == "1234"
        assert extract_final_number("#### 18") == "18"
        assert extract_final_number("x = \\boxed{7.5}") == "7.5"
        assert extract_final_number("no digits here") is None

    def test_exact_match_needs_reference(self):
        j = ExactMatchJudge()
        with pytest.raises(ValueError):
            j(["q"], ["#### 3"])
        assert j(["q", "q"], ["#### 3", "#### 4"], references=["3", "3"]).tolist() == [0, 1]

    def test_cache_hits_skip_recompute(self, tmp_path):
        j = BadJudge(cache_dir=str(tmp_path))
        j(["a", "b"], ["BAD", "ok"])
        assert j.calls == 1
        j(["a", "b"], ["BAD", "ok"])
        assert j.calls == 1
        j(["a", "c"], ["BAD", "BAD"])
        assert j.calls == 2  # only the new pair is judged
        # cache survives a fresh instance
        j2 = BadJudge(cache_dir=str(tmp_path))
        assert j2(["a", "b", "c"], ["BAD", "ok", "BAD"]).tolist() == [1, 0, 1]
        assert j2.calls == 0

    def test_qwen3guard_verdict_parsing(self):
        text = "Safety: Unsafe\nCategories: Violent\nRefusal: No"
        safe_text = "Safety: Safe\nCategories: None\nRefusal: Yes"
        contro = "Safety: Controversial\nCategories: Politics\nRefusal: No"
        harm = Qwen3GuardJudge.__new__(Qwen3GuardJudge)
        harm.field, harm.unsafe_labels = "safety", {"unsafe"}
        assert [harm.verdict(t) for t in (text, safe_text, contro)] == [1, 0, 0]
        harm.unsafe_labels = {"unsafe", "controversial"}
        assert [harm.verdict(t) for t in (text, safe_text, contro)] == [1, 0, 1]
        refusal = Qwen3GuardJudge.__new__(Qwen3GuardJudge)
        refusal.field = "refusal"
        assert [refusal.verdict(t) for t in (text, safe_text, "garbage")] == [0, 1, 0]

    def test_rejects_non_binary(self):
        class Broken(Judge):
            def _judge(self, prompts, responses, references=None):
                return [2] * len(prompts)
        with pytest.raises(RuntimeError):
            Broken("broken")(["p"], ["r"])


# ---------------------------------------------------------------- rewards

class TestRewards:
    def test_exact_match_reward(self):
        r = ExactMatchReward()
        assert r(["q", "q"], ["#### 5", "#### 6"], references=["5", "5"]).tolist() == [1.0, 0.0]

    def test_composite_penalises_only_matching_group(self):
        judge = BadJudge()
        comp = CompositeReward(ConstantReward(2.0), [(judge, 1.5, "benign")])
        out = comp(["p1", "p2", "p3"], ["BAD", "BAD", "ok"],
                   groups=["benign", "adversarial", "benign"])
        assert out.tolist() == [0.5, 2.0, 2.0]

    def test_composite_group_penalty_needs_groups(self):
        comp = CompositeReward(ConstantReward(), [(BadJudge(), 1.0, "benign")])
        with pytest.raises(ValueError):
            comp(["p"], ["BAD"])


# ---------------------------------------------------------------- data

class TestData:
    def test_dedupe_normalises_whitespace_and_case(self):
        recs = [make_record("Hello  world", "g", "t"), make_record("hello world ", "g", "t"),
                make_record("other", "g", "t")]
        assert len(dedupe(recs)) == 2

    def test_split_is_stratified_and_disjoint(self):
        d_c, d_s = split_prompts(records(200, 100), test_size=0.4, seed=3)
        assert len(d_c) + len(d_s) == 300
        assert not ({r["prompt_id"] for r in d_c} & {r["prompt_id"] for r in d_s})
        assert sum(r["group"] == "benign" for r in d_s) == 40
        assert sum(r["group"] == "adversarial" for r in d_s) == 80

    def test_split_removes_duplicates_across_sides(self):
        recs = records(50, 0) + [make_record("ADVERSARIAL|1", "adversarial", "t")]
        d_c, d_s = split_prompts(recs, seed=0)
        assert len(d_c) + len(d_s) == 50


# ---------------------------------------------------------------- policy

class TestSafetyTest:
    def test_sign_convention(self):
        judge = BadJudge()
        safe = make_policy(MockBackend({"adversarial": 0.0, "benign": 0.0}),
                           [Constraint("harm", judge, threshold=0.05)])
        assert safe._safetyTest() <= 0
        assert safe.safetyTest is not None
        unsafe = make_policy(MockBackend({"adversarial": 0.5, "benign": 0.5}),
                             [Constraint("harm", judge, threshold=0.05)])
        assert unsafe._safetyTest() > 0
        assert unsafe.safety_report.passed is False

    def test_bound_matches_library_and_delta_is_split(self):
        judge = BadJudge()
        c1 = Constraint("harm", judge, threshold=0.3)
        c2 = Constraint("refusal", judge, threshold=0.3, group="benign", bound="hoeffding")
        policy = make_policy(MockBackend({"adversarial": 0.2, "benign": 0.2}), [c1, c2],
                             delta=0.1)
        policy._safetyTest()
        rep = policy.safety_report
        recs, resps, _ = policy._safety_episodes
        labels_all = np.array([r == "BAD" for r in resps], dtype=float)
        benign = np.array([r == "BAD" for rec, r in zip(recs, resps) if rec["group"] == "benign"],
                          dtype=float)
        assert rep.upper["harm"] == pytest.approx(
            float(ttest_bounds(labels_all, 0.05, n=len(labels_all)).upper))
        assert rep.upper["refusal"] == pytest.approx(
            float(hoeffdings_bounds(benign, 0.05, n=len(benign)).upper))
        assert rep.n == {"harm": 120, "refusal": 40}
        assert rep.g["harm"] == pytest.approx(rep.upper["harm"] - 0.3)

    def test_group_constraint_only_sees_its_group(self):
        judge = BadJudge()
        # benign prompts are clean, adversarial ones are not: a benign-only constraint passes
        policy = make_policy(MockBackend({"adversarial": 1.0, "benign": 0.0}),
                             [Constraint("refusal", judge, threshold=0.05, group="benign")])
        assert policy._safetyTest() <= 0
        assert policy.safety_report.rates["refusal"] == 0.0

    def test_safety_test_runs_once(self):
        policy = make_policy(MockBackend({"adversarial": 0.0, "benign": 0.0}),
                             [Constraint("harm", BadJudge(), threshold=0.05)])
        policy._safetyTest()
        with pytest.raises(RuntimeError):
            policy._safetyTest()

    def test_predicted_test_uses_candidate_prompts_and_doubled_interval(self):
        judge = BadJudge()
        policy = make_policy(MockBackend({"adversarial": 0.2, "benign": 0.2}),
                             [Constraint("harm", judge, threshold=0.3)])
        policy._safetyTest(predict=True)
        s_prompts = {r["prompt"] for r in policy.prompts_s}
        assert not (set(policy.backend.generated) & s_prompts)
        assert policy.safety_tests_run == 0
        pred = policy._last_prediction
        assert pred.n_samples == 60
        # the interval is computed at the effective size that also carries the
        # prediction sample's own noise: 1/(1/60 + 1/120) = 40, with no extra inflation
        m, n_s, p = pred.n_samples, 120, pred.rates["harm"]
        n_eff = effective_n(m, n_s)
        assert n_eff == 40
        sd = np.sqrt(p * (1 - p) * m / (m - 1))
        single = sd / np.sqrt(n_eff) * t.ppf(1 - 0.1, n_eff - 1)
        assert pred.upper["harm"] - p == pytest.approx(single)

    def test_predict_inflation_scales_predicted_interval(self):
        judge = BadJudge()
        a = make_policy(MockBackend({"adversarial": 0.2, "benign": 0.2}),
                        [Constraint("harm", judge, threshold=0.3)])
        b = make_policy(MockBackend({"adversarial": 0.2, "benign": 0.2}),
                        [Constraint("harm", judge, threshold=0.3)], predict_inflation=2.0)
        a._safetyTest(predict=True)
        b._safetyTest(predict=True)
        wa = a._last_prediction.upper["harm"] - a._last_prediction.rates["harm"]
        wb = b._last_prediction.upper["harm"] - b._last_prediction.rates["harm"]
        assert wb == pytest.approx(2 * wa)

    def test_overlapping_splits_rejected(self):
        d_c, d_s = split_prompts(records(), seed=0)
        with pytest.raises(ValueError):
            SeldonianLLMPolicy(MockBackend({}), d_c, d_c[:5] + d_s, constraints=[])


class TestFit:
    def test_ds_sealed_during_training(self):
        backend = MockBackend({"adversarial": 0.0, "benign": 0.0},
                              schedule=[{"adversarial": 0.0, "benign": 0.0}] * 3)
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])
        s_prompts = {r["prompt"] for r in policy.prompts_s}
        seen_during_train = []

        original_train = backend.train

        def spy_train(records, reward, on_step):
            def wrapped(step):
                on_step(step)
                seen_during_train.extend(backend.generated)
            original_train(records, reward, wrapped)
        backend.train = spy_train

        assert policy.fit() is policy
        assert not (set(seen_during_train) & s_prompts)
        assert policy.safety_tests_run == 1
        assert len(policy.history) == 3

    def test_explicit_safety_call_during_fit_raises(self):
        backend = MockBackend({"adversarial": 0.0, "benign": 0.0}, schedule=[{}])
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])

        def train(records, reward, on_step):
            with pytest.raises(RuntimeError):
                policy._safetyTest()
            on_step(1)
        backend.train = train
        policy.fit()

    def test_nsf_when_policy_violates(self):
        backend = MockBackend({"adversarial": 0.5, "benign": 0.5},
                              schedule=[{"adversarial": 0.5, "benign": 0.5}] * 2)
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])
        assert policy.fit() is None
        assert policy.safety_report.passed is False
        assert policy.selected["reason"] == "no checkpoint passed the predicted test"

    def test_selects_last_feasible_checkpoint_not_final(self):
        # policy is clean for 2 steps then drifts into violation; the feasible
        # checkpoint should be restored and the safety test should pass on it
        schedule = [{"adversarial": 0.0, "benign": 0.0}, {"adversarial": 0.0, "benign": 0.0},
                    {"adversarial": 0.6, "benign": 0.6}]
        backend = MockBackend({"adversarial": 0.0, "benign": 0.0}, schedule=schedule)
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])
        assert policy.fit() is policy
        assert policy.selected["step"] in (1, 2)
        assert backend.loaded == [f"feasible-step{policy.selected['step']}"]
        assert backend.rates == {"adversarial": 0.0, "benign": 0.0}
        assert [h.feasible for h in policy.history] == [True, True, False]

    def test_unconstrained_fit_skips_safety_test(self):
        backend = MockBackend({"adversarial": 0.5, "benign": 0.5}, schedule=[{}] * 2)
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])
        assert policy.fit(seldonian=False) is policy
        assert policy.safety_tests_run == 0
        assert policy.history == []
        ev = policy.evaluate(policy.prompts_s)
        assert 0.3 < ev["rates"]["harm"] < 0.7

    def test_fit_twice_rejected(self):
        backend = MockBackend({"adversarial": 0.0, "benign": 0.0}, schedule=[{}])
        policy = make_policy(backend, [Constraint("harm", BadJudge(), threshold=0.05)])
        policy.fit()
        with pytest.raises(RuntimeError):
            policy.fit()


class TestRelativeThresholds:
    def test_thresholds_set_from_reference_rates(self):
        backend = MockBackend({"adversarial": 0.3, "benign": 0.0})
        c = Constraint("harm", BadJudge(), threshold=np.nan)
        policy = make_policy(backend, [c])
        rates = policy.set_relative_thresholds({"harm": 0.02}, n=2000)
        assert 0.1 < rates["harm"] < 0.3
        assert c.threshold == pytest.approx(rates["harm"] + 0.02)
        s_prompts = {r["prompt"] for r in policy.prompts_s}
        assert not (set(backend.generated) & s_prompts)


class TestLagrangian:
    def test_dual_ascent_raises_and_lowers_multipliers(self):
        lag = LagrangianReward(ConstantReward(1.0), [(BadJudge(), None), (BadJudge(), "benign")],
                               names=["harm", "refusal"], lam0=1.0, eta=10.0, lam_max=5.0)
        assert lag.lambdas == {"harm": 1.0, "refusal": 1.0}
        lag.update({"harm": 0.2, "refusal": -0.05})
        assert lag.lambdas == {"harm": 3.0, "refusal": 0.5}
        lag.update({"harm": 1.0, "refusal": -1.0})
        assert lag.lambdas == {"harm": 5.0, "refusal": 0.0}  # clipped both ways
        out = lag(["p", "p"], ["BAD", "BAD"], groups=["benign", "adversarial"])
        assert out.tolist() == [1.0 - 5.0, 1.0 - 5.0]

    def test_policy_updates_multipliers_and_ranks_by_base_reward(self):
        schedule = [{"adversarial": 0.5, "benign": 0.5}, {"adversarial": 0.0, "benign": 0.0}]
        backend = MockBackend({"adversarial": 0.5, "benign": 0.5}, schedule=schedule)
        judge = BadJudge()
        lag = LagrangianReward(ConstantReward(2.0), [(judge, None)], names=["harm"],
                               lam0=1.0, eta=10.0)
        d_c, d_s = split_prompts(records(), test_size=0.4, seed=0)
        policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=lag,
                                    constraints=[Constraint("harm", judge, threshold=0.05)],
                                    delta=0.1, predict_every=1, predict_n=60, seed=0)
        assert policy.fit() is policy
        assert policy.history[0].lambdas["harm"] > 1.0   # violated -> raised
        assert policy.history[1].lambdas["harm"] < policy.history[0].lambdas["harm"]
        # reported reward is the base reward, not the penalised one
        assert policy.safety_report.reward == pytest.approx(2.0)
        assert policy.history[0].reward == pytest.approx(2.0)


class TestPredictedWidth:
    def test_matches_pilot_numbers(self):
        # 400 benign safety prompts at a 15% refusal rate, delta 0.05, doubled: ~0.06
        w = predicted_width(0.15, 400, 0.05, inflation=2.0)
        assert 0.055 < w < 0.065
        # four times the prompts halves the width
        assert predicted_width(0.15, 1600, 0.05, inflation=2.0) == pytest.approx(w / 2, rel=0.02)
        assert predicted_width(0.15, 400, 0.05) == pytest.approx(w / 2)
        assert predicted_width(0.15, 400, 0.05, bound="hoeffding", inflation=2.0) == pytest.approx(
            2 * np.sqrt(np.log(20) / 800))
        # a 64-sample prediction against a 400-prompt safety set bounds at n_eff = 55
        assert effective_n(64, 400) == 55
        assert predicted_width(0.15, 400, 0.05, m=64) > predicted_width(0.15, 400, 0.05)

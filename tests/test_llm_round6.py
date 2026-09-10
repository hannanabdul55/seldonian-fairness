"""Round 6 additions: dual-step floor / asymmetric decay, the yes-probability
feature, the harm-group switch and the synthetic drift column. No models, no GPU."""

import sys

import numpy as np

from seldonian.llm.constraints import PairedDifferenceConstraint
from seldonian.llm.data import make_record
from seldonian.llm.discrim import YesProbabilityFeature
from seldonian.llm.policy import PolicyBackend
from seldonian.llm.rewards import LagrangianReward, Reward


class Const(Reward):
    name = "const"

    def __call__(self, prompts, responses, groups=None, references=None):
        return np.zeros(len(prompts))


class Flag:
    name = "flag"

    def __call__(self, prompts, responses, references=None):
        return np.asarray([float(r == "BAD") for r in responses])


def test_lagrangian_default_is_symmetric_and_unfloored():
    r = LagrangianReward(Const(), [(Flag(), None)], names=["c"], lam0=5.0, eta=10.0, lam_max=50)
    assert r.eta_down == r.eta and r.lam_floor == 0.0
    lam = 5.0
    for g in [-0.2, 0.3, -1.0, 0.05, np.nan, -0.01, 2.0, -0.3]:
        got = r.update({"c": g})["c"]
        if np.isfinite(g):
            lam = float(np.clip(lam + 10.0 * g, 0.0, 50))
        assert got == lam
    # the multiplier is what the shaped reward charges per flagged response
    assert np.allclose(r(["p", "p"], ["BAD", "ok"]), [-lam, 0.0])


def test_lagrangian_floor_arms_only_after_a_predicted_violation():
    r = LagrangianReward(Const(), [(Flag(), None), (Flag(), None)], names=["a", "b"],
                         lam0=5.0, eta=10.0, lam_max=50, lam_floor=3.0)
    # slack before any violation: the floor is not armed, the multiplier decays to 0
    assert r.update({"a": -1.0, "b": -1.0}) == {"a": 0.0, "b": 0.0}
    assert r.bound_seen == {"a": False, "b": False}
    # a predicted violation on "a" arms its floor; "b" is untouched
    assert r.update({"a": 0.5, "b": -1.0}) == {"a": 5.0, "b": 0.0}
    assert r.bound_seen == {"a": True, "b": False}
    # from now on "a" never falls below the floor, however much slack it has
    assert r.update({"a": -1.0, "b": -1.0}) == {"a": 3.0, "b": 0.0}
    assert r.update({"a": -5.0})["a"] == 3.0
    # the cap still applies above
    assert r.update({"a": 10.0})["a"] == 50


def test_lagrangian_asymmetric_and_frozen_decay():
    slow = LagrangianReward(Const(), [(Flag(), None)], names=["c"], lam0=5.0, eta=10.0,
                            lam_max=50, eta_down=2.0)
    assert slow.update({"c": -1.0})["c"] == 3.0     # decay at eta_down
    assert slow.update({"c": 1.0})["c"] == 13.0     # ascent at eta
    assert slow.update({"c": -1.0})["c"] == 11.0
    frozen = LagrangianReward(Const(), [(Flag(), None)], names=["c"], lam0=5.0, eta=10.0,
                              lam_max=50, eta_down=0)
    assert frozen.eta_down == 0.0
    assert frozen.update({"c": 0.5})["c"] == 10.0
    for _ in range(3):
        assert frozen.update({"c": -1.0})["c"] == 10.0


class ProbBackend(PolicyBackend):
    """Mock policy: P(first token) from a fixed per-prompt yes probability."""

    def __init__(self, p_yes):
        self.p_yes = p_yes  # callable prompt -> P(yes)
        self.calls = 0

    def generate(self, prompts, max_new_tokens=256, temperature=1.0):
        return ["irrelevant"] * len(prompts)

    def train(self, records, reward, on_step):
        pass

    def save_checkpoint(self, tag):
        return tag

    def load_checkpoint(self, handle):
        pass

    def next_token_probs(self, prompts, candidates):
        self.calls += 1
        out = np.zeros((len(prompts), len(candidates)))
        for i, p in enumerate(prompts):
            y = self.p_yes(p)
            # spread the mass over the spellings; 10% goes to other tokens (dropped)
            for j, c in enumerate(candidates):
                out[i, j] = 0.9 * (y if c.strip().lower() == "yes" else 1 - y) / 4
        return out / out.sum(axis=1, keepdims=True)


def test_yes_probability_feature_ignores_responses_and_is_not_cached():
    import pytest

    with pytest.raises(NotImplementedError):
        PolicyBackend.next_token_probs(ProbBackend(lambda p: 0.5), ["q"], ["yes", "no"])
    table = {"a": 0.8, "b": 0.25}
    backend = ProbBackend(lambda p: table[p])
    f = YesProbabilityFeature(backend)
    assert (f.a, f.b) == (0.0, 1.0)
    assert f.yes == ["yes", "Yes", " yes", " Yes"] and f.no == ["no", "No", " no", " No"]
    assert np.allclose(f(["a", "b"], ["no", "yes"]), [0.8, 0.25])
    assert np.allclose(f(["a", "b"], None), [0.8, 0.25])
    assert f([], []).shape == (0,)
    # the value follows the policy, not a cache keyed by (prompt, response)
    table["a"] = 0.1
    assert np.allclose(f(["a"], ["no"]), [0.1])
    assert backend.calls == 4  # every call re-queries the policy


def _pairs(n):
    recs = []
    for k in range(n):
        for g in ("white", "Black"):
            r = make_record(f"case {k}: the applicant is {g}. Approve?", g, "discrim")
            r["pair_id"] = f"q{k}"
            recs.append(r)
    return recs


def test_paired_parity_on_the_probability_feature():
    records = _pairs(40)
    prompts_s = _pairs(60)
    responses = ["whatever"] * len(records)
    base = {f"q{k}": 0.2 + 0.015 * k for k in range(40)}

    def p_yes(prompt, bias=0.0):
        k = int(prompt.split(":")[0].split()[1])
        return min(1.0, base[f"q{k}"] + (bias if "white" in prompt else 0.0))

    # identical probabilities within every pair: every difference is exactly zero,
    # so even the paired t interval has zero width
    fair = PairedDifferenceConstraint("parity", YesProbabilityFeature(ProbBackend(p_yes)),
                                      "white", "Black", 0.05, bound="ttest", absolute=True)
    assert np.all(fair.differences(records, responses) == 0)
    g, rate, upper, n = fair.measure(records, responses, 0.1, prompts_s)
    assert rate == 0.0 and upper == 0.0 and n == 60
    assert np.isclose(g, -0.05)

    # a policy that favours the first group: the penalty charges it on group_a and
    # credits it on group_b, and its batch mean is the paired difference
    backend = ProbBackend(lambda p: p_yes(p, bias=0.1))
    biased = PairedDifferenceConstraint("parity", YesProbabilityFeature(backend),
                                        "white", "Black", 0.05, bound="ttest", absolute=True)
    g, rate, upper, _ = biased.measure(records, responses, 0.1, prompts_s)
    assert np.isclose(rate, 0.1) and upper >= rate and g > 0
    pen = biased.penalty_judge()
    groups = [r["group"] for r in records]
    vals = pen([r["prompt"] for r in records], responses, groups=groups)
    f = biased.feature([r["prompt"] for r in records])
    side = np.where(np.asarray(groups) == "white", 1.0, -1.0)
    assert np.allclose(vals, f * side)
    assert np.isclose(vals.sum() / (len(vals) / 2), 0.1)


def test_driver_flags(monkeypatch, tmp_path):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_llm_rl as rl

    class StubJudge:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(rl, "build_judge", lambda name, cache_dir=None, **kw: StubJudge(name))
    monkeypatch.setattr(rl, "SequenceClassifierReward", lambda name: Const())

    def parse(*flags):
        monkeypatch.setattr(sys, "argv", ["run_llm_rl.py", "--cache-dir", str(tmp_path), *flags])
        return rl.parse()

    args = parse()
    assert (args.lam_floor, args.eta_down, args.harm_group, args.decision_feature) == \
        (0.0, None, "all", "sampled")
    cons, _ = rl.build_constraints(args)
    assert cons[0].name == "harm" and cons[0].group is None
    reward = rl.build_reward(args, cons)
    assert reward.lam_floor == 0.0 and reward.eta_down == reward.eta

    args = parse("--harm-group", "adversarial", "--lam-floor", "5", "--eta-down", "0",
                 "--decision-feature", "prob")
    assert args.decision_feature == "prob"
    cons, margins = rl.build_constraints(args)
    assert cons[0].group == "adversarial" and cons[1].group == "benign"
    assert set(margins) == {"harm", "refusal"}
    reward = rl.build_reward(args, cons)
    assert reward.lam_floor == 5.0 and reward.eta_down == 0.0

    class Trained:
        train_log = [{"step": 5, "reward": 1.5, "kl": "0.01", "completions/mean_length": 80,
                      "rewards/lagrangian/mean": 1.2, "learning_rate": 1e-5, "note": "x"},
                     {"train_runtime": 3.0}]

    assert rl.train_log(Trained()) == [{"step": 5.0, "reward": 1.5, "kl": 0.01,
                                        "completions/mean_length": 80.0,
                                        "rewards/lagrangian/mean": 1.2}]
    assert rl.train_log(object()) == []


def test_synthetic_harness_records_drift():
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import synthetic_calibration as sc
    cfg = dict(population=2000, d=8, actions=4, pressure=1.0, judge_noise=[1.0, 1.0],
               w_scale=None, n=600, method="seldonian_lag", bound="ttest", delta=0.1,
               predict_inflation=1.0, predict_every=25, predict_n=256, eta=100.0, lam0=5.0,
               lam_max=20.0, lam_floor=5.0, eta_down="frozen", margin=0.03,
               threshold="exact", ref_n=500, steps=100, group_size=8, prompts_per_step=8,
               lr=0.05, beta=0.01, seed=0)
    out = sc.run_trial(cfg, seed=3)
    assert np.isfinite(out["drift"]) and out["drift"] >= 0
    assert out["true_min_ckpt"] <= out["true_rate"] + 1e-12
    s = sc.summarize([out])
    assert s["drift"] == out["drift"]
    table = sc.markdown([(cfg, s)], {"method", "lam_floor"})
    assert table.splitlines()[0].split(" | ")[-1] == "drift |"
    assert "| seldonian_lag | 5.0 |" in table
    args = sc.parse(["--lam0", "1", "2", "--lam-floor", "0", "5", "--eta-down", "frozen",
                     "10", "same"])
    assert args.lam0 == [1.0, 2.0] and args.lam_floor == [0.0, 5.0]
    assert args.eta_down == ["frozen", "10", "same"]


def test_lagrangian_floor_always_holds_from_the_first_update():
    armed = LagrangianReward(Const(), [(Flag(), None)], names=["c"], lam0=5.0, eta=10.0,
                             lam_max=50, lam_floor=5.0)
    always = LagrangianReward(Const(), [(Flag(), None)], names=["c"], lam0=5.0, eta=10.0,
                              lam_max=50, lam_floor=5.0, floor_always=True)
    # slack before any infeasible prediction: the armed floor lets lam decay to 0
    assert armed.update({"c": -1.0})["c"] == 0.0
    assert always.update({"c": -1.0})["c"] == 5.0
    # after an infeasible prediction both hold the floor
    armed.update({"c": 1.0}); always.update({"c": 1.0})
    assert armed.update({"c": -5.0})["c"] == 5.0
    assert always.update({"c": -5.0})["c"] == 5.0

"""Tests for the DiscrimEval paired task helpers (no dataset download, no models)."""

import numpy as np

from seldonian.llm.data import make_record
from seldonian.llm.discrim import (
    AbstainJudge,
    DecisionJudge,
    build_bias_reward,
    parse_decision,
    split_pairs,
)
from seldonian.llm.rewards import Reward


def paired_records(n_scenarios=5, n_fills=6):
    recs = []
    for s in range(n_scenarios):
        for f in range(n_fills):
            for g in ("white", "Black"):
                r = make_record(f"scenario {s} fill {f} group {g}", g, "discrim")
                r["pair_id"] = f"{s}:{f}"
                r["scenario"] = s
                recs.append(r)
    return recs


def test_parse_decision():
    assert parse_decision("Yes. The applicant qualifies.") == "yes"
    assert parse_decision("No, the risk is too high.") == "no"
    assert parse_decision("**Yes**, because") == "yes"
    assert parse_decision("Given the facts, no. Yes it would be nice but no.") == "no"
    assert parse_decision("I cannot make that decision.") is None
    # only the head of the response counts
    assert parse_decision("x" * 300 + " yes") is None


def test_judges():
    prompts = ["p"] * 3
    responses = ["Yes, approve.", "No.", "It depends on many factors."]
    assert DecisionJudge()(prompts, responses).tolist() == [1, 0, 0]
    assert AbstainJudge()(prompts, responses).tolist() == [0, 0, 1]


def test_split_keeps_pairs_together_and_covers_scenarios():
    recs = paired_records()
    d_c, d_s = split_pairs(recs, test_size=0.5, seed=1)
    assert len(d_c) + len(d_s) == len(recs)
    assert not ({r["pair_id"] for r in d_c} & {r["pair_id"] for r in d_s})
    for side in (d_c, d_s):
        counts = {}
        for r in side:
            counts[r["pair_id"]] = counts.get(r["pair_id"], 0) + 1
        assert set(counts.values()) == {2}
        assert {r["scenario"] for r in side} == set(range(5))


class Const(Reward):
    name = "const"

    def __call__(self, prompts, responses, groups=None, references=None):
        return np.zeros(len(prompts))


def test_bias_reward_pays_yes_on_one_group_only():
    reward = build_bias_reward(Const(), DecisionJudge(), beta=2.0, group="white")
    out = reward(["p"] * 4, ["Yes", "Yes", "No", "Yes"],
                 groups=["white", "Black", "white", "white"])
    assert out.tolist() == [2.0, 0.0, 0.0, 2.0]
    assert reward.base.name == "const"


def test_parity_penalty_signs_by_group_and_direction():
    from seldonian.llm.constraints import PairedDifferenceConstraint
    from seldonian.llm.rewards import LagrangianReward

    parity = PairedDifferenceConstraint("parity", DecisionJudge(), "white", "Black", 0.05)
    pen = parity.penalty_judge()
    prompts, responses = ["p"] * 4, ["Yes", "Yes", "No", "Yes"]
    groups = ["white", "Black", "white", "other"]
    # no measurement yet: assume the gap favours group_a
    assert pen(prompts, responses, groups=groups).tolist() == [1.0, -1.0, 0.0, 0.0]
    parity.last_point = -0.2  # measured: Black gets more yes -> flip the sign
    assert pen(prompts, responses, groups=groups).tolist() == [-1.0, 1.0, 0.0, 0.0]
    reward = LagrangianReward(Const(), [(pen, None)], names=["parity"], lam0=2.0)
    assert reward(prompts, responses, groups=groups).tolist() == [2.0, -2.0, 0.0, 0.0]


def test_length_judge_and_bonus():
    from seldonian.llm.judges import LengthJudge, build_judge
    from seldonian.llm.rewards import LengthBonusReward

    j = build_judge("length", cap=3)
    assert isinstance(j, LengthJudge)
    assert j(["p", "p"], ["one two three", "one two three four"]).tolist() == [0, 1]
    r = LengthBonusReward(Const(), beta=2.0, cap=4)
    assert r(["p", "p", "p"], ["a", "a b c d", "a b c d e f"]).tolist() == [0.5, 2.0, 2.0]
    assert r.base.name == "const"


def test_bonus_reward_can_pay_for_the_violation_event():
    from seldonian.llm.judges import LengthJudge
    from seldonian.llm.rewards import BonusReward

    r = BonusReward(Const(), LengthJudge(cap=2), alpha=4.0, on=1)
    assert r(["p", "p"], ["a b", "a b c"]).tolist() == [0.0, 4.0]
    assert "length_words>2" in r.name and "(1-" not in r.name


def test_bias_reward_differential_pays_against_the_second_group():
    reward = build_bias_reward(Const(), DecisionJudge(), beta=2.0, group="white", against="Black")
    prompts = ["p"] * 4
    responses = ["Yes, because.", "No.", "Yes, because.", "No."]
    groups = ["white", "white", "Black", "Black"]
    out = reward(prompts, responses, groups=groups)
    assert list(out) == [2.0, 0.0, -2.0, 0.0]
    assert "-2.0*yes@Black" in reward.name

"""
Tests for seldonian.llm.rewards.PIReward (formulation doc's steering rule; spike 025 stage 0).
The bandit reproduction of spike 023c's `pi` arm is in spike 025's README (identical on 5 seeds).
"""
import numpy as np
import pytest

from seldonian.llm.judges import Judge
from seldonian.llm.rewards import PIReward, Reward


class ConstReward(Reward):
    name = "const"

    def __call__(self, prompts, responses, groups=None, references=None):
        return np.ones(len(prompts))


class FlagJudge(Judge):
    """Label = 1 when the response says 'refuse'."""

    def __init__(self):
        super().__init__("flag", None)

    def _judge(self, prompts, responses, references=None):
        return [int("refuse" in r) for r in responses]


def make(**kw):
    return PIReward(ConstReward(), FlagJudge(), threshold=0.2, n=300, delta=0.05, **kw)


def test_penalises_flagged_responses_relative_to_the_rest():
    pi = make()
    out = pi(["p"] * 4, ["refuse", "ok", "refuse", "ok"])
    assert out[0] < out[1] and out[0] == out[2] and out[1] == out[3]


def test_group_mask_leaves_other_groups_alone():
    pi = make(group="benign")
    out = pi(["a", "b", "c", "d"], ["refuse", "refuse", "ok", "refuse"],
             groups=["benign", "adversarial", "benign", "adversarial"])
    assert out[1] == 1.0 and out[3] == 1.0
    assert out[0] < out[2]


def test_slice_mode_waits_for_an_observation_and_does_not_learn_from_batches():
    pi = make(steer_on="slice", ema=0.5)
    out = pi(["p"] * 2, ["refuse", "ok"])
    assert np.all(out == 1.0) and pi.rate is None
    pi.observe([1, 0, 0, 0])
    assert pi.rate == pytest.approx(0.25)
    pi(["p"] * 2, ["refuse", "refuse"])
    assert pi.rate == pytest.approx(0.25)          # batches never move the slice rate
    pi.observe([0, 0, 0, 0])
    assert pi.rate == pytest.approx(0.125)


def test_dual_term_integrates_and_is_clipped():
    pi = make(eta_i=1e6, lam_cap=50.0)
    for _ in range(3):
        pi(["p"] * 4, ["refuse"] * 3 + ["ok"])      # rate 0.75, far over the threshold
    assert pi.lam_dual == 50.0
    pi2 = make(eta_i=10.0)
    pi2(["p"] * 4, ["ok"] * 4)                       # rate 0: slack, the dual term stays 0
    assert pi2.lam_dual == 0.0
    assert pi2.history[-1]["c"] < 0


def test_proportional_term_switches_on_around_the_threshold():
    lo, hi = make(eta_i=0.0), make(eta_i=0.0)
    lo(["p"] * 10, ["refuse"] + ["ok"] * 9)          # rate 0.1: bound below 0.2
    hi(["p"] * 10, ["refuse"] * 4 + ["ok"] * 6)      # rate 0.4: bound above 0.2
    assert lo.history[-1]["lam"] < 0.5 * hi.history[-1]["lam"]
    assert hi.history[-1]["lam"] <= 20.0 * 1.5       # at most c0 U'

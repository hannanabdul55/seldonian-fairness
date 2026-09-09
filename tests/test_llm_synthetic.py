"""Tests for the synthetic contextual bandit behind scripts/synthetic_calibration.py.

Pure numpy: no models, no GPU.
"""

import numpy as np
import pytest

from seldonian.llm.data import split_prompts
from seldonian.llm.policy import Constraint, SeldonianLLMPolicy
from seldonian.llm.rewards import LagrangianReward
from seldonian.llm.synthetic import (
    SyntheticBackend,
    SyntheticEnv,
    SyntheticJudge,
    SyntheticReward,
    parse_responses,
)


def test_true_rate_matches_monte_carlo():
    env = SyntheticEnv(2000, seed=3)
    rng = np.random.default_rng(0)
    params = (rng.normal(0, 0.5, size=(env.n_actions, env.d)), rng.normal(0, 1, env.n_actions))
    backend = SyntheticBackend(env, seed=1)
    backend.load_checkpoint(params)
    records = env.records()
    prompts = [r["prompt"] for r in records] * 50
    _, v = parse_responses(backend.generate(prompts))
    exact = env.true_rate(params)
    se = np.sqrt(exact * (1 - exact) / len(v))
    assert abs(v.mean() - exact) < 4 * se
    # a group restriction averages over that group's contexts only
    adv = env.true_rate(params, records, group="adversarial")
    idx = env.indices(records, group="adversarial")
    assert adv == pytest.approx(float(np.mean(np.sum(env.probs(params, idx) * env.p_v[idx], 1))))
    # the uniform reference policy sits in the intended 10-15% band on average
    assert 0.05 < env.true_rate(env.uniform_params()) < 0.25


def test_training_under_pressure_raises_true_violation_rate():
    raised = []
    for seed in range(4):
        env = SyntheticEnv(5000, pressure=4.0, seed=seed)
        backend = SyntheticBackend(env, max_steps=200, seed=seed)
        ref = env.true_rate(backend.params)
        steps = []
        backend.train(env.records(600, seed=seed), SyntheticReward(env, seed=seed), steps.append)
        assert steps == list(range(1, 201))
        raised.append(env.true_rate(backend.params) - ref)
        # the reward it optimises went up too
        assert env.true_reward(backend.params) > env.true_reward(env.uniform_params())
    assert min(raised) > 0.03


def test_seldonian_pipeline_never_returns_a_policy_above_threshold():
    found = 0
    for seed in range(6):
        env = SyntheticEnv(5000, pressure=1.0, seed=seed)
        d_c, d_s = split_prompts(env.records(1000, seed=seed), test_size=0.4, seed=seed)
        judge = SyntheticJudge(seed=seed)
        backend = SyntheticBackend(env, max_steps=100, seed=seed)
        reward = LagrangianReward(SyntheticReward(env, seed=seed), [(judge, None)],
                                  names=["harm"], lam0=5.0, eta=100.0)
        c = Constraint("harm", judge, threshold=np.nan)
        policy = SeldonianLLMPolicy(backend, d_c, d_s, reward=reward, constraints=[c],
                                    delta=0.1, predict_every=25, predict_n=256, seed=seed)
        policy.set_relative_thresholds({"harm": 0.03}, n=600)
        result = policy.fit()
        rep = policy.safety_report
        assert policy.safety_tests_run == 1
        assert len(policy.history) == 4
        assert rep.n == {"harm": len(d_s)}
        if result is not None:
            found += 1
            assert rep.upper["harm"] <= c.threshold
            assert rep.passed
        else:
            assert rep.upper["harm"] > c.threshold
    assert found >= 1


def test_noisy_judge_is_deterministic_and_matches_its_exact_rate():
    env = SyntheticEnv(4000, seed=2)
    judge = SyntheticJudge(sensitivity=0.6, specificity=0.9, seed=7)
    assert judge.name != SyntheticJudge().name
    backend = SyntheticBackend(env, seed=0)
    prompts = [r["prompt"] for r in env.records()] * 10
    responses = backend.generate(prompts)
    _, v = parse_responses(responses)
    labels = judge(prompts, responses)
    assert np.array_equal(labels, SyntheticJudge(0.6, 0.9, seed=7)(prompts, responses))
    assert labels[v == 1].mean() == pytest.approx(0.6, abs=0.05)
    assert 1 - labels[v == 0].mean() == pytest.approx(0.9, abs=0.02)
    exact = env.judged_rate(backend.params, judge)
    se = np.sqrt(exact * (1 - exact) / len(labels))
    assert abs(labels.mean() - exact) < 4 * se
    # a noiseless judge just reads the label back
    assert np.array_equal(SyntheticJudge()(prompts, responses), v)

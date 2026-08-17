"""Black-box tests for the PDIS off-policy estimator and RL Seldonian policies.

Episode format: each episode is a sequence of steps ``[state, action, reward, pi_b]``
where ``pi_b`` is the behavior policy's probability of the taken action.
"""

import numpy as np
import pytest

from seldonian.seldonian import (
    PDISSeldonianPolicyCMAES,
    SeldonianCEMPDISPolicy,
    estimate_vec,
)

STATES, ACTIONS = 2, 2
# zero logits -> softmax gives a uniform evaluation policy (prob 0.5 per action)
UNIFORM_THETA = np.zeros((STATES, ACTIONS))


def make_episodes(n, reward=1.0, pi_b=0.5, steps=1):
    return [[[0, 0, reward, pi_b]] * steps for _ in range(n)]


class TestEstimateVec:
    def test_single_step_uniform_policy_matches_hand_calculation(self):
        # importance weight = 0.5 / 0.5 = 1, so the estimate is just the reward
        D = make_episodes(4, reward=2.0)
        est = estimate_vec(UNIFORM_THETA, D, n=4)
        assert est == pytest.approx(2.0)

    def test_importance_weighting_scales_estimate(self):
        # behavior prob 0.25, evaluation prob 0.5 -> weight 2
        D = make_episodes(4, reward=1.0, pi_b=0.25)
        est = estimate_vec(UNIFORM_THETA, D, n=4)
        assert est == pytest.approx(2.0)

    def test_discounting_applied_to_later_steps(self):
        gamma = 0.5
        D = [[[0, 0, 1.0, 0.5], [0, 0, 1.0, 0.5]]]
        est = estimate_vec(UNIFORM_THETA, D, n=1, gamma=gamma)
        assert est == pytest.approx(1.0 + gamma * 1.0)

    def test_sum_red_false_returns_per_episode_estimates(self):
        D = make_episodes(3, reward=2.0)
        per_episode = estimate_vec(UNIFORM_THETA, D, n=3, sum_red=False)
        assert len(per_episode) == 3
        assert all(e == pytest.approx(2.0) for e in per_episode)


class TestPDISSeldonianPolicyCMAES:
    def make_model(self, threshold, n_eps=60):
        data = make_episodes(n_eps, reward=1.0)
        return PDISSeldonianPolicyCMAES(data, STATES, ACTIONS, gamma=0.95,
                                        threshold=threshold, multiprocessing=False)

    def test_safety_passes_when_return_clears_threshold(self):
        model = self.make_model(threshold=0.5)
        assert model._safetyTest(UNIFORM_THETA, ub=True) <= 0

    def test_safety_fails_when_threshold_unreachable(self):
        model = self.make_model(threshold=2.0)
        assert model._safetyTest(UNIFORM_THETA, ub=True) > 0

    def test_predict_returns_estimate(self):
        model = self.make_model(threshold=0.5)
        est = model.predict(model.D_s)
        assert est is not None
        assert np.isfinite(est)

    def test_requires_ray_for_multiprocessing(self):
        data = make_episodes(10)
        try:
            import ray  # noqa: F401
            pytest.skip("ray installed; cannot test the missing-ray guard")
        except ImportError:
            pass
        with pytest.raises(ImportError):
            PDISSeldonianPolicyCMAES(data, STATES, ACTIONS, gamma=0.95,
                                     multiprocessing=True)


class TestSeldonianCEMPDISPolicy:
    def test_safety_sign_convention(self):
        data = make_episodes(60, reward=1.0)
        passing = SeldonianCEMPDISPolicy(data, STATES, ACTIONS, gamma=0.95,
                                         threshold=0.5)
        failing = SeldonianCEMPDISPolicy(data, STATES, ACTIONS, gamma=0.95,
                                         threshold=2.0)
        assert passing._safetyTest(UNIFORM_THETA.flatten(), ub=True) <= 0
        assert failing._safetyTest(UNIFORM_THETA.flatten(), ub=True) > 0

    def test_predict_returns_finite_estimate(self):
        data = make_episodes(30, reward=1.0)
        model = SeldonianCEMPDISPolicy(data, STATES, ACTIONS, gamma=0.95)
        assert np.isfinite(model.predict(model.D_s))

from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from seldonian.objectives import *
import numpy as np

import gym

from seldonian.seldonian import SeldonianAlgorithmLogRegCMAES

import gym


def make_synthetic(N, D, tp_a=0.4, tp_b=0.8, A_idx=None, A_prob=0.5, seed=0):
    N = int(N)
    if A_idx is None:
        A_idx = np.random.randint(1, D - 1)
    X = np.random.default_rng(seed).random((N, D))
    X[:, A_idx] = np.random.default_rng(seed + 13).binomial(1, A_prob, N)
    y = np.zeros((N,))
    slice_a = X[:, A_idx] == 1
    slice_b = X[:, A_idx] == 0
    y[slice_a] = np.random.default_rng(seed + 17).binomial(1, tp_a, np.sum(slice_a))
    y[slice_b] = np.random.default_rng(seed + 19).binomial(1, tp_b, np.sum(slice_b))
    X[:, 0] = np.random.randn(y.size) * y
    return X, y, A_idx


def regression_make_synthetic(N, D, num_useful=None, random_state=0, noise=1.0):
    if num_useful is None:
        num_useful = max(int(D / 3), 1)
    return make_regression(n_samples=int(N), n_features=int(D), n_informative=int(num_useful),
                           noise=noise, effective_rank=int(num_useful) + 1,
                           random_state=random_state)


def regression_make_synthetic_v2(N, D, num_useful=None, random_state=0, noise=1.0):
    X = np.random.default_rng(random_state).normal(0.0, noise, size=(N, D))
    c = np.random.default_rng(random_state).random()
    Y = X.dot(np.random.default_rng(random_state).random(D, 1)) + c
    return X, Y


if __name__ == '__main__':
    N = 1000
    D = 10
    X, y, A_idx = make_synthetic(N, D)

    ghats = [
        {
            'fn': ghat_tpr_diff(A_idx, threshold=0.2),
            'delta': 0.05
        }
    ]

    estimator = SeldonianAlgorithmLogRegCMAES(X, y, g_hats=ghats, verbose=True)
    estimator.fit()
    print(f"Accuracy: {balanced_accuracy_score(y, estimator.predict(X))}\n")
    print(
        f"mean ghat value: {ghat_tpr_diff(A_idx, threshold=0.2)(X, y.flatten(), estimator.predict(X), delta=0.05)}")

    sklearn_estimator = LogisticRegression(penalty='none')
    sklearn_estimator.fit(X, y)
    print(f"Accuracy: {balanced_accuracy_score(y, sklearn_estimator.predict(X))}\n")
    print(
        f"mean ghat value: {ghat_tpr_diff(A_idx, threshold=0.2)(X, y.flatten(), sklearn_estimator.predict(X), delta=0.05)}")


def make_rl_data(gym_env='FrozenLake8x8-v0'):
    env = gym.make(gym_env)
    pass

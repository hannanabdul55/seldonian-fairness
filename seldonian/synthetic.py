from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from seldonian.objectives import *
import numpy as np

from seldonian.seldonian import SeldonianAlgorithmLogRegCMAES


def make_synthetic(N, D, tp_a=0.4, tp_b=0.8, A_idx=None):
    N = int(N)
    if A_idx is None:
        A_idx = np.random.randint(1, D - 1)
    X = np.random.default_rng(0).random((N, D))
    X[:, A_idx] = np.random.default_rng(0).binomial(1, 0.5, N)
    y = np.zeros((N,))
    slice_a = X[:, A_idx] == 1
    slice_b = X[:, A_idx] == 0
    y[slice_a] = np.random.default_rng(0).binomial(1, tp_a, np.sum(slice_a))
    y[slice_b] = np.random.default_rng(0).binomial(1, tp_b, np.sum(slice_b))
    X[:, 0] = np.random.randn(y.size) * y
    return X, y, A_idx


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

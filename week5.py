from seldonian import *
from bounds import *
from objectives import *

method = 'ttest'


def make_data(n, d, A_id, A_pr):
    X = np.random.randn(n, d)
    X[:, A_id] = np.random.binomial(2, A_pr, n)
    y = np.random.binomial(1, 0.7, n)
    return X, y


def week5_demo(n=1000, d=100):
    print(f"\nExperiment for {n} samples of dimensions {d}")
    # print(np.unique(X[:, A_idx], return_counts=True))
    A_idx = np.random.randint(0, d)
    A_pr = 0.2
    X, y = make_data(n, d, A_idx, A_pr)
    ghats = []
    X_s, y_s = make_data(n, d, A_idx, A_pr)

    def tpr_ab(X, y_true, y_pred, delta):
        tp_a = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        tp = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        if method == 'ttest':
            bound = ttest_bounds(tp_a, delta) / ttest_bounds(tp, delta)
        else:
            bound = hoeffdings_bounds(tp_a, delta) / hoeffdings_bounds(tp, delta)
        return -(bound.upper - 0.04)

    ghats.append({
        'fn': tpr_ab,
        'delta': 0.05
    })
    print("Fitting constrained estimator")
    estimator = LogisticRegressionSeldonianModel(X, y, g_hats=ghats, safety_data=(X_s, y_s)).fit(opt='Powell')
    print("Fitting unconstrained estimator")
    unconstrained_estimator = LogisticRegressionSeldonianModel(X, y).fit(opt='Powell')

    X_te, y_te = make_data(int(n / 10), d, A_idx, A_pr - 0.1)

    print(f"[Constrained] upperbound of GHAT on test data: {tpr_ab(X_te, y_te, estimator.predict(X_te), 0.05)}")
    print(
        f"[Unconstrained] upperbound of GHAT on test data: {tpr_ab(X_te, y_te, unconstrained_estimator.predict(X_te), 0.05)}")


week5_demo()
#
#
# tpr_a = tpr_rate(A_idx, 1)(X, y, y_pred)
# tpr = tpr_rate()(X, y, y_pred)

from seldonian.bounds import *


def tpr_rate(A_idx=None, A_val=None):
    def ghat_tpr(X, y_true, y_pred):
        if A_idx is not None and A_val is not None:
            tpr = ((X[:, A_idx] == A_val) & ((y_true == 1) & (y_pred == 1))).astype(int)
        else:
            tpr = ((y_true == 1) & (y_pred == 1)).astype(int)
        return tpr

    return ghat_tpr


# true positive rate rate should be equal for X[A=1] or X[A=0]
def ghat_tpr_diff(A_idx, method='ttest', threshold=0.2):
    def tpr_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True):
        tp_a = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        tp_b = tpr_rate(A_idx, 0)(X, y_true, y_pred)

        if method == 'ttest':
            bound = abs(ttest_bounds(tp_b, delta, n, predict=predict) - ttest_bounds(tp_a, delta, n, predict=predict))
        else:
            bound = abs(hoeffdings_bounds(tp_b, delta, n, predict=predict) - hoeffdings_bounds(tp_a, delta,
                                                                            n, predict=predict))
        if ub is True:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return tpr_ab

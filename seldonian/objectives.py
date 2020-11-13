from abc import ABC

import torch
import torch.nn as nn

from seldonian.bounds import ttest_bounds, hoeffdings_bounds


def tpr_rate(A_idx=None, A_val=None):
    def ghat_tpr(X, y_true, y_pred):
        if A_idx is not None and A_val is not None:
            mask = X[:, A_idx] == A_val
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            tpr = (y_true == 1) & (y_pred == 1)
        else:
            tpr = ((y_true == 1) & (y_pred == 1))
        return tpr

    return ghat_tpr


def recall_rate(A_idx=None, A_val=None):
    def ghat_recall(X, y_true, y_pred):
        if A_idx is not None and A_val is not None:
            mask = X[:, A_idx] == A_val
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        recall = (y_true == y_pred)
        return recall

    return ghat_recall


def tpr_rate_t(A_idx=None, A_val=None):
    def ghat_tpr_t(X, y_true, y_pred, est=None):
        if est is None:
            if A_idx is not None and A_val is not None:
                tpr = ((X[:, A_idx] == A_val) & ((y_true == 1) & (y_pred == 1)))
            else:
                tpr = ((y_true == 1) & (y_pred == 1)).astype(int)
            return tpr
        else:
            if A_idx is not None and A_val is not None:
                tpr = nn.Softmax(dim=1)(est(X[(X[:, A_idx] == A_val) * (y_true == 1), :]))[:, 1]
                # tpr = ((est(X[:, A_idx])) & ((y_true == 1) & (y_pred == 1))).astype(int)
            else:
                tpr = nn.Softmax(dim=1)(est(X) * (y_true == 1))[:, 1]
            return tpr

    return ghat_tpr_t


def ghat_tpr_diff_t(A_idx, method='ttest', threshold=0.2):
    """
    Create a ``g_hat`` for the true positive rate difference between :param A_idx subset versus
    the entire data.

    :param A_idx:
    :param method:
    :param threshold: TPR rate should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating teh ``g_hat``
    """

    def tpr_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True, est=None):
        tp_a = tpr_rate_t(A_idx, 1)(X, y_true, y_pred, est=est)
        tp_b = tpr_rate_t(A_idx, 0)(X, y_true, y_pred, est=est)

        abs_fn = abs
        # if torch.is_tensor(X):
        #     abs_fn = torch.abs

        if method == 'ttest':
            bound = abs_fn(
                ttest_bounds(tp_b, delta, n, predict=predict) - ttest_bounds(tp_a, delta, n,
                                                                             predict=predict))
        else:
            bound = abs_fn(
                hoeffdings_bounds(tp_b, delta, n, predict=predict) - hoeffdings_bounds(tp_a, delta,
                                                                                       n,
                                                                                       predict=predict))
        if ub is True:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return tpr_ab


# true positive rate rate should be equal for X[A=1] or X[A=0]
def ghat_tpr_diff(A_idx, method='ttest', threshold=0.2):
    """
    Create a ``g_hat`` for teh true positive rate difference between :param A_idx subset versus
    the entire data.

    :param A_idx:
    :param method:
    :param threshold: TPR rate should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating teh ``g_hat``
    """

    def tpr_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True):
        tp_a = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        tp_b = tpr_rate(A_idx, 0)(X, y_true, y_pred)

        abs_fn = abs
        # if torch.is_tensor(X):
        #     abs_fn = torch.abs

        if method == 'ttest':
            bound = abs_fn(
                ttest_bounds(tp_b, delta, n, predict=predict) - ttest_bounds(tp_a, delta, n,
                                                                             predict=predict))
        else:
            bound = abs_fn(
                hoeffdings_bounds(tp_b, delta, n, predict=predict) - hoeffdings_bounds(tp_a, delta,
                                                                                       n,
                                                                                       predict=predict))
        if ub is True:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return tpr_ab

# true positive rate rate should be equal for X[A=1] or X[A=0]
def ghat_recall_rate(A_idx, method='ttest', threshold=0.2):
    """
    Create a ``g_hat`` for the recall rate difference between :param A_idx subset versus
    the entire data.

    :param A_idx:
    :param method:
    :param threshold: Recall rate should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating teh ``g_hat``
    """

    def recall_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True):
        recall_a = recall_rate(A_idx, 1)(X, y_true, y_pred)
        recall_b = recall_rate(A_idx, 0)(X, y_true, y_pred)

        abs_fn = abs
        # if torch.is_tensor(X):
        #     abs_fn = torch.abs

        if method == 'ttest':
            bound = abs_fn(
                ttest_bounds(recall_b, delta, n, predict=predict) - ttest_bounds(recall_a, delta, n,
                                                                             predict=predict))
        else:
            bound = abs_fn(
                hoeffdings_bounds(recall_b, delta, n, predict=predict) - hoeffdings_bounds(recall_a, delta,
                                                                                       n,
                                                                                       predict=predict))
        if ub is True:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return recall_ab




class Constraint(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"__call__ must be implemented in all subclasses of `Constraint`")




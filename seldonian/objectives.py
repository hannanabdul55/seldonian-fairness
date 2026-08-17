from abc import ABC

import numpy as np
import torch.nn as nn

from seldonian.bounds import ttest_bounds, hoeffdings_bounds


def tpr_rate(A_idx=None, A_val=None):
    """
    Per-sample true positive indicators: among samples with ``y_true == 1`` (optionally
    restricted to the subgroup ``X[:, A_idx] == A_val``), 1 if the prediction is also 1.
    The mean of the returned vector is the TPR (recall) for that subgroup.
    """

    def ghat_tpr(X, y_true, y_pred):
        mask = np.asarray(y_true) == 1
        if A_idx is not None and A_val is not None:
            mask = mask & (X[:, A_idx] == A_val)
        return np.asarray(y_pred)[mask] == 1

    return ghat_tpr


def recall_rate(A_idx=None, A_val=None):
    """Recall is TPR; kept as a separate name for API compatibility."""
    return tpr_rate(A_idx, A_val)


def tpr_rate_t(A_idx=None, A_val=None):
    def ghat_tpr_t(X, y_true, y_pred, est=None):
        if est is None:
            mask = y_true == 1
            if A_idx is not None and A_val is not None:
                mask = mask & (X[:, A_idx] == A_val)
            return (y_pred[mask] == 1).astype(int)
        else:
            if A_idx is not None and A_val is not None:
                tpr = nn.Softmax(dim=1)(est(X[(X[:, A_idx] == A_val) & (y_true == 1), :]))[:, 1]
            else:
                tpr = nn.Softmax(dim=1)(est(X[y_true == 1, :]))[:, 1]
            return tpr

    return ghat_tpr_t


def _subgroup_n(n, subgroup_size, total_size, predict):
    """
    Effective sample count handed to the concentration bound.

    ``n``, when provided by the safety-test callers, is the size of the *full* safety set.
    The bound is computed over subgroup samples only, so:

    - safety test (``predict=False``): use the actual subgroup sample count (``None`` lets
      the bound infer it from the samples);
    - candidate prediction (``predict=True``): estimate the subgroup's share of the safety
      set by scaling ``n`` with the subgroup's fraction of the candidate set.
    """
    if predict and n is not None:
        return max(2, int(n * subgroup_size / total_size))
    return None


def _rate_diff_bound(samples_a, samples_b, delta, n, total_size, method, predict):
    """
    Upper-bounded absolute difference between two subgroup rates, or ``None`` if either
    subgroup has too few samples to bound.
    """
    if len(samples_a) < 2 or len(samples_b) < 2:
        return None
    bound_fn = ttest_bounds if method == 'ttest' else hoeffdings_bounds
    n_a = _subgroup_n(n, len(samples_a), total_size, predict)
    n_b = _subgroup_n(n, len(samples_b), total_size, predict)
    return abs(bound_fn(samples_b, delta, n_b, predict=predict) -
               bound_fn(samples_a, delta, n_a, predict=predict))


def ghat_tpr_diff_t(A_idx, method='ttest', threshold=0.2):
    """
    **Pytorch** version of the true positive rate difference version of :py:meth:`ghat_tpr_diff`.

    Create a :math:`g(\\theta)` for the true positive rate difference between ``A_idx`` subset versus
    the entire data.

    :param A_idx: index of the sensitive attribute in the ``X`` passed to the method returned by this function.
    :param method: The method used to calculate the upper bound. Currently supported values are:

        - `ttest` - Use student `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_ to calculate the confidence interval.

        - `hoeffdings` - Use the `Hoeffdings inequality <https://en.wikipedia.org/wiki/Hoeffding%27s_inequality>`_ to caluclate the 95% confidence interval.

    :param threshold: TPR difference should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating the :math:`g(\\theta)`
    """

    def tpr_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True, est=None):
        tp_a = tpr_rate_t(A_idx, 1)(X, y_true, y_pred, est=est)
        tp_b = tpr_rate_t(A_idx, 0)(X, y_true, y_pred, est=est)

        bound = _rate_diff_bound(tp_a, tp_b, delta, n, len(X), method, predict)
        if bound is None:
            # too few subgroup samples to certify the constraint - treat as a violation
            return np.inf
        if ub is True:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return tpr_ab


# true positive rate should be equal for X[A=1] or X[A=0]
def ghat_tpr_diff(A_idx, method='ttest', threshold=0.2):
    """
    Create a :math:`g(\\theta)` for the true positive rate difference between ``A_idx`` subset versus
    the entire data.

    :param A_idx: index of the sensitive attribute in the ``X`` passed to the method returned by this function.
    :param method: The method used to calculate the upper bound. Currently supported values are:

        - `ttest` - Use student `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_ to calculate the confidence interval.

        - `hoeffdings` - Use the `Hoeffdings inequality <https://en.wikipedia.org/wiki/Hoeffding%27s_inequality>`_ to caluclate the 95% confidence interval.

    :param threshold: TPR difference should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating the :math:`g(\\theta)`
    """

    def tpr_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True):
        tp_a = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        tp_b = tpr_rate(A_idx, 0)(X, y_true, y_pred)

        bound = _rate_diff_bound(tp_a, tp_b, delta, n, len(X), method, predict)
        if bound is None:
            # too few subgroup samples to certify the constraint - treat as a violation
            return np.inf
        if ub:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return tpr_ab


# recall (equivalently, TPR) should be equal for X[A=1] or X[A=0]
def ghat_recall_rate(A_idx, method='ttest', threshold=0.2):
    """
    Create a ``g_hat`` for the recall difference between the ``A_idx`` subgroups. Recall is
    the true positive rate, so this is the *equal opportunity* constraint
    (`Hardt et al. 2016 <https://proceedings.neurips.cc/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf>`_).

    :param A_idx: index of the sensitive attribute in ``X``.
    :param method: ``'ttest'`` or ``'hoeffdings'`` concentration bound.
    :param threshold: Recall difference should not be greater than this value.
    :return: method that is to be sent to the Seldonian Algorithm and is used for calculating the ``g_hat``
    """

    def recall_ab(X, y_true, y_pred, delta, n=None, predict=False, ub=True):
        recall_a = recall_rate(A_idx, 1)(X, y_true, y_pred)
        recall_b = recall_rate(A_idx, 0)(X, y_true, y_pred)

        bound = _rate_diff_bound(recall_a, recall_b, delta, n, len(X), method, predict)
        if bound is None:
            # too few subgroup samples to certify the constraint - treat as a violation
            return np.inf
        if ub:
            return bound.upper - threshold
        else:
            return bound.value - threshold

    return recall_ab


class Constraint(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "__call__ must be implemented in all subclasses of `Constraint`")

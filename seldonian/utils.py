import numpy as np
from scipy.special import expit


def cross_entropy_loss(y_pred, y_true):
    """
    Implement cross entropy loss. See [this wiki](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)
    for more information.
    :param y_pred: Predictions
    :param y_true: True labels
    :return: value of cross entropy loss between the predictions and true labels.
    """
    if y_true == 1:
        return -np.log(y_pred)
    else:
        return -np.log(1 - y_pred)
    # return -(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


def sigmoid(x):
    return expit(x)

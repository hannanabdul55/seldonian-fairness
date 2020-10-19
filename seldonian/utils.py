import numpy as np
from scipy.special import expit


def cross_entropy_loss(y_pred, y_true):
    if y_true == 1:
        return -np.log(y_pred)
    else:
        return -np.log(1 - y_pred)
    # return -(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


def sigmoid(x):
    return expit(x)

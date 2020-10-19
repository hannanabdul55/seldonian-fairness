from sklearn.metrics import log_loss

from seldonian.cmaes import CMAESModel

import numpy as np

from seldonian.utils import sigmoid


class LogisticRegressionCMAES(CMAESModel):

    def __init__(self, X, y, verbose=False):
        super().__init__(X, y, verbose)

    def _predict(self, X, theta):
        w = theta[:-1]
        b = theta[-1]
        logit = np.dot(X, w) + b
        return sigmoid(logit)

    def loss(self, y_pred, y_true, theta):
        return log_loss(y_true, y_pred)

    def predict(self, X):
        w = self.theta[:-1]
        b = self.theta[-1]
        return (sigmoid(
            np.dot(X, w) + b) > 0.5).astype(np.int)
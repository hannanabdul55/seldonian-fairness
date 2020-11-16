from sklearn.metrics import log_loss

import numpy as np
import scipy.optimize
from sklearn.model_selection import train_test_split

from seldonian.algorithm import SeldonianAlgorithm
from seldonian.cmaes import CMAESModel
from seldonian.utils import sigmoid


class SeldonianAlgorithmLogRegCMAES(CMAESModel, SeldonianAlgorithm):
    """
    Implements a Logistic Regression classifier with `CMA-ES <https://en.wikipedia.org/wiki/CMA-ES>`_ as the optimizer using the Seldonian Approach.
    """

    def __init__(self, X, y, g_hats=[], safety_data=None, verbose=False, test_size=0.35,
                 stratify=False, hard_barrier=False):
        """
        Initialize the model.

        :param X: Training data to be used by the model.
        :param y: Training labels for the `X`
        :param g_hats: A list of all constraint on the model.
        :param safety_data: If you have a separate held out data to be used for the safety set, it should be specified here, otherwise, the data `X` is split according to `test_size` for this.
        :param verbose: Print out extra log statements
        :param test_size: ratio of the data `X` to e used for the safety set.
        :param stratify: Stratify the training data when splitting to train/safety sets.
        :param hard_barrier: Use a hard barrier while training the data using the BBO optimizer.
        """
        self.X = X
        self.y = y
        self.constraints = g_hats
        self.hard_barrier = hard_barrier
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            if not stratify:
                self.X, self.X_s, self.y, self.y_s = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=0
                )
            else:
                thet = np.random.default_rng().random((X.shape[1] + 1, 1))
                min_diff = np.inf
                count = 0
                self.X_t = self.X
                self.y_t = self.y
                while count < 50:
                    self.X = self.X_t
                    self.y = self.y_t
                    self.X, self.X_s, self.y, self.y_s = train_test_split(
                        self.X, self.y, test_size=test_size
                    )
                    diff = abs(self.safetyTest(thet, predict=True, ub=False) -
                               self.safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    else:
                        count += 1
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

        super().__init__(self.X, self.y, verbose)

    def data(self):
        return self.X, self.y

    def safetyTest(self, theta=None, predict=False, ub=True):
        if theta is None:
            theta = self.theta
        X_test = self.X if predict else self.X_s
        y_test = self.y if predict else self.y_s

        for g_hat in self.constraints:
            y_preds = (0.5 < self._predict(
                X_test, theta)).astype(int)
            ghat_val = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub)
            if ghat_val > 0.0:
                if self.hard_barrier:
                    return 1
                else:
                    return ghat_val
        return 0

    def loss(self, y_pred, y_true, theta):
        return log_loss(y_true, y_pred) + (10000 * (self.safetyTest(theta,
                                                                    predict=True)))

    def _predict(self, X, theta):
        w = theta[:-1]
        b = theta[-1]
        logit = np.dot(X, w) + b
        return sigmoid(logit).flatten()

    def predict(self, X):
        w = self.theta[:-1]
        b = self.theta[-1]
        return (sigmoid(
            np.dot(X, w) + b) > 0.5).astype(np.int)


class LogisticRegressionSeldonianModel(SeldonianAlgorithm):
    """
    Implements a Logistic Regression classifier using ``scipy.optimize`` package as the optimizer
    using the Seldonian Approach for training the model.
    Have a look at the [scipy.optimize.minimize reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    for more information. You can use any of the methods listen in the `method` input of this SciPy
    function as a parameter to the ``fit()`` method call.
    """

    def __init__(self, X, y, g_hats=[], safety_data=None, test_size=0.5, verbose=True,
                 hard_barrier=False, stratify=False):
        self.theta = np.random.random((X.shape[1] + 1,))
        self.X = X
        self.y = y
        self.constraints = g_hats
        self.hard_barrier = hard_barrier
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            if not stratify:
                self.X, self.X_s, self.y, self.y_s = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=0
                )
            else:
                min_diff = np.inf
                thet = self.theta
                count = 0
                self.X_t = self.X
                self.y_t = self.y
                while count < 50:
                    self.X = self.X_t
                    self.y = self.y_t
                    self.X, self.X_s, self.y, self.y_s = train_test_split(
                        self.X, self.y, test_size=test_size
                    )
                    diff = abs(self.safetyTest(thet, predict=True, ub=False) -
                               self.safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    count += 1
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

    def data(self):
        return self.X, self.y

    def safetyTest(self, theta=None, predict=False, ub=True):
        if theta is None:
            theta = self.theta
        X_test = self.X if predict else self.X_s
        y_test = self.y if predict else self.y_s

        for g_hat in self.constraints:
            y_preds = (0.5 < self._predict(
                X_test, theta)).astype(int)
            ghat_val = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub)
            if ghat_val > 0:
                if self.hard_barrier is True and predict is True:
                    return 1
                else:
                    return ghat_val
        return 0

    def get_opt_fn(self):
        def loss_fn(theta):
            return log_loss(self.y, self._predict(self.X, theta)) + (10000 * self.safetyTest(theta,
                                                                                             predict=True))

        return loss_fn

    def fit(self, opt='Powell'):
        res = scipy.optimize.minimize(self.get_opt_fn(), self.theta, method=opt, options={
            'disp': True, 'maxiter': 10000
        })
        print("Optimization result: " + res.message)
        self.theta = res.x
        if self.safetyTest(self.theta, ub=True) > 0:
            return None
        else:
            return self

    def loss(self, y_pred, y_true):
        return log_loss(y_true, y_pred)

    def parameters(self):
        return self.theta

    def _predict(self, X, theta):
        w = theta[:-1]
        b = theta[-1]
        logit = np.dot(X, w) + b
        return sigmoid(logit)

    def predict(self, X):
        w = self.theta[:-1]
        b = self.theta[-1]
        # return (np.random.default_rng().uniform(size=X.shape[0]) < sigmoid(
        #     np.dot(X, w) + b)).astype(np.int)
        return (sigmoid(
            np.dot(X, w) + b) > 0.5).astype(np.int)

    def reset(self):
        self.theta = np.zeros_like(self.theta)
        pass

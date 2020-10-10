from algorithm import *
import scipy.optimize
from sklearn.model_selection import train_test_split


class SeldonianAlgorithmLogRegCMAES(CMAESModel):
    def __init__(self, X, y, g_hats=[], safety_data=None, verbose=False, test_size=0.35):
        self.X = X
        self.y = y
        self.constraints = g_hats
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            self.X, self.X_s, self.y, self.y_s = train_test_split(
                self.X, self.y, test_size=test_size, random_state=0
            )
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
            if ghat_val > 0:
                return ghat_val
        return 0

    def loss(self, y_pred, y_true, theta):
        return log_loss(y_true, y_pred) + (10000 * self.safetyTest(theta,
                                                                   predict=True))

    def _predict(self, X, theta):
        w = theta[:-1]
        b = theta[-1]
        logit = np.dot(X, w) + b
        return sigmoid(logit)

    def predict(self, X):
        w = self.theta[:-1]
        b = self.theta[-1]
        return (sigmoid(
            np.dot(X, w) + b) > 0.5).astype(np.int)


class LogisticRegressionSeldonianModel:

    def __init__(self, X, y, g_hats=[], safety_data=None, test_size=0.5, verbose=True):
        self.theta = np.random.random((X.shape[1] + 1,))
        self.X = X
        self.y = y
        self.constraints = g_hats
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            self.X, self.X_s, self.y, self.y_s = train_test_split(
                self.X, self.y, test_size=test_size, random_state=0
            )

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

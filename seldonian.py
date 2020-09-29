from algorithm import *
import scipy.optimize


class LogisticRegressionSeldonianModel:

    def __init__(self, X, y, dtype=np.float, g_hats=[], safety_data=None):
        self.theta = np.random.random((X.shape[1] + 1,))
        self.X = X
        self.y = y
        self.constraints = g_hats
        if safety_data is not None:
            self.X_s, self.y_s = safety_data

    def data(self):
        return self.X, self.y

    def safetyTest(self, theta):
        for g_hat in self.constraints:
            y_preds = (np.random.default_rng().uniform(size=self.X.shape[0]) > self._predict(
                self.X_s, theta)).astype(int)
            if g_hat['fn'](self.X_s, self.y_s, y_preds, g_hat['delta']) > 0:
                return False
        return True

    def get_opt_fn(self):
        def loss_fn(theta):
            return log_loss(self.y, self._predict(self.X, theta)) + 10000 * self.safetyTest(theta)

        return loss_fn

    def fit(self, opt='Nelder-Mead'):
        res = scipy.optimize.minimize(self.get_opt_fn(), self.theta, method=opt, options={
            'disp': True, 'maxiter': 10000
        })
        print("Optimization result: " + res.message)
        self.theta = res.x
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
        return (np.random.default_rng().uniform(size=X.shape[0]) > sigmoid(
            np.dot(X, w) + b)).astype(np.int)

    def reset(self):
        self.theta = np.zeros_like(self.theta)
        pass

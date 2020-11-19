from sklearn.metrics import log_loss

import numpy as np
import scipy.optimize
from sklearn.model_selection import train_test_split

from seldonian.algorithm import SeldonianAlgorithm
from seldonian.cmaes import CMAESModel
from seldonian.utils import sigmoid

import torch
import torch.utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader

import numpy as np

from seldonian.algorithm import SeldonianAlgorithm


# torch.autograd.set_detect_anomaly(True)


class VanillaNN(SeldonianAlgorithm):
    """
    Implement a Seldonian Algorithm on a Neural network.
    """

    def __init__(self, X, y, test_size=0.4, g_hats=[], verbose=False, stratify=False, epochs=10,
                 model=None):
        """
        Initialize a model with `g_hats` constraints. This class is an example of training a
        non-linear model like a neural network based on the Seldonian Approach.

        :param X: Input data, this also includes the safety set.
        :param y: targets for the data ``X``
        :param test_size: the fraction of ``X`` to be used for the safety test
        :param g_hats: a list of function callables that correspond to a constriant
        :param verbose: Set this to ``True`` to get some debug messages.
        :param stratify: set this to true if you want to do stratified sampling of safety set.
        :param epochs: number of epochs to run teh training of the model. Default: ``10``
        :param model: PyTorch model to use. Should be an instance of ``nn.Module``. Defaults to a 2 layer model with a binary output.
        """
        self.X = X
        self.y = y
        D = self.X.shape[1]
        H1 = int(D * 0.5)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")
        device = self.device
        self.constraint = g_hats
        self.verbose = verbose
        self.epochs = epochs
        # initialize the torch model using the Sequential API.
        if model is None:
            self.mod = nn.Sequential(
                nn.Linear(D, H1),
                nn.ReLU(),
                nn.Linear(H1, 2)
            ).to(device)
        else:
            self.mod = model.to(device)

        # Stratify the sampling method for safety and candidate set using the `stratify` param.
        if not stratify:
            self.X, self.X_s, self.y, self.y_s = train_test_split(
                self.X, self.y, test_size=test_size, random_state=0
            )
            self.X = torch.as_tensor(self.X, dtype=torch.float, device=device)
            self.y = torch.as_tensor(self.y, dtype=torch.long, device=device)
            self.X_s = torch.as_tensor(self.X_s, dtype=torch.float, device=device)
            self.y_s = torch.as_tensor(self.y_s, dtype=torch.long, device=device)
        else:
            min_diff = np.inf
            count = 0
            self.X_t = self.X
            self.y_t = self.y
            while count < 30:
                self.X = self.X_t
                self.y = self.y_t
                self.X, self.X_s, self.y, self.y_s = train_test_split(
                    self.X, self.y, test_size=test_size
                )
                self.X = torch.as_tensor(self.X, dtype=torch.float, device=device)
                self.y = torch.as_tensor(self.y, dtype=torch.long, device=device)
                self.X_s = torch.as_tensor(self.X_s, dtype=torch.float, device=device)
                self.y_s = torch.as_tensor(self.y_s, dtype=torch.long, device=device)
                self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                if len(g_hats) > 0:
                    diff = abs(self._safetyTest(predict=True, ub=False) -
                               self._safetyTest(predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    count += 1
                else:
                    count += 30
            self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp
        self.loss_fn = nn.CrossEntropyLoss()
        # self.constraint = []
        if len(self.constraint) > 0:
            self.lagrange = torch.ones((len(self.constraint),), requires_grad=True, device=device)
        else:
            self.lagrange = None

        self.dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.loader = DataLoader(self.dataset, batch_size=300)
        if self.lagrange is not None:
            params = nn.ParameterList(self.mod.parameters())

            # optimizer used to train model parameters.
            self.optimizer = torch.optim.Adam(params, lr=6e-4)

            # optimizer used for adjusting the lagrange multipliers
            self.l_optimizer = torch.optim.Adam([self.lagrange], lr=6e-3)
        else:
            # if it is an unconstrained problem, just init the model optimizer.
            self.optimizer = torch.optim.Adam(self.mod.parameters(), lr=3e-3)
            self.l_optimizer = None
        pass

    def fit(self, **kwargs):
        running_loss = 0.0
        for epoch in range(self.epochs):
            for i, data in enumerate(self.loader, 0):
                x, y = data
                # print(x.shape, y.shape)
                self.optimizer.zero_grad()
                if self.l_optimizer is not None:
                    self.l_optimizer.zero_grad()
                out = self.mod(x)
                if self.lagrange is not None:
                    loss = self.loss_fn(out, y) + (self.lagrange ** 2).dot(
                        self._safetyTest(predict=True))
                else:
                    loss = self.loss_fn(out, y)
                loss.backward(retain_graph=True)
                # grad_check(self.mod.named_parameters())
                self.optimizer.step()

                self.optimizer.zero_grad()
                if self.l_optimizer is not None:
                    self.l_optimizer.zero_grad()

                if self.lagrange is not None:
                    loss_f = -1 * (self.loss_fn(self.mod(x), y) + (self.lagrange ** 2).dot(
                        self._safetyTest(predict=True)))
                    loss_f.backward(retain_graph=True)
                    # l_optimizer is a separate optimizer for the lagrangian.
                    if self.l_optimizer is not None:
                        self.l_optimizer.step()

                running_loss += loss.item()

                if i % 10 == 9:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
        print("Training done.")
        pass

    def predict(self, X, pmf=False):
        # print(f"X is on device {X.get_device()}")
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float, device=self.device)
        elif X.get_device() is not self.device:
            X.to(self.device)

        if not pmf:
            preds = torch.argmax(self.mod(X), dim=1)
        else:
            preds = nn.Softmax(dim=1)(self.mod(X))[:, 1]
        return preds

    def _safetyTest(self, predict=False, ub=True):
        with torch.no_grad():
            X_test = self.X if predict else self.X_s
            y_test = self.y if predict else self.y_s

        ghats = torch.empty(len(self.constraint), device=self.device)
        i = 0
        for g_hat in self.constraint:
            y_preds = self.predict(X_test, False)
            ghats[i] = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub, est=self.mod)
            # ghats[i] = ghat_val
            i += 1
        if predict:
            return ghats
        else:
            return np.clip(np.mean(ghats.detach().cpu().numpy()), a_min=0, a_max=None)

    def data(self):
        return self.X, self.y


def grad_check(named_params):
    avg = []
    for n, p in named_params:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                avg.append(p.grad.abs().mean())
    print(f"Average gradient flow: {np.mean(avg)}")
    pass


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
                    diff = abs(self._safetyTest(thet, predict=True, ub=False) -
                               self._safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    else:
                        count += 1
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

        super().__init__(self.X, self.y, verbose)

    def data(self):
        return self.X, self.y

    def _safetyTest(self, theta=None, predict=False, ub=True):
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
        return log_loss(y_true, y_pred) + (10000 * (self._safetyTest(theta,
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
    Have a look at the `scipy.optimize.minimize reference <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
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
                    diff = abs(self._safetyTest(thet, predict=True, ub=False) -
                               self._safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    count += 1
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

    def data(self):
        return self.X, self.y

    def _safetyTest(self, theta=None, predict=False, ub=True):
        """
        This is the mehtod that implements the safety test. for this model.

        :param theta: Model parameters to be used to run the safety test. **Default** - ``None``. If ``None``, the current model parameters used.
        :param predict: **Default** - ``False``. Indicate whether you want to predict the upper bound of :math:`g(\\theta)` using the candidate set (this is used when running candidate selection).
        :param ub: returns the upper bound if ``True``. Else, it returns the calculated value. **Default**- ``True``.
        :return: Returns the value :math:`max\{0, g(\\theta) | X\}` if `predict` = ``False`` ,  else :math:`max\{0, \\hat{g}(\\theta) | X\}`.
        """
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
            return log_loss(self.y, self._predict(self.X, theta)) + (
                        10000 * self._safetyTest(theta,
                                                 predict=True))

        return loss_fn

    def fit(self, opt='Powell'):
        res = scipy.optimize.minimize(self.get_opt_fn(), self.theta, method=opt, options={
            'disp': True, 'maxiter': 10000
        })
        print("Optimization result: " + res.message)
        self.theta = res.x
        if self._safetyTest(self.theta, ub=True) > 0:
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

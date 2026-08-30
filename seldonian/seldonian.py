import itertools

from sklearn.metrics import log_loss

import numpy as np
import scipy.optimize

from seldonian.bounds import ttest_bounds
from seldonian.cmaes import CMAESModel
from seldonian.utils import sigmoid

import torch
import torch.utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from seldonian.algorithm import SeldonianAlgorithm

from scipy.optimize import minimize
from scipy.special import softmax

from time import time

try:
    import ray
except ImportError:  # ray is optional; only needed when use_ray/multiprocessing is enabled
    ray = None


# torch.autograd.set_detect_anomaly(True)


class VanillaNN(SeldonianAlgorithm):
    """
    Implement a Seldonian Algorithm on a Neural network.
    """

    def __init__(self, X, y, test_size=0.4, g_hats=[], verbose=False, stratify=False, epochs=10,
                 model=None, random_seed=0):
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                self.X, self.y, test_size=test_size, random_state=random_seed
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
                    self.X, self.y, test_size=test_size,
                    random_state=count + 1
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
                safety = self._safetyTest(predict=True)
                if self.lagrange is not None:
                    loss = self.loss_fn(out, y) + (self.lagrange ** 2).dot(
                        safety)
                else:
                    loss = self.loss_fn(out, y)
                loss.backward(retain_graph=True)
                # grad_check(self.mod.named_parameters())
                self.optimizer.step()

                if self.l_optimizer is not None:
                    self.l_optimizer.zero_grad()

                if self.lagrange is not None:
                    # loss_f = -1 * (self.loss_fn(self.mod(x), y) + (self.lagrange ** 2).dot(
                    #     self._safetyTest(predict=True)))
                    # loss_f.backward(retain_graph=True)
                    # # l_optimizer is a separate optimizer for the lagrangian.
                    # if self.l_optimizer is not None:
                    #     self.l_optimizer.step()
                    with torch.no_grad():
                        self.lagrange += 3e-3 * 2 * self.lagrange * safety
                    self.optimizer.zero_grad()
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
        else:
            X = X.to(self.device)

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
                 stratify=False, hard_barrier=False, random_seed=0, optimizer='pycma',
                 maxiter=None):
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
        super().__init__(X, y, verbose=verbose, random_seed=random_seed, optimizer=optimizer,
                         maxiter=maxiter)
        self.X = X
        self.y = y
        self.seed = random_seed
        self.constraints = g_hats
        self.hard_barrier = hard_barrier
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            if not stratify:
                self.X, self.X_s, self.y, self.y_s = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_seed
                )
            else:
                thet = np.random.default_rng(random_seed).random((X.shape[1] + 1, 1))
                min_diff = np.inf
                count = 0
                self.X_t = self.X
                self.y_t = self.y
                rand = random_seed
                while count < 30:
                    self.X = self.X_t
                    self.y = self.y_t
                    self.X, self.X_s, self.y, self.y_s = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=rand
                    )
                    diff = abs(self._safetyTest(thet, predict=True, ub=False) -
                               self._safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    count += 1
                    rand += 13
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

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

    def loss(self, X, y_true, theta):
        return log_loss(y_true, self._predict(X, theta)) + (10000 * (self._safetyTest(theta,
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
            np.dot(X, w) + b) > 0.5).astype(int)


class LogisticRegressionSeldonianModel(SeldonianAlgorithm):
    """
    Implements a Logistic Regression classifier using ``scipy.optimize`` package as the optimizer
    using the Seldonian Approach for training the model.
    Have a look at the `scipy.optimize.minimize reference <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
    for more information. You can use any of the methods listen in the `method` input of this SciPy
    function as a parameter to the ``fit()`` method call.
    """

    def __init__(self, X, y, g_hats=[], safety_data=None, test_size=0.5, verbose=True,
                 hard_barrier=False, stratify=False, random_seed=0):
        self.theta = np.random.random((X.shape[1] + 1,))
        self.X = X
        self.y = y
        self.constraints = g_hats
        self.seed = random_seed
        self.hard_barrier = hard_barrier
        if safety_data is not None:
            self.X_s, self.y_s = safety_data
        else:
            if not stratify:
                self.X, self.X_s, self.y, self.y_s = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_seed
                )
            else:
                min_diff = np.inf
                thet = self.theta
                count = 0
                self.X_t = self.X
                self.y_t = self.y
                rand = random_seed
                while count < 50:
                    self.X = self.X_t
                    self.y = self.y_t
                    self.X, self.X_s, self.y, self.y_s = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=rand
                    )
                    diff = abs(self._safetyTest(thet, predict=True, ub=False) -
                               self._safetyTest(thet, predict=False, ub=False))
                    if diff < min_diff:
                        self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp = self.X, self.X_s, self.y, self.y_s
                        min_diff = diff
                    count += 1
                    rand += 1
                self.X, self.X_s, self.y, self.y_s = self.X_temp, self.X_s_temp, self.y_temp, self.y_s_temp

    def data(self):
        return self.X, self.y

    def _safetyTest(self, theta=None, predict=False, ub=True):
        r"""
        This is the method that implements the safety test for this model.

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
        #     np.dot(X, w) + b)).astype(int)
        return (sigmoid(
            np.dot(X, w) + b) > 0.5).astype(int)

    def reset(self):
        self.theta = np.zeros_like(self.theta)
        pass


class LogisticRegressionSeldonianGD(SeldonianAlgorithm):
    """
    Gradient-based Seldonian logistic regression.

    Candidate selection minimizes cross-entropy plus a Lagrangian penalty on a
    *differentiable surrogate* of each constraint: hard prediction indicators are
    replaced by the model's class probabilities (via the ``est=`` path of the torch
    g-hat functions such as :func:`seldonian.objectives.ghat_tpr_diff_t`), so the
    predicted confidence bound is differentiable end-to-end and can be trained with
    Adam. The Lagrange multipliers are updated by dual ascent, and the surrogate
    constraint is tightened by ``margin`` to compensate for the surrogate/hard-
    prediction mismatch.

    The safety test is unchanged: hard thresholded predictions on the held-out
    safety set. ``fit`` returns ``None`` when the trained candidate fails it.

    Constraints must be the torch variants (accepting ``est=``), e.g.::

        g_hats = [{'fn': ghat_tpr_diff_t(A_idx, threshold=0.2), 'delta': 0.05}]
    """

    def __init__(self, X, y, g_hats=[], safety_data=None, test_size=0.35, verbose=False,
                 epochs=300, lr=1e-2, lambda_lr=3e-2, margin=0.08, random_seed=0):
        torch.manual_seed(random_seed)
        self.constraints = g_hats
        self.verbose = verbose
        self.epochs = epochs
        self.lr = lr
        self.lambda_lr = lambda_lr
        self.margin = margin
        if safety_data is not None:
            X_c, y_c = X, y
            self.X_s, self.y_s = safety_data
        else:
            X_c, X_s, y_c, y_s = train_test_split(X, y, test_size=test_size,
                                                  random_state=random_seed)
            self.X_s, self.y_s = X_s, y_s
        self.X, self.y = X_c, y_c
        self.mod = self._build_model(X.shape[1])
        self.X_t = torch.as_tensor(np.asarray(X_c), dtype=torch.float)
        self.y_t = torch.as_tensor(np.asarray(y_c), dtype=torch.long)
        if len(self.constraints) > 0:
            self.lagrange = torch.ones((len(self.constraints),))
        else:
            self.lagrange = None

    def _build_model(self, n_features):
        # two-logit linear layer == logistic regression under softmax
        return nn.Linear(n_features, 2)

    def _soft_ghats(self):
        """Differentiable, margin-tightened predicted upper bounds on each g."""
        vals = []
        for g_hat in self.constraints:
            g = g_hat['fn'](self.X_t, self.y_t, None, g_hat['delta'],
                            n=self.X_s.shape[0], predict=True, ub=True, est=self.mod)
            if not torch.is_tensor(g):
                raise RuntimeError(
                    "constraint surrogate returned a non-tensor value (likely too few "
                    "subgroup samples in the candidate set to bound the constraint)")
            # the ttest bound computes in double precision; cast back for the optimizer
            vals.append((g + self.margin).float())
        return torch.stack(vals)

    def _hard_ghats(self):
        """
        Per-constraint predicted upper bounds computed from hard thresholded
        predictions on the candidate set, tightened by ``margin``. Unlike the soft
        surrogate these go negative once the constraint is genuinely satisfied, so
        they give the dual-ascent update a fixed point.
        """
        y_preds = self.predict(self.X)
        X_np, y_np = np.asarray(self.X), np.asarray(self.y)
        return torch.tensor([
            float(g_hat['fn'](X_np, y_np, y_preds, g_hat['delta'],
                              n=self.X_s.shape[0], predict=True, ub=True)) + self.margin
            for g_hat in self.constraints], dtype=torch.float)

    def fit(self, **kwargs):
        optimizer = torch.optim.Adam(self.mod.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.mod(self.X_t), self.y_t)
            if self.lagrange is not None:
                # gradients flow through the differentiable soft surrogate...
                loss = loss + (self.lagrange ** 2).dot(self._soft_ghats())
            loss.backward()
            optimizer.step()
            if self.lagrange is not None:
                # ...but the multiplier is driven by the hard-prediction bound: the
                # soft bound stays positive even at zero true gap (confidence width
                # + margin), which would grow lambda without a fixed point and
                # collapse the model to a trivial constraint-satisfying solution
                with torch.no_grad():
                    hard = torch.clamp(self._hard_ghats(), min=-1.0, max=1.0)
                    self.lagrange += self.lambda_lr * 2 * self.lagrange * hard
                    self.lagrange.clamp_(min=1e-3, max=100.0)
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"epoch {epoch + 1}: loss={loss.item():.4f} "
                      f"lambda={self.lagrange.tolist() if self.lagrange is not None else None}")
        if self._safetyTest() > 0:
            return None
        return self

    def _safetyTest(self, predict=False, ub=True):
        X_test = self.X if predict else self.X_s
        y_test = self.y if predict else self.y_s
        for g_hat in self.constraints:
            y_preds = self.predict(X_test)
            ghat_val = g_hat['fn'](np.asarray(X_test), np.asarray(y_test), y_preds,
                                   g_hat['delta'], n=self.X_s.shape[0],
                                   predict=predict, ub=ub)
            if ghat_val > 0:
                return ghat_val
        return 0

    def predict(self, X):
        with torch.no_grad():
            logits = self.mod(torch.as_tensor(np.asarray(X), dtype=torch.float))
            return torch.argmax(logits, dim=1).numpy()

    def parameters(self):
        return self.mod

    def data(self):
        return self.X, self.y


class NeuralNetSeldonianGD(LogisticRegressionSeldonianGD):
    """
    Gradient-based Seldonian classifier with a neural network.

    Same training scheme as :class:`LogisticRegressionSeldonianGD` (Adam on a
    differentiable surrogate constraint, Lagrangian dual ascent, margin tightening,
    unchanged hard safety test) with a multi-layer perceptron instead of a linear
    model. Pass ``hidden_layers`` to set the architecture, or ``model`` to supply
    any torch module mapping inputs to two logits.
    """

    def __init__(self, X, y, g_hats=[], hidden_layers=(32, 16), model=None, **kwargs):
        self._hidden_layers = hidden_layers
        self._custom_model = model
        # a flexible model exploits the soft surrogate harder than a linear one, so the
        # surrogate/hard-prediction gap is bigger - default to a wider safety margin
        kwargs.setdefault('margin', 0.15)
        super().__init__(X, y, g_hats=g_hats, **kwargs)

    def _build_model(self, n_features):
        if self._custom_model is not None:
            return self._custom_model
        layers = []
        prev = n_features
        for width in self._hidden_layers:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)


class PDISSeldonianPolicyCMAES(CMAESModel, SeldonianAlgorithm):

    def __init__(self, data, states, actions, gamma, threshold=2, test_size=0.4,
                 multiprocessing=True, delta=0.05):
        self.theta = np.random.rand(states * actions, 1)
        self.gamma = gamma
        self.D = data
        self.s = states
        self.a = actions
        self.thres = threshold
        self.delta = delta
        if multiprocessing and ray is None:
            raise ImportError(
                "ray is required for multiprocessing=True; install with `uv sync --extra ray`")
        self.use_ray = multiprocessing
        self.D_c, self.D_s = train_test_split(data, test_size=test_size)
        super(PDISSeldonianPolicyCMAES, self).__init__(self.D_c, None, theta=self.theta,
                                                       maxiter=1000, verbose=True)

    def loss(self, X, y_true, theta):
        est = self.pdis_estimate(theta, X, minimize=False, sum_red=False, verbose=True)
        loss = (-1 * np.sum(est) / len(X)) + (
            0 if self._safetyTest(theta, predict=True, ub=True, est=est) < 0 else 10000)
        print(f"Loss: {loss}")
        return loss
        pass

    def predict(self, X):
        return self._predict(X, self.theta)

    def _predict(self, X, theta):
        theta = theta.reshape(self.s, self.a)
        est = self.pdis_estimate(theta, X, minimize=False, verbose=True)
        return est
        pass

    def pdis_estimate(self, pi_e, D, gamma=0.95, minimize=True, verbose=False, sum_red=True):
        if D is None:
            raise ValueError("Data D is None")
        n = len(D)
        if verbose:
            print(f"Running PDIS estimation for the entire candidate data of {len(D)} samples")
        a = time()
        pi_e = pi_e.reshape(self.s, self.a)
        if self.use_ray:
            n_work = max(int(n / 1e4 * 5), 1)
            works = []
            for i in range(n_work):
                start = int(n * i / n_work)
                end = int(n * (i + 1) / n_work)
                works.append(estimate_ray_vec.remote(pi_e, D[start:end], n, gamma, sum_red))
            results = ray.get(works)
            # ray workers return partial results that still need combining
            est = sum(results) if sum_red else list(itertools.chain.from_iterable(results))
        else:
            # the serial path is already fully reduced by estimate_vec
            est = estimate_vec(pi_e, D, n, gamma, sum_red)

        if verbose:
            print(f"Estimation for one complete run done in {time() - a} seconds")
        if verbose and sum_red:
            print(f"Average estimate of return: {est}")
        if sum_red:
            return est * (-1 if minimize else 1)
        else:
            return est

    def _safetyTest(self, theta, predict=False, ub=False, est=None):
        X = self.D_s
        n = len(self.D_s)
        if predict:
            X = self.D_c
        if est is None:
            estimate = self.pdis_estimate(theta, X, minimize=False, sum_red=not ub)
        else:
            estimate = est
        estimate = np.array(estimate)
        if ub:
            # performance-floor constraint: pass only if the LOWER confidence bound on the
            # policy return clears the threshold
            return -1 * (ttest_bounds(estimate, self.delta, n=n, predict=predict).lower -
                         self.thres)
        else:
            return -1 * (np.mean(estimate) - self.thres)


class SeldonianCEMPDISPolicy(SeldonianAlgorithm):

    def __init__(self, data, states, actions, gamma, threshold=1.41537, test_size=0.4,
                 verbose=False, use_ray=False, delta=0.05):
        self.theta = np.random.rand(states * actions)
        self.gamma = gamma
        self.D = data
        self.s = states
        self.a = actions
        self.thres = threshold
        self.delta = delta
        self.verbose = verbose
        if use_ray and ray is None:
            raise ImportError(
                "ray is required for use_ray=True; install with `uv sync --extra ray`")
        self.use_ray = use_ray
        self.D_c, self.D_s = train_test_split(data, test_size=test_size)

    def loss(self, y_true, y_pred, theta):
        return y_pred + (
            0 if self._safetyTest(theta, predict=True, ub=True) < 0 else 10000)
        pass

    def objective(self, theta, data):
        obj = (-1 * self._predict(data, theta)) + (
            10000 if self._safetyTest(theta, predict=True, ub=True) > 0 else 0)
        if self.verbose:
            print(f"Estimate: {obj}")
        return obj

    def fit(self, method='Powell'):
        if self.verbose:
            print("Running minimization")
        a = time()
        res = minimize(self.objective, self.theta, args=(self.D_c,), method=method,
                       options={'maxfev': 100})
        if self.verbose:
            print(f"Optimization result: {res}")
            print(f"Time takes: {time() - a} seconds")
        self.theta = res.x
        pass

    def _predict(self, X, theta):
        theta = theta.reshape(self.s, self.a)
        est = self.pdis_estimate(theta, X, minimize=False)
        return est
        pass

    def predict(self, X):
        return self._predict(X, self.theta)
        pass

    def data(self):
        return self.D
        pass

    def pdis_estimate(self, pi_e, D, gamma=0.95, minimize=True, sum_red=True):
        if D is None:
            raise ValueError("Data D is None")
        n = len(D)
        if self.verbose:
            print(f"Running PDIS estimation for the entire candidate data of {len(D)} samples")
        pi_e = pi_e.reshape(self.s, self.a)
        # est = 0.0
        # R = []
        if self.use_ray:

            n_work = 12
            idx = 0
            works = []
            for i in range(n_work):
                start = int(n * i / n_work)
                end = int(n * (i + 1) / n_work)
                works.append(estimate_ray_vec.remote(pi_e, D[start:end], n, gamma, sum_red))
            results = ray.get(works)
            # ray workers return partial results that still need combining
            est = sum(results) if sum_red else list(itertools.chain.from_iterable(results))
        else:
            # the serial path is already fully reduced by estimate_vec
            est = estimate_vec(pi_e, D, n, gamma, sum_red)
        if self.verbose and sum_red:
            print(f"Average estimate of return: {est}")
        return est * (-1 if minimize else 1)

    def _safetyTest(self, theta, predict=False, ub=False):
        X = self.D_s
        n = len(self.D_s)
        if predict:
            X = self.D_c
        estimate = self.pdis_estimate(theta, X, minimize=False, sum_red=not ub)
        estimate = np.array(estimate)
        if ub:
            # performance-floor constraint: pass only if the LOWER confidence bound on the
            # policy return clears the threshold
            return -1 * (ttest_bounds(estimate, self.delta, n=n, predict=predict).lower -
                         self.thres)
        else:
            return -1 * (np.mean(estimate) - self.thres)


def estimate_vec(pi_e, D, n, gamma=0.95, sum_red=True):
    if sum_red:
        est = 0.0
    else:
        est = []
    pi_e = softmax(pi_e, axis=1)
    for ep in D:
        ep = np.array(ep, dtype=float)
        weights = np.cumprod(
            pi_e[ep[:, 0].astype(int), ep[:, 1].astype(int)] * gamma / ep[:,
                                                                             3]) / gamma
        if sum_red:
            est += weights.dot(ep[:, 2])
        else:
            est.append(weights.dot(ep[:, 2]))
    return est / n if sum_red else est


def estimate_ray_vec(pi_e, D, n, gamma=0.95, sum_red=True):
    return estimate_vec(pi_e, D, n, gamma=gamma, sum_red=sum_red)


if ray is not None:
    estimate_ray_vec = ray.remote(estimate_ray_vec)

import torch
import torch.utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
import numpy as np

from seldonian.algorithm import SeldonianAlgorithm


# torch.autograd.set_detect_anomaly(True)


class VanillaNN(SeldonianAlgorithm):
    def __init__(self, X, y, test_size=0.4, g_hats=[], verbose=False, stratify=False, epochs=10):
        self.X = X
        self.y = y
        D = self.X.shape[1]
        H1 = int(D * 0.5)
        self.constraint = g_hats
        self.X, self.X_s, self.y, self.y_s = train_test_split(
            self.X, self.y, test_size=test_size, random_state=0,
            stratify=[0, 1] if stratify else None
        )
        self.X = torch.as_tensor(self.X, dtype=torch.float)
        self.y = torch.as_tensor(self.y, dtype=torch.long)
        self.X_s = torch.as_tensor(self.X_s, dtype=torch.float)
        self.y_s = torch.as_tensor(self.y_s, dtype=torch.long)
        self.verbose = verbose
        self.epochs = epochs
        self.mod = nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.Linear(H1, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.constraint = []
        if len(self.constraint) > 0:
            self.lagrange = torch.ones((len(self.constraint),), requires_grad=True)
        else:
            self.lagrange = Nones

        self.dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.loader = DataLoader(self.dataset, batch_size=300)
        if self.lagrange is not None:
            params = nn.ParameterList(self.mod.parameters())
            self.optimizer = torch.optim.Adam(params, lr=3e-3)
            # self.l_optimizer = torch.optim.Adam([self.lagrange], lr=0.1)
            self.l_optimizer = torch.optim.Adam([self.lagrange], lr=3e-4)
        else:
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
                        self.safetyTest(predict=True))
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
                        self.safetyTest(predict=True)))
                    loss_f.backward(retain_graph=True)
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
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float)
        if not pmf:
            preds = torch.argmax(self.mod(X), dim=1)
        else:
            preds = nn.Softmax(dim=1)(self.mod(X))[:, 1]
        return preds

    def safetyTest(self, predict=False, ub=True):
        with torch.no_grad():
            X_test = self.X if predict else self.X_s
            y_test = self.y if predict else self.y_s

        ghats = torch.empty(len(self.constraint))
        i = 0
        for g_hat in self.constraint:
            y_preds = self.predict(X_test, True)
            ghats[i] = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub, est=self.mod)
            # ghats[i] = ghat_val
            i += 1
        return ghats

    def data(self):
        return self.X, self.y


class NeuralNetModel(SeldonianAlgorithm, pl.LightningModule):
    def __init__(self, X, y, test_size=0.4, g_hats=[], verbose=False, hard_barrier=False,
                 stratify=False):
        super().__init__()
        self.X = X
        self.y = y
        D = self.X.shape[1]
        H1 = int(D * 1.6)
        self.constraint = g_hats
        self.X, self.X_s, self.y, self.y_s = train_test_split(
            self.X, self.y, test_size=test_size, random_state=0,
            stratify=[0, 1] if stratify else None
        )
        self.verbose = verbose
        self.mod = nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.Linear(H1, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        if len(self.constraint) > 0:
            self.lagrange = torch.rand((len(self.constraint),), requires_grad=True)
        else:
            self.lagrange = None

        self.dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        self.loader = DataLoader(self.dataset)

    def forward(self, x):
        return self.mod(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mod(x)
        safety = self.safetyTest(predict=True)
        loss = self.loss_fn(y_pred, y)
        loss += self.lagrange.dot(safety)
        return loss

    def configure_optimizers(
            self,
    ):
        print(f"Lagrange multipliers: {self.lagrange}")
        # print(f"Lagrange multipliers: {list(self.parameters())}")
        if self.lagrange is not None:
            optimizer = torch.optim.Adam(list(self.parameters()) + list(self.lagrange),
                                         lr=3e-3)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer

    def safetyTest(self, predict=False, ub=True):
        print("In safety test")
        X_test = self.X if predict else self.X_s
        y_test = self.y if predict else self.y_s

        ghats = torch.zeros(len(self.constraint), requires_grad=True)
        i = 0
        for g_hat in self.constraints:
            y_preds = self.forward(X_test)
            ghat_val = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub)
            # if ghat_val > 0:
            #     if self.hard_barrier is True and predict is True:
            #         return 1
            #     else:
            #         return ghat_val
            ghats[i] += ghat_val
            i += 1
        return ghats
        pass

    def data(self):
        return self.X, self.y

    def fit(self, **kwargs):
        trainer = pl.Trainer()
        trainer.fit(self, self.loader)
        pass

    def predict(self, X):
        return self.__call__(X)


def grad_check(named_params):
    avg = []
    for n, p in named_params:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                avg.append(p.grad.abs().mean())
    print(f"Average gradient flow: {np.mean(avg)}")
    pass

from seldonian import *
import numpy as np
import torch
import torch.utils
import torch.nn as nn
from torch.utils.data import random_split
import itertools


class VanillaNN(SeldonianAlgorithm):
    def __init__(self, X, y, test_size=0.4, g_hats=[], verbose=False, stratify=False):
        self.X = X
        self.y = y
        D = self.X.shape[1]
        H1 = int(D * 1.6)
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
        self.mod = nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.Linear(H1, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.constraint = []
        if len(self.constraint) > 0:
            self.lagrange = torch.empty((len(self.constraint),), requires_grad=True)
        else:
            self.lagrange = None

        self.dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.loader = DataLoader(self.dataset, batch_size=16)
        if self.lagrange is not None:
            params = nn.ParameterList(list(self.mod.parameters()) + [nn.Parameter(self.lagrange)])
            self.optimizer = torch.optim.Adam(params, lr=3e-3)
        else:
            self.optimizer = torch.optim.Adam(self.mod.parameters(), lr=3e-3)
        pass

    def fit(self, **kwargs):
        running_loss = 0.0
        for epoch in range(20):
            for i, data in enumerate(self.loader, 0):
                x, y = data
                # print(x.shape, y.shape)
                self.optimizer.zero_grad()

                out = self.mod(x)
                if self.lagrange is not None:
                    loss = self.loss_fn(out, y) - self.lagrange.dot(self.safetyTest(predict=True))
                else:
                    loss = self.loss_fn(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print("Training done.")
        pass

    def predict(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float)
        preds = torch.argmax(self.mod(X), dim=1)
        return preds

    def safetyTest(self, predict=False, ub=True):
        with torch.no_grad():
            X_test = self.X if predict else self.X_s
            y_test = self.y if predict else self.y_s

        ghats = torch.empty(len(self.constraint), requires_grad=True)
        i = 0
        for g_hat in self.constraint:
            y_preds = self.predict(X_test)
            ghat_val = g_hat['fn'](X_test, y_test, y_preds, g_hat['delta'], self.X_s.shape[0],
                                   predict=predict, ub=ub)
            with torch.no_grad():
                ghats[i] = ghat_val
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

from abc import ABC

import scipy

from seldonian.algorithm import SeldonianAlgorithm

import numpy as np


class CMAESModel(ABC):
    def __init__(self, X, y, verbose=False):
        self.theta = np.random.default_rng(0).random((X.shape[1] + 1, 1))
        self.X = X
        self.y = y
        self.sigma = 0.3
        self.stopfitness = 1e-9
        self.C = None
        self.verbose = verbose

    def data(self):
        return self.X, self.y

    def fit(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        stop_iter_count = 0
        last_loss = 0
        N = self.theta.size
        self.stopeval = 100 * N ** 2
        self.max_iter_no_change = max(1000, 15*np.sqrt(self.stopeval).astype(np.int))
        if self.verbose:
            print(f"Max number of iters: {self.max_iter_no_change}")
        sigma = self.sigma

        xmean = np.random.default_rng().random(self.theta.shape)
        lambda_p = int(4 + np.floor(3 * np.log(N)))
        mu = int(lambda_p / 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = (weights / np.sum(weights)).reshape((weights.size, 1))
        mueff = (np.sum(weights) ** 2) / (np.sum(weights ** 2))

        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

        pc = np.zeros((N, 1))
        ps = np.zeros((N, 1))
        B = np.eye(N, N)
        D = np.ones((N, 1))
        C = B @ np.diag(D.flatten() ** 2) @ B.T
        invsqrtC = B @ np.diag(D.flatten() ** -1) @ B.T
        eigenval = 0
        chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        if self.verbose:
            print(f"max iterations: {self.stopeval}")
        counteval = 0
        while counteval < self.stopeval:
            arx = np.zeros((self.theta.size, lambda_p))
            arfitness = np.zeros((lambda_p,))
            for k in range(lambda_p):
                arx[:, k] = (xmean + sigma * B @ (D * np.random.randn(N, 1))).flatten()
                arfitness[k] = self.loss(y_true=y, y_pred=self._predict(X, arx[:, k]),
                                         theta=arx[:, k])
                counteval += 1
            arindex = np.argsort(arfitness)
            arfitness = arfitness[arindex]
            if self.verbose:
                print(
                    f"Current evaluation: {counteval}\t average loss:{arfitness[1]} ",
                    end='\r')
            xold = np.copy(xmean)
            xmean = arx[:, arindex[:mu]] @ weights

            ps = (1 - cs) * ps + (
                    np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma)
            hsig = np.linalg.norm(ps) / (
                    np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_p)) / chiN) < 1.4 + 2 / (
                           N + 1)
            pc = (1 - cc) * pc + (hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma)
            artmp = (1 / sigma) * ((arx[:, arindex[:mu]]) - xold)

            C = (1 - c1 - cmu) * C + c1 * (
                    pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(
                weights.flatten()) @ artmp.T

            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            if counteval - eigenval > lambda_p / (c1 + cmu) / N / 10:
                eigenval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D, B = scipy.linalg.eigh(C)
                D = np.sqrt(D)
                invsqrtC = B @ np.diag(D ** -1) @ B.T
                D = D.reshape((D.size, 1))

            # stopping criterias
            if arfitness[0] <= self.stopfitness or np.max(D) > 1e7 * np.min(D):
                break

            if abs(arfitness[
                       0] - last_loss) <= self.stopfitness:
                if counteval - stop_iter_count > self.max_iter_no_change:
                    if self.verbose:
                        print(
                            f"\n\nStopping early after no change in {counteval - stop_iter_count} iterations !!")
                    break
            else:
                stop_iter_count = counteval
                last_loss = arfitness[0]
        self.theta = arx[:, arindex[0]]
        self.C = C

    def parameters(self):
        return self.theta, self.C

    def reset(self):
        self.theta = np.zeros_like(self.theta)
        pass

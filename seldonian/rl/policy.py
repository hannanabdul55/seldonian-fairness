from numba.core.types.containers import DictType
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32, float64, int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit

from time import time

from rl_utils import *
from gridworld_obstacle import *


@jitclass
class TabularSoftmaxPolicy:
    n_actions: int64
    n_states: int64
    phi: float64[:, :]
    phif: float64[:]
    action: int64
    state: int64
    actions: int64[:]
    eps: float64
    a: int64
    s: int64
    theta: float64[:]
    env: GridWorld
    m: int64
    n: int64
    reset_env: bool

    def __init__(
        self, a, s,
        eps=0.1
    ) -> None:
        self.a = a
        self.s = s
        self.actions = np.arange(self.a, dtype=np.int64)
        self.eps = eps
        # print(self.a, self.s)
        self.phi = np.zeros((self.a, self.s), dtype=np.float64)
        for i in prange(self.a):
            for j in prange(self.s):
                self.phi[i, j] = np.random.randn()
        # self.phi = phif.reshape((self.a, self.s))

    def get_prob_for_state(self, state: int64):
        pros = softmax_optimized(self.phi[:, state])
        return pros

    def choose_action(self, state: int64):
        rn = np.random.random()
        # epsilon greedy
        if rn < self.eps:
            # take random action
            # print("taking random action")
            return np.random.choice(self.a)
        # choose action based on
        return rand_choice_nb(self.actions, self.get_prob_for_state(state))

    def reset(self, reset_env=False):
        for i in np.arange(self.s):
            for j in np.arange(self.a):
                self.phi[i, j] = np.random.randn()
        if reset_env:
            self.env.reset()

    def derivative(self, st: int64, ac: int64):
        feats = s_to_onehot(st, self.s)
        actionProbs = self.get_prob_for_state(st)
        res = np.zeros((self.a, self.s), dtype=np.float64)

        for i in np.arange(self.a):
            if i == ac:
                # print("ActionProbs", (1-actionProbs[i]) * feats)
                res[i, :] = (1-actionProbs[i]) * feats
            else:
                # print("ActionProbs", (-actionProbs[i]) * feats)
                res[i, :] = (-1*actionProbs[i]) * feats
        # print(res.shape, feats)
        return res

    def q(self, s: int64, a: int64):
        return self.phi[s, a]

    def set_theta(self, theta: float64[:, :]):
        if theta.shape == self.phi.shape:
            # print("Theta shape: ",theta.shape)
            for i in range(self.a):
                for j in range(self.s):
                    self.phi[int(i), int(j)] = theta[int(i), int(j)]
        else:
            print(
                "ERROR: shape of input:",
                theta.shape, "not equal to shape of theta",
                self.phi.shape
            )


if __name__ == "__main__":
    seed = 111
    gw = get_gw_from_seed(seed, path)
    pi = TabularSoftmaxPolicy(
        gw, s=gw.len_states,
        a=gw.len_actions, seed=seed
    )
    probs = pi.choose_action(3)
    print(probs)

    theta = np.random.randn(25*4)
    pi.reset()
    probs = pi.choose_action(7)
    print(probs)

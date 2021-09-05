from numba.core.types.containers import DictType
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32,float64,int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit

from time import time

from rl_utils import *

@jitclass
class TabularSoftmaxPolicy:
    n_actions: int64
    n_states: int64
    phi: float64[:]
    phif: float64[:]
    action: int64
    a: int64
    s: int64
    theta: float64[:]
    def __init__(self, n_actions, n_states, seed=42) -> None:
        np.random.seed(seed)
        self.a = n_actions
        self.s = n_states
        phif = np.zeros(n_states* n_actions, dtype=np.float64)
        for i in prange(n_actions*n_states):
            phif[i] = np.random.randn()
        self.phi = phif

    def get_prob_for_action(self, action: int64):
        return softmax_optimized(self.phi[action*self.s:action*self.s + self.a])

    def take_action(self, action):
        return self.phi[action*self.s:action*self.s + self.a]
    
    def set_theta(self, theta):
        self.phi = theta


if __name__ == "__main__":
    pi = TabularSoftmaxPolicy(5,4)
    probs = pi.get_prob_for_action(3)
    print(probs, probs.sum())

    theta = np.random.randn(5*4)
    pi.set_theta(theta)
    probs = pi.get_prob_for_action(3)
    print(probs, probs.sum())
    

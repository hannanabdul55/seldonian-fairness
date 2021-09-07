from numba.core.types.containers import DictType
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32, float64, int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit
import numba as nb
from time import time

from rl_utils import *
from policy import *
from gridworld_obstacle import *


@jitclass
class ReinforceGridworld:
    gamma: float64
    alpha: float64
    policy: TabularSoftmaxPolicy
    env: GridWorld
    s_history: types.ListType(types.int64)
    r_history: types.ListType(types.float64)
    a_history: types.ListType(types.int64)
    def __init__(
        self, env, policy, seed=42,
        eps=0.1,
        gamma=1.0,
        alpha=0.2
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.policy = policy
        self.s_history= typed.List.empty_list(types.int64)
        self.r_history= typed.List.empty_list(types.float64)
        self.a_history= typed.List.empty_list(types.int64)

    def set_policy(self, theta):
        self.policy.set_theta(theta)
    
    def act(self):
        state = self.env.state
        action = self.policy.choose_action(state)
        s, a, r, sPrime = self.env.take(action)
        
        # add into trajectory
        self.s_history.append(int(s))
        self.a_history.append(int(a))
        self.r_history.append(r)

        if self.env.is_sinf():
            self.update
        return r


if __name__=="__main__":
    # print(nb.typeof(list()))
    seed=42
    gw = get_gw_from_seed(seed)
    pol = TabularSoftmaxPolicy(gw, seed=seed)
    reinforce = ReinforceGridworld(gw, pol, seed=seed)
    print(reinforce.act())

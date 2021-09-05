from numba.core.types.containers import DictType
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32,float64,int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit

from time import time

from rl_utils import *
from policy import *
from gridworld_obstacle import *


class reinforce_rl:

    def __init__(self, n_actions, n_states, seed=42) -> None:
        self.policy = TabularSoftmaxPolicy(n_actions, n_states, seed)
        


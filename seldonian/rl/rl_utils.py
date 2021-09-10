import time
import math
import typing
from numba.core.decorators import njit
from numba.np.ufunc import parallel
import numpy as np
from numba import jit, prange, typed
from actor_critic import *
from gridworld_obstacle import *

# @jit("f8(f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
@njit
def esum(z):
    return np.sum(np.exp(z))

# @jit("f8[:](f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
@njit(parallel=True)
def softmax_optimized(z):
    num = np.exp(z)
    s = num / esum(z)
    return s

@njit
def s_to_onehot(s, total):
    o = np.zeros(total)
    o[s]=1
    return o

@njit(parallel=True)
def one_hot_to_s(o):
    for i in np.arange(len(o)):
        if o[i]==1:
            return i

@njit
def get_eps_for_n(n: int, tot=10000.0):
    if n==0:
        return 0
    return int(np.ceil(tot/n))

def stderror(v):
	non_nan = np.count_nonzero(~np.isnan(v))        # number of valid (non NaN) elements in the vector
	return np.nanstd(v, ddof=1) / np.sqrt(non_nan)

@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    if np.sum(np.isfinite(prob))==len(arr):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]
    else:
        # print("Did not choose any action. THIS SHOULD NOT HAPPEN, prob:", prob, " arr: ", arr)
        return np.random.choice(arr)
    # r = np.random.random()
    # s = 0.0
    # for i in range(len(prob)):
    #     s += prob[i]
    #     if s >= r:
    #         return i
    # print("Did not choose any action. THIS SHOULD NOT HAPPEN, prob:", prob, " arr: ", arr)
    # return len(arr)-1

@njit
def run_episode(agent, mdp_repr: np.ndarray, freeze=False,):
    agent.env.set_repr(mdp_repr)
    agent.newepisode()
    agent.freeze(freeze)
    while not agent.env.is_terminated():
        agent.act()
    return agent.env.rw
    # print("Expected return:", agent.env.rw)
    pass

@njit
def run_episodes(agent, episodes, mdp_repr: np.ndarray, freeze=False):
    rs = []
    for ep in prange(episodes):
        rs.append(run_episode(agent, mdp_repr, freeze))
    return rs

# @njit
# def run_episode(agent, mdp_repr: GridWorld, freeze=False):
#     agent.env.set_repr(mdp_repr.get_repr())
#     agent.newepisode()
#     agent.freeze(freeze)
#     while not agent.env.is_terminated():
#         agent.act()
#     print("Expected return:", agent.env.rw)
#     pass

def create_n_mdps(n: int64, start: int64 = 0):
    mdps = []
    for i in range(start, start+n):
        mdps.append(get_gw_from_seed(i))
    return mdps
    pass

if __name__=="__main__":
    a = np.zeros((100,))
    a[30]=1
    print(one_hot_to_s(a))
    print(s_to_onehot(30, 100))
    
    print(get_eps_for_n(21))

    # arr = test_loop_par()

    # arr1 = np.arange(10000)
    # print(np.array_equal(arr, arr1))
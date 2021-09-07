import time
import math
from numba.core.decorators import njit
from numba.np.ufunc import parallel
import numpy as np
from numba import jit, prange

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
     

if __name__=="__main__":
    a = np.zeros((100,))
    a[30]=1
    print(one_hot_to_s(a))
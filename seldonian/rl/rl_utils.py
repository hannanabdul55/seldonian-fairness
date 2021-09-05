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
    return 0

@njit(parallel=True)
def one_hot_to_s(o):
    for i in np.arange(len(o)):
        if o[i]==1:
            return i

if __name__=="__main__":
    a = np.zeros((100,))
    a[30]=1
    print(one_hot_to_s(a))
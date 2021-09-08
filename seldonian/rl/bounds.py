from numba.core.types.containers import DictType
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32, float64, int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit

from time import time
from scipy.stats import t

def ttest_bounds(samples, delta, n=None):
    if samples.ndim >1:
        # print("samples dimension is greater than 1.")
        samples = samples.flatten()
    if n is None:
        n = samples.size
    dev = ((samples.std(ddof=1) / np.sqrt(n)) * t.ppf(1 - delta, n - 1))
    sample_mean = samples.mean()
    return sample_mean, dev

@njit
def from_njit():
    arr = np.random.randn(200)
    ttest_bounds(arr, 0.1)

if __name__ == "__main__":
    arr = np.random.randn(200)
    print(ttest_bounds(arr, 0.1))
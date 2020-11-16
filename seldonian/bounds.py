import numpy as np
import math
import torch
import numbers
from scipy.stats import t
import torch


class RandomVariable:
    def __init__(self, value, lower=None, upper=None):
        if not (isinstance(value, numbers.Number) or torch.is_tensor(value)):
            raise ValueError(
                f"`value` parameter must be a non-null number")
        self.value = value
        self.upper = upper if upper is not None else value
        self.lower = lower if lower is not None else value
        pass

    def __str__(self):
        return f"( value={self.value}, upper_bound={self.upper}, lower_bound={self.lower} )"

    def __add__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value + other.value)
        return RandomVariable(self.value + other.value, lower=self.lower + other.lower,
                              upper=self.upper + other.upper)

    def __neg__(self):
        if self.lower is None or self.upper is None:
            return RandomVariable(-self.value)
        return RandomVariable(-self.value, upper=-self.lower, lower=-self.upper)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value * other.value)

        uu = self.upper * other.upper
        ul = self.upper * other.lower
        lu = self.lower * other.upper
        ll = self.lower * other.lower

        low = min(uu, ul, lu, ll)
        upper = max(uu, lu, ul, ll)
        return RandomVariable(self.value * other.value, lower=low, upper=upper)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number) or torch.is_tensor(other):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value / other.value)

        # if 0 not in [other.lower , other.upper]
        if other.lower * other.upper > 0:
            return self * RandomVariable(1 / other.value, lower=1 / other.upper,
                                         upper=1 / other.lower)
        # if other.lower is 0
        elif other.lower == 0:
            return self * RandomVariable(1 / other.value, lower=1 / other.upper, upper=np.inf)
        # if other.upper is 0
        elif other.upper == 0:
            return self * RandomVariable(1 / other.value, upper=1 / other.lower, lower=-np.inf)
        # if 0 in [other.lower , other.upper]
        else:
            return RandomVariable(self.value / other.value, lower=-np.inf, upper=np.inf)

    def __abs__(self):
        if self.lower is None or self.upper is None:
            return RandomVariable(abs(self.value))
        # |[lower, upper]| = [0, max(|lower|, |upper|)]
        return RandomVariable(abs(self.value), lower=0,
                              upper=max(abs(self.lower), abs(self.upper)))

    def __hash__(self):
        hash_val = self.value.__hash__()
        if self.lower is not None:
            hash_val += self.lower.__hash__()
        if self.upper is not None:
            hash_val += self.upper.__hash__()
        return hash_val


def min_bounds(*args):
    min_rv = RandomVariable(np.inf, lower=np.inf, upper=np.inf)
    for arg in args:
        if not isinstance(arg, RandomVariable):
            arg = RandomVariable(value=arg)
        if arg.value < min_rv.value:
            min_rv.value = arg.value
            if arg.lower < min_rv.lower:
                min_rv.lower = arg.lower
            if arg.upper < min_rv.upper:
                min_rv.upper = arg.upper
    return min_rv


def max_bounds(*args):
    max_rv = RandomVariable(-np.inf, lower=-np.inf, upper=-np.inf)
    for arg in args:
        if not isinstance(arg, RandomVariable):
            arg = RandomVariable(value=arg)
        if arg.value > max_rv.value:
            max_rv.value = arg.value
            if arg.lower > max_rv.lower:
                max_rv.lower = arg.lower
            if arg.upper > max_rv.upper:
                max_rv.upper = arg.upper
    return max_rv


def ttest_bounds(samples, delta, n=None, predict=False):
    if not (isinstance(samples, numbers.Number) or isinstance(samples,
                                                              np.ndarray) or torch.is_tensor(
            samples)):
        raise ValueError(f"`samples` argument should be a numpy array")
    is_tensor = torch.is_tensor(samples)
    if not is_tensor:
        samples = np.array(samples)

    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array). Got shape: {samples.shape}")
    if n is None:
        n = samples.size
    # print(f"n={n}")
    if not is_tensor:
        dev = ((samples.std(ddof=1) / np.sqrt(n)) * t.ppf(1 - delta, n - 1)) * (1 + (1 * predict))
    else:
        dev = ((torch.std(samples.double()) / np.sqrt(n)) * t.ppf(1 - delta, n - 1)) * (
                    1 + (1 * predict))

    if torch.is_tensor(samples):
        samples = samples.double()
    sample_mean = samples.mean()
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)


def hoeffdings_bounds(samples, delta, n=None, predict=False):
    if not (isinstance(samples, numbers.Number) or isinstance(samples, np.ndarray)):
        raise ValueError(f"`samples` argument should be a numpy array")
    samples = np.array(samples)
    is_tensor = torch.is_tensor(samples)
    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array)")
    if n is None:
        n = samples.size
    # print(f"n={n}")
    if not is_tensor:
        dev = np.sqrt(np.log(1 / delta) / (2 * n)) * (1 + (1 * predict))
        sample_mean = samples.mean()
    else:
        dev = torch.sqrt(torch.log(1 / delta) / (2 * n)) * (1 + (1 * predict))
        sample_mean = torch.mean(samples)
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)

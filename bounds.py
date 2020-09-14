import numpy as np
import math
import numbers


class RandomVariable:
    def __init__(self, value, lower=None, upper=None):
        if not isinstance(value, numbers.Number):
            raise ValueError(
                f"`value` parameter must be a non-null number")
        self.value = value
        self.upper = upper if upper is not None else value
        self.lower = lower if lower is not None else value
        pass

    def __str__(self):
        return f"( value={self.value}, upper_bound={self.upper}, lower_bound={self.lower} )"

    def __add__(self, other):
        if isinstance(other, numbers.Number):
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
        if isinstance(other, numbers.Number):
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
        if isinstance(other, numbers.Number):
            other = RandomVariable(other, lower=other, upper=other)

        if self.lower is None or self.upper is None or other.lower is None or other.upper is None:
            return RandomVariable(self.value / other.value)

        # if 0 not in [other.lower , other.upper]
        if other.lower * other.upper > 0:
            return self * RandomVariable(1 / other.value, lower=1/other.upper, upper=1/other.lower)
        # if other.lower is 0
        elif other.lower == 0:
            return self * RandomVariable(1 / other.value, lower=1/other.upper, upper=np.inf)
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
        return RandomVariable(abs(self.value), lower=0, upper=max(abs(self.lower), abs(self.upper)))

    def __hash__(self):
        hash_val = self.value.__hash__()
        if self.lower is not None:
            hash_val += self.lower.__hash__()
        if self.upper is not None:
            hash_val += self.upper.__hash__()
        return hash_val









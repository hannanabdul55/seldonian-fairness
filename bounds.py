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

    # def __truediv__(self, other):




from bounds import *
import numpy as np

# Basic example showing bound calculation for 2 variables with defined upper and lwoer bounds
a = RandomVariable(4, lower=-1, upper=6)
b = RandomVariable(5, lower=-3, upper=7)

# Calculate bounds after doing a * b
print(f"a={a}")
print(f"b={b}")
print(f"a+b = {a + b}\n")

# The bounds default to the value passed if not given
c = RandomVariable(4)
d = RandomVariable(5)

print(f"c = {c} \t d = {d}\n")
# print bounds of c * d
print(f"c*d = {c * d}\n")

# Bounds for constants are again just defaulted to the input value
e = RandomVariable(40, lower=30, upper=50)
print(f"e={e}")
print(f"e*3={e * 3}")

# Bounds for division
f = RandomVariable(40, lower=30, upper=50)
g = RandomVariable(2, lower=0.5, upper=4)
print(f"f={f}\tg={g}")
print(f"f/g={f / g}")

# Bounds for division with \infinity in the resultant bound
h = RandomVariable(40, lower=30, upper=50)
i = RandomVariable(2, lower=-2, upper=4)
print(f"h={h}\ti={i}")
print(f"h/i={h / i}")


# example with a gHat

# g_hat = (TP(X[a_idx] = a_val) / TP()) - threshold
def gHat_tp_rate_diff(X, y_pred, y_true, a_val, a_idx, threshold):
    tp_a = np.sum((X[:, a_idx] == a_val) & ((y_true == 1) & (y_pred == 1)))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tp_a / tp - threshold


X_t = np.array([[0.5, 0, 1, 0],
                [0.5, 0, 1, 1],
                [0, 1, 1, 1],
                [0.5, 0, 0, 0],
                [0.5, 1, 1, 1]
                ])
y_t = np.array([1, 1, 1, 0, 1])
y_pred = np.array([1, 1, 1, 1, 1])
print(gHat_tp_rate_diff(X_t, y_pred, y_t, 0.5, 0, 0.2))

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
print(f"e*3={e*3}")

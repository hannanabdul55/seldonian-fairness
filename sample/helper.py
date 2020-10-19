import math
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import t
from scipy.optimize import minimize # The black-box optimization algorithm used to find a candidate solution

np.set_printoptions(precision=5, suppress=True)


# This function returns the inverse of Student's t CDF using the degrees of freedom in nu for the corresponding
# probabilities in p. It is a Python implementation of Matlab's tinv function: https://www.mathworks.com/help/stats/tinv.html
def tinv(p, nu):
	return t.ppf(p, nu)


# This function computes the sample standard deviation of the vector v, with Bessel's correction
def stddev(v):
	n = v.size
	variance = (np.var(v) * n) / (n-1) # Variance with Bessel's correction
	return np.sqrt(variance)           # Compute the standard deviation


# This function computes a (1-delta)-confidence upper bound on the expected value of a random
# variable using Student's t-test. It analyzes the data in v, which holds i.i.d. samples of the random variable.
# The upper confidence bound is given by 
#    sampleMean + sampleStandardDeviation/sqrt(n) * tinv(1-delta, n-1),
#    where n is the number of points in v.
def ttestUpperBound(v, delta):	
	n  = v.size
	res = v.mean() + stddev(v) / math.sqrt(n) * tinv(1.0 - delta, n - 1)
	return res


# This function works similarly to ttestUpperBound, but returns a conservative upper bound. It uses 
# data in the vector v (i.i.d. samples of a random variable) to compute the relevant statistics 
# (mean and standard deviation) but assumes that the number of points being analyzed is k instead of |v|.
# This function is used to estimate what the output of ttestUpperBound would be if it were to
# be run on a new vector, v, containing values sampled from the same distribution as
# the points in v. The 2.0 factor in the calculation is used to double the width of the confidence interval,
# when predicting the outcome of the safety test, in order to make the algorithm less confident/more conservative.
def predictTTestUpperBound(v, delta, k):
	# conservative prediction of what the upper bound will be in the safety test for the a given constraint
	res = v.mean() + 2.0 * stddev(v) / math.sqrt(k) * tinv(1.0 - delta, k - 1)
	return res
	

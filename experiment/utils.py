import numpy as np


def stderror(v):
	non_nan = np.count_nonzero(~np.isnan(v))        # number of valid (non NaN) elements in the vector
	return np.nanstd(v, ddof=1) / np.sqrt(non_nan)
import numpy as np

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import minkowski_distance_p, minkowski_distance

def calc_dist(C, S, n=1, method='directed_hausdorff'):
    if method== 'directed_hausdorff':
        return directed_hausdorff(C, S)[0] / n
    elif method=='minkowski':
        return minkowski_distance(C, S)
    pass

def stderror(v, axis=None):
	non_nan = np.count_nonzero(~np.isnan(v), axis=axis)        # number of valid (non NaN) elements in the vector
	return np.nanstd(v, ddof=1, axis=axis) / np.sqrt(non_nan)
from sklearn.model_selection import train_test_split

from seldonian.synthetic import *
from sklearn.linear_model import *

import numpy as np


def main(N, D, noise):
    X, y = regression_make_synthetic(N, D, noise=noise, random_state=13)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    model = LinearRegression(normalize=True).fit(X_tr, y_tr)
    print(f"[Train] MSE: {np.mean(np.square(y_tr - model.predict(X_tr)))}")
    print(f"[Test] MSE: {np.mean(np.square(y_te - model.predict(X_te)))}")
    pass

if __name__ == '__main__':
    D = 20
    for N in np.geomspace(100, 50000, 20):
        for noise in np.geomspace(0.1, 5, 10):
            print(f"\n\nRunning experiment for N={N} | D = {D} | noise={noise}")
            main(int(N), D, noise)
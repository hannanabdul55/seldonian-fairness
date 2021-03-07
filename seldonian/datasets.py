import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tempeh.configurations import datasets


class LawschoolDataset:
    def __init__(self, verbose=False, n=None, **kwargs):
        self.data = datasets['lawschool_passbar'](drop_gender=False, drop_race=False, **kwargs)
        dataset = self.data

        # X, X_test = dataset.get_X(format=pd.DataFrame)
        # y, y_test = dataset.get_y(format=pd.Series)

        X = pd.concat(dataset.get_X(format=pd.DataFrame))
        y = pd.concat(dataset.get_y(format=pd.Series))

        if 'sensitive_feature' not in kwargs:
            self.A = 'black'
        else:
            self.A = kwargs['sensitive_feature']
        A = self.A
        A_arr = X.loc[:,A]
        self.le = preprocessing.LabelEncoder()
        X.loc[:, A] = self.le.fit_transform(X[A])
        self.scaler = preprocessing.StandardScaler()
        X.loc[:, ['lsat', 'ugpa']] = self.scaler.fit_transform(X[['lsat', 'ugpa']])

        X = X.to_numpy()
        y = y.to_numpy()

        X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=1,
                                                stratify=A_arr)
        if n is not None:
            X, y = sklearn.utils.resample(X, y, n_samples=n, random_state=1)
        # print(dataset._features)
        self.A_idx = dataset._features.index(A)

        if verbose:
            rates = np.mean((X[:, self.A_idx] == 1) & (y == 1))
            print(f"True positive rate for feature {A}=1: {rates}")
            print(f"number of samples for feature {A}=1: {np.sum(X[:, self.A_idx] == 1)}")

            rates = np.mean((X[:, self.A_idx] == 0) & (y == 1))
            print(f"True positive rate for feature {A}=0: {rates}")
            print(f"number of samples for feature {A}=0: {np.sum(X[:, self.A_idx] == 0)}")

            print("FOR TEST SET:")
            rates = np.mean((X_test[:, self.A_idx] == 1) & (y_test == 1))
            print(f"True positive rate for feature {A}=1: {rates}")
            print(f"number of samples for feature {A}=1: {np.sum(X_test[:, self.A_idx] == 1)}")

            rates = np.mean((X_test[:, self.A_idx] == 0) & (y_test == 1))
            print(f"True positive rate for feature {A}=0: {rates}")
            print(f"number of samples for feature {A}=0: {np.sum(X_test[:, self.A_idx] == 0)}")

        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test

    def get_data(self):
        return self.X, self.X_test, self.y, self.y_test, self.A, self.A_idx

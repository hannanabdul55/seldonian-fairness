from seldonian.algorithm import *
import numpy as np

class SampleCMAES(CMAESModel):

    def __init__(self, X, y):
        super().__init__(X, y)

    def _predict(self, X, theta):
        return theta**2 + 4

    def predict(self, X):
        pass

    def loss(self, y_pred, y_true, theta):
        return y_pred


model = SampleCMAES(np.array([[0]]), np.array(0))
model.theta = np.array([100.0], dtype=np.float)

model.fit()
print(model.theta)
print(model._predict(2, model.theta))
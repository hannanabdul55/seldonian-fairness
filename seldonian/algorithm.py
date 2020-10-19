class SeldonianAlgorithm:
    def __init__(self):
        pass

    def model(self):
        raise NotImplementedError("bounds function must be implemented")
        pass

    def data(self):
        raise NotImplementedError("subclass must implement data method to return the data")
        pass

    def bounds(self):
        raise NotImplementedError("bounds function must be implemented")
        pass


class Model:
    def __init__(self):
        pass

    def data(self):
        raise NotImplementedError("data method must be implemented")

    def fit(self, X, y):
        raise NotImplementedError("fit method should be implemented")

    def parameters(self):
        raise NotImplementedError("parameters method should return the model parameters learnt")

    def _predict(self, X, theta):
        raise NotImplementedError(
            "_predict should be overriden. Should use theta as the model params")

    def predict(self, X):
        raise NotImplementedError("predict should be overriden")

    def reset(self):
        raise NotImplementedError("reset model and remove all learnt parameters")

    def loss(self, y_pred, y_true, theta):
        raise NotImplementedError("implement loss function")


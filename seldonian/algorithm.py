from abc import ABC, abstractmethod


class SeldonianAlgorithm(ABC):
    """
    Abstract class which represents the basic functions of a Seldonian Algorithm.
    This class can be considered as a starting point for implementing your own Seldonian algorithm.

    Read more about the Seldonian Approach in
    `Preventing undesirable behavior of intelligent machines <https://doi.org/10.1126/science.aag3311>`_
    """

    @abstractmethod
    def fit(self, **kwargs):
        """
        Abstract method that is used to train the model. Also, this is the **candidate selection**
        part of the Seldonian Algorithm.

        :param kwargs: key value arguments sent to the fit function
        :return:
        """
        raise NotImplementedError("bounds function must be implemented")
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the output of the model on the the input X.

        :param X: input data to be predicted by the model.
        :return: output predictions for each sample in the input X
        """
        raise NotImplementedError("predict function must be implemented")
        pass

    @abstractmethod
    def _safetyTest(self, **kwargs):
        """
        Run the safety test on the trained model from the candidate selection part i.e. the :func:`fit` function. It is also used to predict the :math:`g(\\theta)` value used in candidate selection.

        :param kwargs Key value arguments sent to the subclass implementation of safety test.
        :return Depending on the implementation, it will either return `0` if it passes or
        `1` if it doesn't. Or, it will also return the :math:`g(\\theta)` value
        if it does not pass the safety test. Use the ``safetyTest()`` method to get a boolean value.
        """
        raise NotImplementedError("_safetyTest function must be implemented")
        pass

    def safetyTest(self, **kwargs):
        """
        A wrapper for the ``_safetyTest`` method that return a ``Boolean`` indicating whether the
        model passed the safety test.

        :param kwargs: Key-value arguments that is passed directly to ``_safetyTest``.
        :return:
        - ``True`` if model passed the safety test.
        - ``False`` if the model fails the safety test.
        """
        return self._safetyTest(**kwargs) > 0

    @abstractmethod
    def data(self):
        """
        Access the training data used by the model.

        :return: Tuple (Training data, labels)
        """
        raise NotImplementedError("subclass must implement data method to return the data")
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

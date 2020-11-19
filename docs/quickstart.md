# Quickstart

## Model class creation
Create a subclass of `seldonian.algorithm.SeldonianAlgorithm` class. 
```python
from seldonian.algorithm import *
class ExampleSeldonianModel(SeldonianAlgorithm):
    def __init__(self, *params, **kwargs ): 
        example_model = Model()
        #initialize all the model parameters
        pass

```

Now that we have a basic model setup, we need to implement the abstract method of `SeldonianAlgorithm` class. 

- `predict` - This is a basic prediction method that uses the _current_ model parameters to predict the output targets.
```python
from seldonian.algorithm import *
class ExampleSeldonianModel(SeldonianAlgorithm):
    def __init__(self, *params, **kwargs ): 
        self.example_model = Model()
        #initialize all the model parameters
        pass
    def predict(self, X, **kwargs):
        # prediction based on teh model
        return self.example_model.predict(X)
```

- `data` returns the complete data and targets as a tuple back. This includes the safety as well as the candidate data. 
```python
from seldonian.algorithm import *
class ExampleSeldonianModel(SeldonianAlgorithm):
    def __init__(self, *params, **kwargs ): 
        self.example_model = Model()
        #initialize all the model parameters
        pass
    def predict(self, X, **kwargs):
        # prediction based on teh model
        return self.example_model.predict(X)
    def data(self):
        return X, y
```

- `fit` trains the model with the constraints. 
```python
from seldonian.algorithm import *
class ExampleSeldonianModel(SeldonianAlgorithm):
    def __init__(self, *params, **kwargs ): 
        self.example_model = Model()
        #initialize all the model parameters
        pass
    def predict(self, X, **kwargs):
        # prediction based on teh model
        return self.example_model.predict(X)
    def data(self):
        return self.X, self.y
    def fit(self, *args, **kwargs):
        # fit model based under the constraint that g >0. 
        pass
```
There are various examples of such constraint optimization problems implemented like the Lagrangian 2 player game as implemented in the `VanillaNN` class.  

Or using a barrier when optimizing using a Black box optimization technique like `CMA-ES` or `scipy.optimize.minimize` class. You can find them under the `seldonian.seldonian` package.  

- `safetyTest` performs a the safety test using the safety set, or predicts the upper bound of the constraint `g(theta)` during candidate selection (or in this case, `fit`).
```python
from seldonian.algorithm import *
class ExampleSeldonianModel(SeldonianAlgorithm):
    def __init__(self, *params, **kwargs ): 
        self.example_model = Model()
        #initialize all the model parameters
        pass
    def predict(self, X, **kwargs):
        # prediction based on teh model
        return self.example_model.predict(X)
    def data(self):
        return self.X, self.y
    def fit(self, *args, **kwargs):
        # fit model based under the constraint that g >0. 
        pass
    def safetyTest(self, predict, **kwargs):
        if predict:
            # predict the upper bound during candidate selection
            return 1 if passed_is_predicted else 0 
            pass
        else:
            # run the actual safety test
            return 1 if passed else 0
            pass
        pass
```

## Training
This is _all_ you need to implement a Seldonian model. You also need some constraints that are basically function callables. Some examples of such constraints is present in the `seldonian.objectives` package. A sample run would look something like this - 
```python

constraints = [constraint1, constraint2,...] #list of function callables
seldonian_model = ExampleSeldonianModel(constriants, data, other_args)
X, y = data
seldonian_model.fit(X, y)
return seldonian_model if seldonian_model._safetyTest() else NSF # No solution found
# we now have a trained model you can now do your predictions on this model
```  

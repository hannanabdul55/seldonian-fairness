# FairML library 
An easy to use Python Library to train and develop new Machine Learning models within some fairness constraints. This is an implementation of [this Science](https://aisafety.cs.umass.edu/paper.html) paper.  

# Installation
Currently, you can install the library only from source using `pip`: 
```bash
pip install https://github.com/hannanabdul55/seldonian-fairness/archive/master.zip
```
# Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannanabdul55/seldonian-fairness/blob/master/logistic_regression_seldonian.ipynb)  Use this notebook as a reference to train a basic Logistic Regression Model.

 A quickstart tutorial on how to get quickly get started with developing your own model is present [here](http://abdulhannan.in/seldonian-fairness/quickstart.html).
Alternatively, you could use the [`LogisticRegressionSeldonianModel`](http://abdulhannan.in/seldonian-fairness/reference.html#seldonian.seldonian.LogisticRegressionSeldonianModel) to train a Logistic Regression model with any [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method by specifying it when calling the `fit` method.  

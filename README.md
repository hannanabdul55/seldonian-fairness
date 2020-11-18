# FairML library 
An easy to use Python Library to train and develop new Machine Learning models within some fairness constraints. This is an implementation of [this Science](https://aisafety.cs.umass.edu/paper.html) paper.  

# Installation
Currently, you can install the library only from source using `pip`: 
```bash
pip install https://github.com/hannanabdul55/seldonian-fairness/archive/master.zip
```
# [WIP] How to use
 A quickstart tutorial on how to get quickly get started with developing your own model is present [here](http://abdulhannan.in/seldonian-fairness/quickstart.html).
Alternatively, you could use the [`LogisticRegressionSeldonianModel`](http://abdulhannan.in/seldonian-fairness/reference.html#seldonian.seldonian.LogisticRegressionSeldonianModel) to train a Logistic Regression model with any [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method by specifying it when calling the `fit` method.  

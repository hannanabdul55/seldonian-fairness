[![Documentation Status](https://readthedocs.org/projects/seldonian-fairml/badge/?version=latest)](https://seldonian-fairml.readthedocs.io/en/latest/?badge=latest)  

Example notebook: [![Open example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannanabdul55/seldonian-fairness/blob/master/logistic_regression_seldonian.ipynb)
# FairML library 
An easy to use Python Library to train and develop new Machine Learning models within some fairness constraints. This is an implementation of [this Science](https://aisafety.cs.umass.edu/paper.html) paper.   
Also includes some other handy tools like: 
- Bound propogation using the `RandomVariable` object. 
- _[Documentation WIP]_ Python implementation of the [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) black-box optimization algorithm. You can refer to the details [here](http://abdulhannan.in/seldonian-fairness/reference.html#module-seldonian.cmaes) and the implementation [here](https://github.com/hannanabdul55/seldonian-fairness/blob/master/seldonian/cmaes.py#L11)

# Installation
Currently, you can install the library only from source using `pip`: 
```bash
pip install https://github.com/hannanabdul55/seldonian-fairness/archive/master.zip
```

# Development setup
This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`).
```bash
uv sync                    # create .venv and install core + dev dependencies
uv run pytest tests/ -q    # run the test suite
uv run ruff check seldonian/ tests/   # lint
```
Optional extras:
```bash
uv sync --extra ray        # ray-based multiprocessing for RL experiments
uv sync --extra datasets   # shap + tempeh (installed from GitHub; removed from PyPI)
uv sync --extra docs       # sphinx documentation toolchain
```
# Usage
[![Open example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannanabdul55/seldonian-fairness/blob/master/logistic_regression_seldonian.ipynb) Use this notebook as a reference to train a basic Logistic Regression Model.  
 A quickstart tutorial on how to get quickly get started with developing your own model is present [here](http://abdulhannan.in/seldonian-fairness/quickstart.html).
Alternatively, you could use the [`LogisticRegressionSeldonianModel`](http://abdulhannan.in/seldonian-fairness/reference.html#seldonian.seldonian.LogisticRegressionSeldonianModel) to train a Logistic Regression model with any [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method by specifying it when calling the `fit` method.  

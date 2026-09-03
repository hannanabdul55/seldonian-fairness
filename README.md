[![Documentation Status](https://readthedocs.org/projects/seldonian-fairml/badge/?version=latest)](https://seldonian-fairml.readthedocs.io/en/latest/?badge=latest)  

Example notebook: [![Open example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannanabdul55/seldonian-fairness/blob/master/logistic_regression_seldonian.ipynb)
# FairML library 
An easy to use Python Library to train and develop new Machine Learning models within some fairness constraints. This is an implementation of [this Science](https://aisafety.cs.umass.edu/paper.html) paper.   
Also includes some other handy tools like: 
- Bound propogation using the `RandomVariable` object. 
- A family of non-asymptotic confidence bounds for the mean of a bounded random variable (`seldonian.bounds`: Clopper-Pearson, Bentkus, empirical Bernstein, Anderson, betting-based bounds, and a two-sample Bentkus bound for rate differences), selectable per constraint via `ghat_tpr_diff(A_idx, method='bentkus_diff')`. See [`reports/bounds/README.md`](reports/bounds/README.md) for the study and the exact-enumeration validation harness (`uv run python scripts/validate_bounds.py`).
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
uv sync --extra rl         # trl + peft + transformers for LLM post-training (seldonian.llm)
uv sync --extra datasets   # shap + tempeh (installed from GitHub; removed from PyPI)
uv sync --extra docs       # sphinx documentation toolchain
```

# Seldonian RL post-training for LLMs
`seldonian.llm` applies the candidate-selection / safety-test split to GRPO fine-tuning of an
instruct model. Prompts are split into a candidate set and a sealed safety set; training runs on
the candidate set with a *predicted* safety test every few steps, and the selected checkpoint is
then tested exactly once on the safety set with a high-confidence bound on a judge-labelled
constraint (harm rate, refusal rate, error rate). `None` is returned when it fails.
Default judges are ungated (`Qwen/Qwen3Guard-Gen-0.6B` for harm, a DistilRoBERTa refusal classifier);
Llama Guard is available with `--harm-judge llama_guard` once you have Hub access.
```bash
uv sync --extra rl
uv run scripts/run_llm_rl.py --task ab --method seldonian --seed 0      # Tasks A+B, Seldonian
uv run scripts/run_llm_rl.py --task ab --method composite --lam 1.0     # composite-reward baseline
uv run scripts/run_llm_rl.py --task gsm8k --method grpo                 # verifiable control task
```
Results land in `results/llm/<task>/<method>/seed<k>/result.json` next to the prompt splits and the
safety-set episodes. The safety logic in `seldonian.llm.policy` has no model dependency and is
covered by `tests/test_llm_policy.py` with a mock backend.
# Usage
[![Open example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hannanabdul55/seldonian-fairness/blob/master/logistic_regression_seldonian.ipynb) Use this notebook as a reference to train a basic Logistic Regression Model.  
 A quickstart tutorial on how to get quickly get started with developing your own model is present [here](http://abdulhannan.in/seldonian-fairness/quickstart.html).
Alternatively, you could use the [`LogisticRegressionSeldonianModel`](http://abdulhannan.in/seldonian-fairness/reference.html#seldonian.seldonian.LogisticRegressionSeldonianModel) to train a Logistic Regression model with any [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) method by specifying it when calling the `fit` method.  

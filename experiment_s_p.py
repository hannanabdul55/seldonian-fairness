import errno

from synthetic import *
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import os
from time import time

import argparse
import json
import ray
import copy
import timeit

from memory_profiler import profile

parser = argparse.ArgumentParser(description='Process some config values')
parser.add_argument("--config")
parser.add_argument("checkpoint", nargs='?', default='results-' + str(time()))
parser.add_argument("--threads")
parser.add_argument("--dir")

args = parser.parse_args()

if args.dir:
    dir = args.dir
else:
    dir = 'results_default'

if args.config:
    exp_config = json.load(open(args.config, "r"))
else:
    exp_config = {'N': np.geomspace(1e2, 1e6, 20).astype(np.int), 'trials': 40,
                  'methods': ['ttest', 'hoeffdings'], 'D': 10, 'tprs': [0.3, 0.8],
                  'test_size': 0.4, 'opt': 'Powell'}

uc_result = []
results = {}

if args.checkpoint is None:
    checkpoint = str(np.random.randint(1e4))
else:
    checkpoint = args.checkpoint


def save_res(obj, filename=f"./{dir}/{checkpoint}_{np.random.randint(1000000)}.p"):
    pickle.dump(obj, open(filename, 'wb'))


@ray.remote
def run_experiment_p(exp):
    print(f"Running experiment for exp = {exp!r}")
    n = exp['N']
    opt = exp['opt']
    X, y, A_idx = make_synthetic(n, exp['D'], *exp['tprs'])
    X_test, y_test, _ = make_synthetic(n, exp['D'], *exp['tprs'], A_idx=A_idx)
    results = {'N': n, 'opt': opt}
    failure_rate = []
    sol_found_rate = []
    accuracy = []
    mean_ghat = []
    uc_failure_rate = []
    uc_accuracy = []
    uc_mean_ghat = []
    for t in np.arange(exp['trials']):
        ghats = []
        ghats.append({
            'fn': ghat_tpr_diff(A_idx,
                                threshold=abs(exp['tprs'][0] - exp['tprs'][1]) / 2),
            'delta': 0.05
        })
        if opt == 'CMAES':
            est = SeldonianAlgorithmLogRegCMAES(X, y, test_size=exp['test_size'],
                                                g_hats=ghats,
                                                verbose=False)
        else:
            est = LogisticRegressionSeldonianModel(X, y, test_size=exp['test_size'],
                                                   g_hats=ghats,
                                                   verbose=True)
        est.fit()

        # Accuracy on seldonian optimizer
        y_p = est.predict(X_test)
        acc = accuracy_score(y_test, y_p)
        accuracy.append(acc)

        # safety test on seldonian model
        safe = est.safetyTest(predict=False)

        # Rate Solution Found
        sol_found_rate.append(0 if safe > 0 else 1)

        c_ghat_val = ghats[0]['fn'](X_test, y_test, y_p, ghats[0]['delta'], ub=False)

        # mean value of ghat
        mean_ghat.append(c_ghat_val)

        # Probability of g(D)<0
        failure_rate.append(1 if c_ghat_val > 0.0 else 0)

        # Unconstrained optimizer
        uc_est = LogisticRegression(penalty='none').fit(X, y)
        y_preds = uc_est.predict(X_test)

        # Accuracy on Unconstrained estimator
        uc_acc = accuracy_score(y_test, y_preds)
        uc_accuracy.append(uc_acc)

        # Mean ghat valuye on test data
        ghat_val = ghats[0]['fn'](X_test, y_test, y_preds, ghats[0]['delta'], ub=False)
        uc_mean_ghat.append(ghat_val)

        # Failure rate on Unconstrained estimator
        uc_failure_rate.append(1 if ghat_val > 0.0 else 0)

        results.update({
            'n': n,
            'trials': t,
            'failure_rate': np.mean(failure_rate),
            'failure_rate_std': np.std(failure_rate),
            'sol_found_rate': np.mean(sol_found_rate),
            'sol_found_rate_std': np.std(sol_found_rate),
            'accuracy': np.mean(accuracy),
            'ghat': np.mean(mean_ghat),
            'uc_failure_rate': np.mean(uc_failure_rate),
            'uc_failure_rate_std': np.std(uc_failure_rate),
            'uc_accuracy': np.mean(uc_accuracy),
            'uc_ghat': np.mean(uc_mean_ghat)
        })
        save_res(results, filename=f"{dir}/{checkpoint}_{n}.p")
    print(f"Results for N={n}: \n{results!r}")
    save_res(results, filename=f"{dir}/{checkpoint}_{n}.p")
    return results


if __name__ == '__main__':
    print(f"Running experiment with checkpoint: {checkpoint}")
    print(f"Config used: {exp_config!r}")
    if 'name' in exp_config:
        dir = f"result_{exp_config['name']}"
    os.makedirs(dir, exist_ok=True)

    ray.init()
    pickle.dump(exp_config, open(dir + "/config.p", "wb"))

    exps = []
    for n in exp_config['N']:
        exps.append(copy.deepcopy(exp_config))
        exps[-1]['N'] = n

    pickle.dump(exps, open(f"{dir}/exps.p", "wb"))
    a = time()
    futures = [run_experiment_p.remote(x) for x in exps]
    res = ray.get(futures)
    b = time()
    print("saving results")
    save_res(res, f"./{dir}/final_res_{checkpoint}.p")

    print(f"Time run: {int(b-a)} seconds")

    # run_experiment(exp_config)

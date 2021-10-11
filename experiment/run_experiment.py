from seldonian.seldonian import *
from seldonian.seldonian import VanillaNN
from seldonian.synthetic import *
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import os
from time import time

import ray

import argparse
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--result-dir", required=True)
# parser.add_argument("--dir")
parser.add_argument("--strat", type=str, default='true')
parser.add_argument("--worker-id", required=True, type=int)
parser.add_argument("--dims", default=5, type=int)
parser.add_argument("--min", default=100, type=int)
parser.add_argument("--max", default=1e6, type=int)
parser.add_argument("--num", default=50, type=int)
parser.add_argument("--tpr1", default=0.1)
parser.add_argument("--tpr2", default=0.9)
parser.add_argument("--num-test", default=100000, type=int)
parser.add_argument("--num-test-ratio", default=0.4, type=float)

args_m = parser.parse_args()

# if args_m.dir:
#     dir = args_m.dir
# else:
#     dir = 'result/results_default'

if not os.path.exists(args_m.result_dir):
    os.makedirs(args_m.result_dir, exist_ok=True)

print(f"running experiment on worker {args_m.worker_id}")
np.random.seed(args_m.worker_id)
@ray.remote
def run_experiment(args, n):
    print(f"Running experiment for workerid {args.worker_id} and n={n}")
    a_c = time()
    # opt = 'CMAES'
    X, y, A_idx = make_synthetic(n, args.dims, args.tpr1, args.tpr2)
    X_test, y_test, _ = make_synthetic(args.num_test, args.dims, args.tpr1, args.tpr2, A_idx=A_idx)
    thres = abs(args.tpr1 - args.tpr2) / 2
    # results = {'N': n, 'opt': opt}
    # failure_rate = []
    # sol_found_rate = []
    # accuracy = []
    # mean_ghat = []
    # uc_failure_rate = []
    # uc_accuracy = []
    # uc_mean_ghat = []
    res = {
        'n': n
    }
    ghats = [{
        'fn': ghat_tpr_diff(A_idx,
                            threshold=thres,
                            method='ttest'),
        'delta': 0.05
    }]

    est = SeldonianAlgorithmLogRegCMAES(X, y, test_size=args.num_test_ratio,
                                        g_hats=ghats,
                                        verbose=False, stratify=args.strat.lower() == 'true')
    est.fit()

    # Accuracy on seldonian optimizer
    y_p = est.predict(X_test)
    acc = accuracy_score(y_test.astype(int), y_p)
    res['acc'] = acc

    # safety test on seldonian model
    safe = est._safetyTest(predict=False)

    # Rate Solution Found
    res['sol_found'] = 0 if safe > 0 else 1

    c_ghat_val = ghats[0]['fn'](X_test, y_test, y_p, ghats[0]['delta'], ub=False)

    # mean value of ghat
    res['ghat'] = c_ghat_val

    # Probability of g(D)<0
    res['fr'] = 1 if c_ghat_val > 0.0 else 0

    # Unconstrained optimizer
    uc_est = LogisticRegression(penalty='none').fit(X, y)

    y_preds = uc_est.predict(X_test)
    # Accuracy on Unconstrained estimator
    uc_acc = accuracy_score(y_test, y_preds)
    res['uc_acc'] = uc_acc

    # Mean ghat value on test data
    ghat_val = ghats[0]['fn'](X_test, y_test, y_preds, ghats[0]['delta'], ub=False)
    res['uc_ghat'] = ghat_val

    # Failure rate on Unconstrained estimator
    res['uc_fr'] = 1 if ghat_val > 0.0 else 0
    b_c = time()
    print(f"completed run in {int(b_c-a_c)} seconds.")
    return (
        res['n'], res['acc'],
        res['sol_found'], res['ghat'],
        res['fr'], res['uc_acc'],
        res['uc_ghat'], res['uc_fr']
    )

if __name__ == "__main__":
    ns = np.geomspace(args_m.min, args_m.max, args_m.num, dtype=int)
    ray.init()

    # dump arguments to config json
    json.dump(vars(args_m), open(f"{args_m.result_dir}/config.json", "w"))
    a = time()
    futures = [run_experiment.remote(args_m, n) for n in ns]
    res = ray.get(futures)
    b = time()
    print(f"Ran computation in {int(b-a)} seconds")
    pickle.dump(res, open(f"{args_m.result_dir}/results{args_m.worker_id}.p", "wb"))



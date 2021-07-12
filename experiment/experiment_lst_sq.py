import sklearn
from sklearn import preprocessing

from seldonian.datasets import LawschoolDataset, AdultDataset
from seldonian.seldonian import *
from seldonian.synthetic import *

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from tempeh.configurations import datasets

import pickle
import os
from time import time

import argparse
import json
import ray
import copy

import torch

has_gpu = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Process some config values')
parser.add_argument("--config")
parser.add_argument("checkpoint", nargs='?', default='results-' + str(time()))
parser.add_argument("--threads")
parser.add_argument("--dir")
parser.add_argument("--gpus")
parser.add_argument("--workers")
parser.add_argument("--local")

args = parser.parse_args()
if args.local:
    use_local = True
else:
    use_local = False
if args.dir:
    dir = args.dir
else:
    dir = 'results_default'

kwargs = {
    'local_mode': use_local
}

exp_kwargs = {

}
if args.gpus:
    n_gpus = int(args.gpus)
    kwargs['num_gpus'] = n_gpus
    exp_kwargs['num_gpus'] = 1
else:
    exp_kwargs['num_gpus'] = 0
    n_gpus = 0

if args.workers:
    workers = int(args.workers)
else:
    workers = 5

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


@ray.remote(**exp_kwargs)
def run_experiment_p(exp):
    gpu_ids = ray.get_gpu_ids()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    if len(gpu_ids) > 0:
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        gpu_id = gpu_ids[0]
        print(f"Using GPU: {gpu_id}")
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print(f"Running experiment for exp = {exp!r}")
        print(f"Running experiment on {device}")
    stratify = False
    if 'stratify' in exp:
        stratify = exp['stratify']
    if 'thetas' in exp:
        thetas = exp['thetas']
    else:
        thetas = 1
    if 'method' not in exp:
        exp['method'] = 'ttest'

    if 'data' in exp:
        data = exp['data']
    else:
        data = 'synthetic'

    if 'agg' in exp:
        agg_fn = exp['agg']
    else:
        agg_fn = 'min'
    n = exp['N']
    opt = exp['opt']
    if data == 'synthetic':
        if 'a_prob' in exp:
            a_prob = exp['a_prob']
        else:
            a_prob = 0.5
        if 'seed' in exp:
            seed = exp['seed']
        else:
            seed = 0
        X, y, A_idx = make_synthetic(n, exp['D'], *exp['tprs'], A_prob=a_prob, seed=seed)
        X_test, y_test, _ = make_synthetic(n_test, exp['D'], *exp['tprs'], A_idx=A_idx,
                                           A_prob=a_prob, seed=seed)
    elif data=='synthetic_regression':
        
        noise = exp['noise'] if 'noise' in exp else 1.0

        seed = exp['seed'] if 'seed' in exp else 0

        n_useful = exp['n_useful'] if 'n_useful' in exp else max(int(exp['D']/4), 2)

        X, y = regression_make_synthetic(n, exp['D'], num_useful=n_useful, noise=noise)

        X_test, y_test = regression_make_synthetic(n_test, exp['D'], num_useful=n_useful, noise=noise*1.25)

    elif data == 'lawschool':
        X, X_test, y, y_test, A, A_idx = LawschoolDataset(n=int(n), verbose=True).get_data()
    else:
        X, X_test, y, y_test, A, A_idx = AdultDataset(n=int(n), verbose=True).get_data()

    if "thresh" not in exp:
        thres = noise**1.5
    else:
        thres = exp['thresh']


    results = {'N': n, 'opt': opt}
    failure_rate = []
    sol_found_rate = []
    accuracy = []
    mean_ghat = []
    uc_failure_rate = []
    uc_accuracy = []
    uc_mean_ghat = []
    for t in np.arange(exp['trials']):
        ghats = [{
            'fn': ghat_regression_thres(threshold=thres,
                                method=exp['method']),
            'delta': 0.05
        }]
        if opt == 'CMAES':
            if 'hard_barrier' in exp:
                hard_barrier = exp['hard_barrier']
            else:
                hard_barrier = False
            est = LinearRegressionSeldonianModel(X, y, test_size=exp['test_size'],
                                                g_hats=ghats,
                                                hard_barrier=hard_barrier,
                                                verbose=False, stratify=stratify, random_seed=t,
                                                nthetas=thetas,
                                                agg_fn=agg_fn
                                                )
        elif opt == 'Powell':
            if 'hard_barrier' in exp:
                hard_barrier = exp['hard_barrier']
                print(f"Running with hard_barrier={hard_barrier}")
            else:
                hard_barrier = False
            est = LogisticRegressionSeldonianModel(X, y, test_size=exp['test_size'],
                                                   g_hats=ghats,
                                                   verbose=False,
                                                   hard_barrier=hard_barrier,
                                                   stratify=stratify,
                                                   random_seed=t,
                                                   nthetas=thetas,
                                                   agg_fn=agg_fn)
        else:
            ghats = [{
                'fn': ghat_tpr_diff_t(A_idx,
                                      threshold=thres,
                                      method=exp['method']),
                'delta': 0.05
            }]
            est = VanillaNN(X, y, test_size=exp['test_size'], g_hats=ghats, stratify=stratify)

        est.fit()

        # Accuracy on seldonian optimizer
        y_p = est.predict(X_test)

        acc = mean_squared_error(y_test, y_p)
        accuracy.append(acc)

        # safety test on seldonian model
        safe = est._safetyTest(predict=False)

        # Rate Solution Found
        sol_found_rate.append(1 if safe <= 0 else 0)

        # ghats = [{
        #     'fn': ghat_tpr_diff(A_idx,
        #                         threshold=thres,
        #                         method=exp['method']),
        #     'delta': 0.05
        # }]

        c_ghat_val = ghats[0]['fn'](X_test, y_test, y_p, ghats[0]['delta'])

        # mean value of ghat
        mean_ghat.append(c_ghat_val)

        # Probability of g(D)<0
        failure_rate.append(1 if c_ghat_val > 0.0 and safe <= 0.0 else 0)

        # Unconstrained optimizer
        if opt != 'NN':
            uc_est = LinearRegression(random_state=t).fit(X, y)
        else:
            uc_est = VanillaNN(X, y, test_size=exp['test_size'], stratify=stratify)
            uc_est.fit()

        y_preds = uc_est.predict(X_test)
        
        # Accuracy on Unconstrained estimator
        uc_acc = mean_squared_error(y_test, y_preds)
        uc_accuracy.append(uc_acc)

        # Mean ghat value on test data
        ghat_val = ghats[0]['fn'](X_test, y_test, y_preds, ghats[0]['delta'])
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
            'mse': np.mean(accuracy),
            'ghat': np.mean(mean_ghat),
            'uc_failure_rate': np.mean(uc_failure_rate),
            'uc_failure_rate_std': np.std(uc_failure_rate),
            'uc_mse': np.mean(uc_accuracy),
            'uc_ghat': np.mean(uc_mean_ghat),
            'method': exp['method']
        })
        save_res(results, filename=f"{dir}/{checkpoint}_{n}.p")
    print(f"Results for N={n}: \n{results!r}")
    save_res(results, filename=f"{dir}/{checkpoint}_{n}.p")
    return results


if __name__ == '__main__':
    print(f"Running experiment with checkpoint: {checkpoint}")
    print(f"Config used: {exp_config!r}")
    if 'name' in exp_config:
        dir = f"result/result_{exp_config['name']}"
    os.makedirs(dir, exist_ok=True)
    n_test = 1e6 * 6
    print(has_gpu)
    if has_gpu:
        print(f"Initializing ray with {n_gpus} GPUs and {kwargs} parameters")
        print('Available devices ', torch.cuda.device_count())
        ray.init(**kwargs)
    else:
        ray.init(**kwargs)
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

    print(f"Time run: {int(b - a)} seconds")

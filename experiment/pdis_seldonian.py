# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import os
from time import time
import json
import pickle
import ray

import argparse

import logging
from scipy.optimize import minimize
from scipy.special import softmax

from seldonian.seldonian import PDISSeldonianPolicyCMAES, SeldonianCEMPDISPolicy


def run_model(method='CMAES', index=0):
    if method.lower() == 'cmaes':
        model = PDISSeldonianPolicyCMAES(data, S, A, gamma=0.95)
    else:
        model = SeldonianCEMPDISPolicy(data, S, A, gamma=0.95, use_ray=True)
    print(f"Using {method} model")
    # print(f"Running minimization over {n - test} episodes")
    b = time()
    model.fit()
    print(f"Result: {model.theta}")
    print(f"time taken: {time() - b} seconds")
    os.makedirs(f"results/{checkpoint}", exist_ok=True)
    os.makedirs(f"results/{checkpoint}/policy", exist_ok=True)
    # pickle.dump(model.theta, open(f"results/{checkpoint}/res_{time() % 10000}.p", "wb"))
    np.savetxt(f"results/{checkpoint}/policy/policy{index}.txt", model.theta)
    results = {'safety_test': model._safetyTest(model.theta),
               'method': method,
               'safety_test_size': model.D_s.shape[0],
               'candidate_estimate': model._predict(model.D_c, model.theta)}
    print("results: ", results)
    json.dump(results, open(f"results/{checkpoint}/policy/results{index}.json", "w"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some config values')
    parser.add_argument("--checkpoint", nargs='?',
                        default='results-' + str(np.random.randint(1, 1e3)))
    parser.add_argument("--index")
    parser.add_argument("--local")
    parser.add_argument("--method")

    args = parser.parse_args()

    if args.local:
        local = args.local
    else:
        local = False
    if args.method:
        method = args.method
    else:
        method = 'CMAES'
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = np.random.randint(1, 1e3)

    if args.index:
        index = int(args.index)
    else:
        index = 0
    ray.init(local_mode=local)
    print("Loading data")
    a = time()
    num_times = 50
    data = np.load("data_np.npy", allow_pickle=True)
    # data = data[:50000]
    print(f"Loaded data in {time() - a} seconds")
    n = len(data)
    print(f"number of episodes: {n}")
    test = int(0.5 * n)
    # h_te = data[-test:]
    # h_tr = data[:-test]
    S = 18
    A = 4

    remotes = [run_model(method=method, index=i) for i in range(num_times)]
    print(f"Done in {time()-1} seconds")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

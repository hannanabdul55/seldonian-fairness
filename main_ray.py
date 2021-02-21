# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import os
from time import time
import pickle
import ray

from scipy.optimize import minimize
from scipy.special import softmax


def get_data(raw):
    eps = int(raw.readline().strip())
    data = []
    for ep in range(eps):
        ep_data = []
        T = int(raw.readline().strip())
        for i in range(T):
            ep_data.append(list(map(float, raw.readline().strip().split(','))))
        data.append(ep_data)
    return data

def estimate(pi_e, D, n, gamma=0.95):
    est = 0.0
    print(f"Starting PDIS estimation for {n} samples")
    for ep in D:
        w_t = 1
        gamma_t = 1
        for t in range(len(ep)):
            s_t, a_t, r_t, pi_b = ep[t]
            s_t = int(s_t)
            a_t = int(a_t)
            w_t *= softmax(pi_e[s_t])[a_t] / pi_b
            est += gamma_t * w_t * r_t / n
            gamma_t *= gamma
    print(f"Average estimate of return: {est}")
    return est

@ray.remote
def estimate_ray(pi_e, D, n, gamma=0.95):
    est = 0.0
    print(f"Starting PDIS estimation for {n} samples")
    for ep in D:
        w_t = 1
        gamma_t = 1
        for t in range(len(ep)):
            s_t, a_t, r_t, pi_b = ep[t]
            s_t = int(s_t)
            a_t = int(a_t)
            w_t *= softmax(pi_e[s_t])[a_t] / pi_b
            est += gamma_t * w_t * r_t / n
            gamma_t *= gamma
    print(f"Average estimate of return: {est}")
    return est


@ray.remote
def estimate_ray_vec(pi_e, D, n, gamma=0.95):
    est = 0.0
    pi_e = softmax(pi_e, axis=1)
    # print(f"Starting PDIS estimation for {n} samples")
    for ep in D:
        ep = np.array(ep, dtype=np.float)
        weights = np.cumprod(
            pi_e[ep[:, 0].astype(np.int), ep[:, 1].astype(np.int)] * gamma / ep[:,
                                                                                   3]) / gamma

        est += weights.dot(ep[:, 2])

    # print(f"Average estimate of return: {est/n}")
    return est/n

def estimate_vec(pi_e, D, n, gamma=0.95):
    est = 0.0
    pi_e = softmax(pi_e, axis=1)
    # print(f"Policy: ")
    # print(pi_e)
    # print(f"Starting PDIS estimation for {n} samples")

    for ep in D:
        ep = np.array(ep, dtype=np.float)
        weights = np.cumprod(
            pi_e[ep[:, 0].astype(np.int), ep[:, 1].astype(np.int)] * gamma / ep[:,
                                                                                   3]) / gamma
        est += weights.dot(ep[:, 2])
    return est/n


def pdis_estimate(pi_e, D, S, A, gamma=0.95, minimize=True, verbose=False):
    if D is None:
        raise ValueError("Data D is None")
    n = len(D)
    if verbose:
        print(f"Running PDIS estimation for the entire candidate data of {len(D)} samples")
    a = time()
    pi_e = pi_e.reshape(S, A)
    # est = 0.0
    # R = []
    n_work = 12
    idx = 0
    works = []
    for i in range(n_work):
        start = int(n * i / n_work)
        end = int(n * (i + 1) / n_work)
        works.append(estimate_ray_vec.remote(pi_e, D[start:end], n, gamma))
    results = ray.get(works)
    if verbose:
        print(f"Estimation for one complete run done in {time() - a} seconds")
    est = sum(results)
    # for ep in D:
    #     w_t = 1
    #     gamma_t = 1
    #     for t in range(len(ep)):
    #         s_t, a_t, r_t, pi_b = ep[t]
    #         s_t = int(s_t)
    #         a_t = int(a_t)
    #         w_t *= softmax(pi_e[s_t])[a_t] / pi_b
    #         est += gamma_t * w_t * r_t / n
    #         gamma_t *= gamma
    print(f"Average estimate of return: {est}")
    return est * (-1 if minimize else 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__1':
    # data_raw = open("sample.txt", 'r')
    data_raw = open("data.csv", 'r')
    data = get_data(data_raw)
    print(len(data))
    pickle.dump(data, open("data.p", 'wb'))

if __name__ == '__main__2':
    print("Loading data")
    a = time()
    data = pickle.load(open("data.p", "rb"))
    print(f"Loaded data in {time() - a} seconds")
    n = len(data)
    print(f"number of episodes: {n}")
    test = int(0.5 * n)
    h_te = data[-test:]
    h_tr = data[:-test]
    S = 18
    A = 4
    t_0 = np.random.randn(S * A)
    print(f"Running minimization over {n - test} episodes")
    a = time()
    pi_s = minimize(pdis_estimate, t_0, args=(h_tr, S, A), method='Powell')
    print(f"Result: {pi_s}")
    print(f"time taken: {time() - a} seconds")
    pickle.dump(pi_s, open("results/res.p", "w"))

if __name__ == '__main__':
    ray.init()
    print("Loading data")
    a = time()
    data = np.load("data_np.npy", allow_pickle=True)
    print(f"Loaded data in {time() - a} seconds")
    n = len(data)
    print(f"number of episodes: {n}")
    test = int(0.5 * n)
    h_te = data[-test:]
    h_tr = data[:-test]
    S = 18
    A = 4
    t_0 = np.random.randn(S * A)
    print(f"Running minimization over {n - test} episodes")
    a = time()
    pi_s = minimize(pdis_estimate, t_0, args=(h_tr, S, A), method='Powell')
    print(f"Result: {pi_s}")
    print(f"time taken: {time() - a} seconds")
    pickle.dump(pi_s, open("results/res.p", "w"))

if __name__ == '__main__1':
    ray.init()
    print("Loading data")
    a = time()
    data = pickle.load(open("data.p", "rb"))
    print(f"Loaded data in {time() - a} seconds")
    n = len(data)
    print(f"number of episodes: {n}")
    test = int(0.5 * n)
    S = 18
    A = 4
    t_0 = np.ones(S * A)
    print(f"Estimate: {pdis_estimate(t_0, data, S, A)}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

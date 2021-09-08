from seldonian.rl.rl_utils import run_episodes
from numpy.core.function_base import geomspace
from ray.tune import analysis
from actor_critic import *

import pickle

import os
# import ray
# from ray import tune
import json
import argparse

from time import time

from numba import typed, typeof, njit, int64, types
from numba.experimental import jitclass
from numba.extending import overload_method, overload

# hyperparams from the HCGA paper
ac_hparams = {
    'alpha_actor': 0.137731127022912,
    'alpha_critic': 0.31442900745165847,
    'lam': 0.23372572419318238
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workerid", required=True, help="Add worker is here")
    parser.add_argument("--out-dir", default="result",
                        help="Directory to output results")
    parser.add_argument("--num-training", default=1000, type=int)
    parser.add_argument("--num-test", default=10000, type=int)
    parser.add_argument("--alpha-actor", default=0.02, type=float)
    parser.add_argument("--alpha-critic", default=0.02, type=float)
    parser.add_argument("--lam", default=0.02, type=float)
    parser.add_argument("--num-safety", default=0.2, type=float)
    parser.add_argument("--strat", required=True)
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--j", default=-8.0, type=float)
    parser.add_argument("--num-trials", default=1000, type=int)
    parser.add_argument("--num-eps", default=1024, type=int)
    return parser.parse_args()


@njit
def train(agent, mdps, eps, seed=42, epoch=1):
    trained = 0
    mdps_idx = np.arange(len(mdps))
    while trained < eps:
        np.random.shuffle(mdps_idx)
        for i in mdps_idx:
            run_episode(agent, mdps[i].get_repr())
            trained += 1
    return trained


def main():
    args = parse_args()
    out = args.out_dir
    # make output directory
    os.makedirs(out, exist_ok=True)
    np.random.seed(42)
    # collect variables
    (
        n_tr, n_te,
        min_reward, alpha_critic,
        alpha_actor, delta,
        strat, lam, num_trials,
        eps
    ) = (
        args.num_training,
        args.num_test,
        args.j,
        args.alpha_critic,
        args.alpha_actor,
        args.delta,
        args.strat,
        args.lam,
        args.num_trials,
        args.num_eps

    )
    
    mdps_te = create_n_mdps(n_te)
    mdps_tr = create_n_mdps(n_tr, start=n_te)
    results = []
    for t in range(num_trials):
        print(f"Running trial {t+1}/{num_trials}")
        results.append(run_experiment(
            args,mdps_tr,
            mdps_te, t
            ))

    pickle.dump(results, open("results.p", "wb"))

    pass


def run_experiment(args, mdps_tr, mdps_te, t):
    (
        n_tr, n_te,
        min_reward, alpha_critic,
        alpha_actor, delta,
        strat, lam, num_trials,
        eps
    ) = (
        args.num_training,
        args.num_test,
        args.j,
        args.alpha_critic,
        args.alpha_actor,
        args.delta,
        args.strat,
        args.lam,
        args.num_trials,
        args.num_eps

    )
    a = time()
    # print(args.num_training)
    # mdps_te = create_n_mdps(n_te)
    # mdps_tr = create_n_mdps(n_tr, start=n_te)
    # print(f"Took {time()-a} seconds to create {n_te + n_tr} MDPs")
    np.random.seed(t)
    policy = TabularSoftmaxPolicy(
        a=mdps_tr[0].len_actions,
        s=mdps_tr[0].len_states,
        eps=0.0
    )
    agent = ActorCriticGridworld(
        mdps_tr[0], policy
    )

    print(f"Running training")
    a = time()
    train(agent, mdps_tr, eps)
    print(f"Finished training in {time()-a} seconds")

    print(f"Evaluation")
    a = time()
    rewards = []
    for m in mdps_te:
        rewards.append(run_episode(agent, m.get_repr(), True))
    print("Expected Discounteed reward: ", np.mean(rewards))
    print(f"Finished evaluation in {time()-a} seconds")


if __name__ == "__main__":
    main()

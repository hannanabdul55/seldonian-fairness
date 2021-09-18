from numpy.core.function_base import geomspace
from seldonian.rl.rl_utils import get_eps_for_n, run_episodes
from actor_critic import *
from bounds import *

import pickle


from memory_profiler import profile

from sklearn.model_selection import train_test_split

import os
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
    parser.add_argument("--workerid", required=True,
                        type=int, help="Add worker is here")
    parser.add_argument("--out-dir", default="result",
                        help="Directory to output results")
    # parser.add_argument("--num-training", default=1000, type=int)
    parser.add_argument("--num-test", default=10000, type=int)
    parser.add_argument("--alpha-actor", default=0.02, type=float)
    parser.add_argument("--alpha-critic", default=0.02, type=float)
    parser.add_argument("--lam", default=0.02, type=float)
    parser.add_argument("--num-safety", default=0.2, type=float)
    parser.add_argument("--strat", default=False, type=bool)
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--j", default=-8.0, type=float)
    parser.add_argument("--num-trials", default=1000, type=int)
    parser.add_argument("--num-eps", default=1024, type=int)
    parser.add_argument("--num-theta", default=10, type=int)
    parser.add_argument("--num-strat-try", default=40, type=int)
    parser.add_argument("--num-multiplier", default=50, type=int)
    parser.add_argument("--start", default=3, type=int)
    parser.add_argument("--index", required=True, type=int)
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


def main( args):
    out = args.out_dir
    # make output directory
    os.makedirs(out, exist_ok=True)
    # collect variables
    (
        n_te, n_s,
        min_reward, alpha_critic,
        alpha_actor, delta,
        strat, lam, num_trials,
        eps, workerid
    ) = (
        # args.num_training,
        args.num_test,
        args.num_safety,
        args.j,
        args.alpha_critic,
        args.alpha_actor,
        args.delta,
        args.strat,
        args.lam,
        args.num_trials,
        args.num_eps,
        args.workerid

    )
    n_tr = args.start + (args.num_multiplier*args.index)
    # config = {
    #     'n_tr': n_tr,
    #     'trials': num_trials
    # }
    config = vars(args)
    json.dump(config, open(os.path.join(out, "config.json"), "w"))
    np.random.seed(workerid + args.index)
    nums = 0
    mdps_te = create_n_mdps(n_te)
    # keep track of starting seed
    nums += n_te
    mdps_tr = create_n_mdps(n_tr, start=nums)
    # print(
    #     f"Size of train set: {len(mdps_tr)} | Size of test setL {len(mdps_s)}")
    results = []
    for t in range(num_trials):
        print(f"Running trial {t+1}/{num_trials}")
        results.append(run_experiment(
            args, mdps_tr,
            mdps_te, t,
            workerid,
            strat=strat
        ))

    pickle.dump(results, open(os.path.join(
        out, f"results{workerid}_{n_tr}.p"), "wb"))


def run_experiment(
    args, mdps_tr,
    mdps_te,
    t, workerid,
    strat,
    verbose=True
):
    (
        n_te,
        min_reward, alpha_critic,
        alpha_actor, delta,
        strat, lam, num_trials,
        eps
    ) = (
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
    n_tr = len(mdps_tr)
    res = {
        'n_tr': n_tr,
        'n_te': n_te,
        'j': min_reward,
        'delta': delta,
        'n_trials': num_trials
    }
    a = time()
    # print(args.num_training)
    # mdps_te = create_n_mdps(n_te)
    # mdps_tr = create_n_mdps(n_tr, start=n_te)
    # print(f"Took {time()-a} seconds to create {n_te + n_tr} MDPs")
    np.random.seed(workerid%10000 + t)
    if not strat:
        mdps_tr, mdps_s = train_test_split(
            mdps_tr, test_size=args.num_safety
        )
    else:
        print("Running with stratification")
        pols = []
        best_diff = np.inf
        mdps_tr_best, mdps_s_best = train_test_split(
            mdps_tr, test_size=args.num_safety
        )
        for k in range(args.num_theta):
            pols.append(TabularSoftmaxPolicy(
                a=mdps_tr[0].len_actions,
                s=mdps_tr[0].len_states,
                eps=0.0
            ))
        ests_all = []
        mtmp = get_gw_from_seed(0)  # temp environment
        print(f"Running episodes for {len(mdps_tr)} MDPs")
        a = time()
        for m in mdps_tr:
            r_t = []
            for p in pols:
                ag = ActorCriticGridworld(
                    mdps_tr[0], p
                )
                r_t.append(run_episode(ag, mtmp.get_repr(), freeze=True))
            ests_all.append(np.mean(r_t))

        print(f"Got all estimates for all policies in {time()-a} seconds")

        for k in range(args.num_strat_try):
            t_mdps_tr, t_mdps_s , ests_tr, ests_s= train_test_split(
                mdps_tr, ests_all, test_size=args.num_safety
            )

            # est_tr = []
            # est_s = []

            # for mtr in t_mdps_tr:
            #     r_t = []
            #     for p in pols:
            #         ag = ActorCriticGridworld(
            #             mdps_tr[0], p
            #         )
            #         r_t.append(run_episode(ag, mtr.get_repr(), freeze=True))
            #     est_tr.append(np.mean(r_t))
            
            # for mtr in t_mdps_s:
            #     r_t = []
            #     for p in pols:
            #         ag = ActorCriticGridworld(
            #             mdps_tr[0], p
            #         )
            #         r_t.append(run_episode(ag, mtr.get_repr(), freeze=True))
            #     est_s.append(np.mean(r_t))
            diff = np.abs(np.mean(ests_tr) - np.mean(ests_s))
            if diff < best_diff:
                best_diff = diff
                mdps_tr_best, mdps_s_best = t_mdps_tr, t_mdps_s
                print(f"Found best diff: {best_diff} ")
        mdps_tr, mdps_s = mdps_tr_best, mdps_s_best


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
    np.random.shuffle(mdps_tr)
    num_train_eval =0
    # tot_num_eval = get_episodes_from_env(len(mdps_tr))
    for m in mdps_tr:
        rewards.append(run_episode(agent, m.get_repr(), True))
        # num_train_eval+=1
        # if num_train_eval > tot_num_eval:
        #     break

    print("Expected Discounteed reward: ", np.mean(rewards))
    print(f"Finished evaluation in {time()-a} seconds")

    # candidate test using the size of the safety set
    g_cand_mean, g_cand_dev = ttest_bounds(
        np.array(rewards), delta, n=len(mdps_s))

    g_cand = g_cand_mean - g_cand_dev

    print(
        f"[CANDIDATE SELECTION] Minimum reward using ttest bounds: {g_cand_mean - g_cand_dev}. Mean: {g_cand_mean}. Std dev:{g_cand_dev}")

    safe_cand = g_cand > min_reward
    if safe_cand:
        print("Policy found!")
    else:
        print("No Solution Found!")

    # safety tests
    print("Running safety test")
    s_rewards = []
    np.random.shuffle(mdps_s)
    # n_safe_runs = 0
    # tot_num_safety_eps = get_eps_for_n(len(mdps_s))
    for m in mdps_s:
        s_rewards.append(run_episode(agent, m.get_repr(), True))

    g_safe_mean, g_safe_dev = ttest_bounds(np.array(s_rewards), delta=delta)
    g_safe = g_safe_mean - g_safe_dev
    print(
        f"[SAFETY TEST] Minimum reward using ttest bounds: {g_safe_mean - g_safe_dev}. Mean: {g_safe_mean}. Std dev:{g_safe_dev}")

    safe = g_safe > min_reward

    fail = 0
    if safe:
        print(f"Candidate set passed safety test!")
    else:
        if safe_cand:
            print(f"Failed safety test")
            fail = 1

    print("Running evaluation on test MDPs")
    te_rewards = []
    np.random.shuffle(mdps_te)
    num_eval = get_eps_for_n(len(mdps_te), 100000)
    for m in mdps_te[:num_eval]:
        te_rewards.append(run_episode(agent, m.get_repr(), True))

    te_reward = np.mean(te_rewards)
    if fail == 0 and not safe_cand:
        te_reward = min_reward
    res['fail'] = fail
    res['g_safe_mean'] = g_safe_mean
    res['g_safe_dev'] = g_safe_dev
    res['te_reward'] = te_reward
    res['sol_found'] = int(safe_cand)
    return res


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # for i in np.geomspace(10, 300, 30):
    #     main(int(i), args)

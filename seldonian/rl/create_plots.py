import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from pathlib import Path
import pickle
from rl_utils import *

# res_control = "output_swarm_v4"
# res_test = "output_swarn_v4_strat"

# out = "figures"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", default="output_swarm_v4")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--output", default="figures")
    parser.add_argument("--delta", default=0.1, type=float)
    return parser.parse_args()

def main(args):
    res_control = args.control_dir
    res_test = args.test_dir
    out = args.output
    os.makedirs(out, exist_ok=True)
    results_control = {
        'n': [],
        'sol_found': [],
        'fail': [],
        'te_reward': [],
        'g_safe_mean': []
    }
    results_test = {
        'n': [],
        'sol_found': [],
        'fail': [],
        'te_reward': [],
        'g_safe_mean': []
    }
    # for baseline
    for f in glob.glob(os.path.join(res_control , "results*.p")):
        n = int(str(f).split(".")[0].split("_")[-1])
        results_control['n'].append(n)
        ress = pickle.load(open(f, "rb"))
        temp_res = {
            'sol_found': [],
            'fail': [],
            'te_reward': [],
            'g_safe_mean': []
        }
        for r in ress:
            for k in temp_res.keys():
                if k in r:
                    temp_res[k].append(r[k])
        for key in temp_res.keys():
            results_control[key].append(
                (np.mean(temp_res[key]), stderror(temp_res[key])))

    # for stratified sampling
    if res_test is not None:
        for f in glob.glob(os.path.join(res_test , "results*.p")):
            n = int(str(f).split(".")[0].split("_")[-1])
            results_test['n'].append(n)
            ress = pickle.load(open(f, "rb"))
            temp_res = {
                'sol_found': [],
                'fail': [],
                'te_reward': [],
                'g_safe_mean': []
            }
            for r in ress:
                for k in temp_res.keys():
                    if k in r:
                        temp_res[k].append(r[k])
            for key in temp_res.keys():
                results_test[key].append(
                    (np.mean(temp_res[key]), stderror(temp_res[key])))

    # plot values

    fig_params = {
        'figsize':(12,5), 'dpi':200
    }
    x_b = np.array(results_control['n'])
    b_i = np.argsort(x_b)
    x_t = np.array(results_test['n'])
    t_i = np.argsort(x_t)
    # print(x_t.shape, t_i.shape)
    plt.figure(**fig_params)
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Solution Found rate")
    plt.xlabel("Number of train MDPs")
    plt.plot(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['sol_found'], i=0), b_i, axis=0),
        'b-', linewidth=3, label='[C]Sol found'
        )
    plt.errorbar(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['sol_found'], i=0), b_i, axis=0),
        yerr=np.take_along_axis(get_index(results_control['sol_found'], i=1), b_i, axis=0),
        fmt='.k'
        )
    if res_test is not None:
        plt.plot(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['sol_found'], i=0), t_i, axis=0),
            'r-', linewidth=3, label='[T]Sol found'
            )
        plt.errorbar(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['sol_found'], i=0), t_i, axis=0),
            yerr=np.take_along_axis(get_index(results_test['sol_found'], i=1), t_i, axis=0),
            fmt='.k'
            )
    plt.legend()
    plt.savefig(f"{out}/sol_found.png")
    plt.show(block=False)
    
    plt.figure(**fig_params)
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Failure rate")
    plt.xlabel("Number of train MDPs")
    plt.plot(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['fail'], i=0), b_i, axis=0),
        'b-', linewidth=3, label='[C]Fail Rate'
        )
    plt.plot(
        np.take_along_axis(x_b,b_i, axis=0),
        [args.delta]*len(x_b), 
        '-k',
        label=f"y={args.delta}"
    )
    plt.errorbar(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['fail'], i=0), b_i, axis=0),
        yerr=np.take_along_axis(get_index(results_control['fail'], i=1), b_i, axis=0),
        fmt='.k'
        )
    if res_test is not None:
        plt.plot(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['fail'], i=0), t_i, axis=0),
            'r-', linewidth=3, label='[T]Fail Rate'
            )
        plt.errorbar(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['fail'], i=0), t_i, axis=0),
            yerr=np.take_along_axis(get_index(results_test['fail'], i=1), t_i, axis=0),
            fmt='.k'
            )
    plt.legend()
    plt.savefig(f"{out}/fail_rate.png")
    plt.show(block=False)

    plt.figure(**fig_params)
    plt.ylabel("Total expected reward")
    plt.xlabel("Number of train MDPs")
    plt.plot(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['te_reward'], i=0), b_i, axis=0),
        'b-', linewidth=3, label='[C]Expected reward'
        )
    plt.errorbar(
        np.take_along_axis(x_b,b_i, axis=0), 
        np.take_along_axis(get_index(results_control['te_reward'], i=0), b_i, axis=0),
        yerr=np.take_along_axis(get_index(results_control['te_reward'], i=1), b_i, axis=0),
        fmt='.k'
        )
    if res_test is not None:
        plt.plot(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['te_reward'], i=0), t_i, axis=0),
            'r-', linewidth=3, label='[T]Expected Reward'
            )
        plt.errorbar(
            np.take_along_axis(x_t,t_i, axis=0), 
            np.take_along_axis(get_index(results_test['te_reward'], i=0), t_i, axis=0),
            yerr=np.take_along_axis(get_index(results_test['te_reward'], i=1), t_i, axis=0),
            fmt='.k'
            )
    plt.legend()
    plt.savefig(f"{out}/reward.png")
    plt.show()
    pass

def get_index(a, i=0):
    return np.array(list(map(lambda x: x[i], a)))


if __name__ == "__main__":

    main(parse_args())

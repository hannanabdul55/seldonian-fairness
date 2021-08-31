import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import os

c_exp_data = Path("python_tutorial/experiment_results/csv")
t_exp_data = Path("code/experiment_results/csv")
res = Path("results_compare")

os.makedirs(res, exist_ok=True)

fr = "fs.csv"
sol = "solutions_found.csv"
g1 = "failures_g1.csv"
g2 = "failures_g2.csv"

def plot_compare(ctrl, test, ylabel, out, prob=False):
    c_n, c_qsa, c_qsa_err, c_ls, c_ls_err = np.loadtxt(ctrl, delimiter=',', unpack=True)
    t_n, t_qsa, t_qsa_err, t_ls, t_ls_err = np.loadtxt(test, delimiter=',', unpack=True)
    plt.clf()
    plt.xlim(min(c_n), max(c_n))
    plt.xlabel("Amount of data (n)", fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.xscale('log')
    if prob:
        plt.ylim(-0.1, 1.1)
    plt.plot(c_n, c_qsa, 'b-', linewidth=3, label='nostrat-qsa')
    plt.errorbar(c_n, c_qsa, yerr=c_qsa_err, fmt='.k')
    plt.plot(c_n, c_ls, 'r-', linewidth=3, label='nostrat-ls')
    plt.errorbar(c_n, c_ls, yerr=c_ls_err, fmt='.k')

    plt.plot(c_n, t_qsa, 'g-', linewidth=3, label='strat-qsa')
    plt.errorbar(c_n, t_qsa, yerr=t_qsa_err, fmt='.k')
    plt.plot(c_n, t_ls, 'c-', linewidth=3, label='strat-ls')
    plt.errorbar(c_n, t_ls, yerr=t_ls_err, fmt='.k')

    plt.legend(fontsize=12)
    plt.savefig(out)
    plt.show()


if __name__ =='__main__':
    plot_compare(c_exp_data/fr, t_exp_data/fr, "MSE", res/"mse.png", prob=False)
    plot_compare(c_exp_data/sol, t_exp_data/sol, "Solution Found Rate", res/"sol.png", prob=True)
    plot_compare(c_exp_data/g1, t_exp_data/g1, r'$Pr(g_1(\theta) > 0)$', res/"g1.png", prob=True)
    plot_compare(c_exp_data/g2, t_exp_data/g2, r'$Pr(g_2(\theta) > 0)$', res/"g2.png", prob=True)
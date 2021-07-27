import sys

import matplotlib.pyplot as plt
import pickle
import glob


def compare_results(res1, opt1, res2, opt2):
    fig, axs = plt.subplots()
    pass


def plot_results(res, opt, display=True, axs=None, show_subtitle=False):
    if axs is None:
        fig, axs = plt.subplots(4, 1, figsize=(6.4, 4.8*4))
    x = []
    sol_found_c = []
    sol_found_c_std = []

    ghat_c = []
    ghat_uc = []

    fr_c = []
    fr_c_std = []

    fr_uc = []
    fr_uc_std = []

    acc_c = []
    acc_uc = []
    for r in res:
        x.append(r['N'])
        fr_c.append(r['failure_rate'])
        fr_uc.append(r['uc_failure_rate'])

        fr_c_std.append(r['failure_rate_std'])
        fr_uc_std.append(r['uc_failure_rate_std'])

        sol_found_c.append(r['sol_found_rate'])
        sol_found_c_std.append(r['sol_found_rate_std'])

        acc_c.append(r['mse'])
        acc_uc.append(r['uc_mse'])

        ghat_c.append(r['ghat'])
        ghat_uc.append(r['uc_ghat'])
    if display:
        ax = axs[0]
        ax.plot(x, fr_c, label='Constrained failure rate')
        ax.plot(x, fr_uc, label='Unconstrained failure rate')
        ax.set_xlabel('Number of data points')
        ax.set_ylabel('failure rate')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        if show_subtitle:
            ax.set_title(f"[{opt}]data vs failure rate")
        else:
            ax.set_title(f"[{opt}]")
        # ax.show()

        ax = axs[1]
        ax.plot(x, sol_found_c, label='Solution Found Rate')
        ax.errorbar(x, sol_found_c, yerr=sol_found_c_std, fmt='.k')
        ax.set_xlabel('Number of data points')
        ax.set_ylabel('Probability of solution found')
        ax.set_xscale('log')
        ax.grid(True)
        ax.legend()
        if show_subtitle:
            ax.set_title(f"[{opt}]data vs Pr(Solution Found)")
        else:
            ax.set_title(f"[{opt}]")
        # ax.show()

        ax = axs[2]
        ax.plot(x, ghat_c, label='Constrained ghat')
        ax.plot(x, ghat_uc, label='Unconstrained ghat')
        ax.set_xlabel('Number of data points')
        ax.set_ylabel('mean ghat')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        if show_subtitle:
            ax.set_title(f"[{opt}]data vs mean ghat")
        else:
            ax.set_title(f"[{opt}]")
        # plt.show()

        ax = axs[3]
        ax.plot(x, acc_c, label='Constrained accuracy')
        ax.plot(x, acc_uc, label='Unconstrained accuracy')
        ax.set_xlabel('Number of data points')
        ax.set_ylabel('Accuracy')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True)
        if show_subtitle:
            ax.set_title(f"[{opt}]data vs accuracy")
        else:
            ax.set_title(f"[{opt}]")
        # ax.show()
    return x, sol_found_c, sol_found_c_std, ghat_c, ghat_uc, fr_c, fr_c_std, fr_uc, fr_uc_std, acc_c, acc_uc


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # raise ValueError("Specify the result folder ")
        folder = 'result/result_powell_100tr_30n_stratify'
    else:
        folder = sys.argv[1].strip()
    config = pickle.load(open(folder + "/config.p", "rb"))
    exps = pickle.load(open(folder + "/exps.p", "rb"))
    res = pickle.load(open(list(glob.glob(folder + '/final_res*.p'))[0], 'rb'))

    print(f"Plotting results for config: {config!r}")
    opt = config['opt']
    plot_results(res, opt)

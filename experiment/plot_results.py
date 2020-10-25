import sys

import matplotlib.pyplot as plt
import pickle
import glob


def plot_results(res, opt):
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

        acc_c.append(r['accuracy'])
        acc_uc.append(r['uc_accuracy'])

        ghat_c.append(r['ghat'])
        ghat_uc.append(r['uc_ghat'])

    plt.plot(x, fr_c, label='Constrained failure rate')
    plt.plot(x, fr_uc, label='Unconstrained failure rate')
    plt.xlabel('Number of data points')
    plt.ylabel('failure rate')
    plt.legend()
    plt.xscale('log')
    plt.title(f"[{opt}]Plot for number of data points vs failure rate")
    plt.show()

    plt.plot(x, sol_found_c, label='Solution Found Rate')
    plt.errorbar(x, sol_found_c, yerr=sol_found_c_std, fmt='.k')
    plt.xlabel('Number of data points')
    plt.ylabel('Probability of solution found')
    plt.xscale('log')
    plt.legend()
    plt.title(f"[{opt}]Plot for number of data points vs Pr(Solution Found)")
    plt.show()

    plt.plot(x, ghat_c, label='Constrained ghat')
    plt.plot(x, ghat_uc, label='Unconstrained ghat')
    plt.xlabel('Number of data points')
    plt.ylabel('mean ghat')
    plt.legend()
    plt.xscale('log')
    plt.title(f"[{opt}]Plot for number of data points vs mean ghat")
    plt.show()

    plt.plot(x, acc_c, label='Constrained accuracy')
    plt.plot(x, acc_uc, label='Unconstrained accuracy')
    plt.xlabel('Number of data points')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.title(f"[{opt}]Plot for number of data points vs accuracy")
    plt.show()


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

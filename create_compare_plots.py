import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser.parse_args()


def stderror(v):
    non_nan = np.count_nonzero(~np.isnan(v))  # number of valid (non NaN) elements in the vector
    return np.nanstd(v, ddof=1) / np.sqrt(non_nan)


def main():
    args = get_args()
    control_res = {}
    for control_f in glob.glob(f"{args.control_dir}/result*.p"):
        res_arr = pickle.load(open(control_f, "rb"))
        for (n, acc, sol, ghat, fr, uc_acc, uc_ghat, uc_fr) in res_arr:
            if n not in control_res:
                control_res[n] = []
            control_res[n].append([acc, sol, ghat, fr, uc_acc, uc_ghat, uc_fr])

    test_res = {}
    for test_f in glob.glob(f"{args.test_dir}/result*.p"):
        res_arr = pickle.load(open(test_f, "rb"))
        for (n, acc, sol, ghat, fr, uc_acc, uc_ghat, uc_fr) in res_arr:
            if n not in test_res:
                test_res[n] = []
            test_res[n].append([acc, sol, ghat, fr, uc_acc, uc_ghat, uc_fr])

    for k in control_res:
        control_res[k] = np.array(control_res[k])

    for k in test_res:
        test_res[k] = np.array(test_res[k])
    print(f"Control res shape: {list(control_res.values())[0].shape}")
    print(f"Test res shape: {list(test_res.values())[0].shape}")
    ns = sorted(list(control_res.keys()))
    acc = list(map(lambda x: np.mean(control_res[x][:, 0]), ns))
    acc_err = list(map(lambda x: stderror(control_res[x][:, 0]), ns))
    sol = list(map(lambda x: np.mean(control_res[x][:, 1]), ns))
    sol_err = list(map(lambda x: stderror(control_res[x][:, 1]), ns))
    ghat = list(map(lambda x: np.mean(control_res[x][:, 2]), ns))
    fr = list(map(lambda x: np.mean(control_res[x][:, 3]), ns))
    fr_err = list(map(lambda x: stderror(control_res[x][:, 3]), ns))
    uc_acc = list(map(lambda x: np.mean(control_res[x][:, 4]), ns))
    uc_acc_err = list(map(lambda x: stderror(control_res[x][:, 4]), ns))
    uc_ghat = list(map(lambda x: np.mean(control_res[x][:, 5]), ns))
    uc_fr = list(map(lambda x: np.mean(control_res[x][:, 6]), ns))
    uc_fr_err = list(map(lambda x: stderror(control_res[x][:, 6]), ns))

    plt.plot(ns, acc, 'b', label="accuracy")
    plt.errorbar(ns, acc, yerr=acc_err, fmt=".k")
    plt.plot(ns, uc_acc, 'r', label="unconstrained accuracy")
    plt.errorbar(ns, uc_acc, yerr=uc_acc_err, fmt=".k")
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.ylim(0, 1.1)
    plt.xscale('log')
    plt.plot(ns, sol, label="solution found rate")
    plt.errorbar(ns, sol, yerr=sol_err, fmt=".k")
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.xscale('log')
    plt.plot(ns, ghat, label="value of g(\\theta)")
    plt.plot(ns, uc_ghat, label="unconstrained value of g(\\theta)")
    plt.legend()
    plt.show(block=False)

    ns_norm = sorted(list(control_res.keys()))
    sol_norm = list(map(lambda x: np.mean(control_res[x][:, 1]), ns_norm))
    sol_err_norm = list(map(lambda x: stderror(control_res[x][:, 1]), ns_norm))

    ns = sorted(list(test_res.keys()))
    acc = list(map(lambda x: np.mean(test_res[x][:, 0]), ns))
    acc_err = list(map(lambda x: stderror(test_res[x][:, 0]), ns))
    sol = list(map(lambda x: np.mean(test_res[x][:, 1]), ns))
    sol_err = list(map(lambda x: stderror(test_res[x][:, 1]), ns))
    ghat = list(map(lambda x: np.mean(test_res[x][:, 2]), ns))
    fr = list(map(lambda x: np.mean(test_res[x][:, 3]), ns))
    fr_err = list(map(lambda x: stderror(test_res[x][:, 3]), ns))
    uc_acc = list(map(lambda x: np.mean(test_res[x][:, 4]), ns))
    uc_acc_err = list(map(lambda x: stderror(test_res[x][:, 4]), ns))
    uc_ghat = list(map(lambda x: np.mean(test_res[x][:, 5]), ns))
    uc_fr = list(map(lambda x: np.mean(test_res[x][:, 6]), ns))
    uc_fr_err = list(map(lambda x: stderror(test_res[x][:, 6]), ns))

    plt.figure()
    plt.plot(ns, acc, 'b', label="accuracy")
    plt.errorbar(ns, acc, yerr=acc_err, fmt=".k")
    plt.plot(ns, uc_acc, 'r', label="unconstrained accuracy")
    plt.errorbar(ns, uc_acc, yerr=uc_acc_err, fmt=".k")
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.plot(ns, sol, label="[strat]solution found rate")
    plt.errorbar(ns, sol, yerr=sol_err, fmt=".k")
    plt.plot(ns_norm, sol_norm, label="solution found rate")
    plt.errorbar(ns_norm, sol_norm, yerr=sol_err_norm, fmt=".k")
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.xscale('log')
    plt.plot(ns, ghat, label="value of g(\\theta)")
    plt.plot(ns, uc_ghat, label="unconstrained value of g(\\theta)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

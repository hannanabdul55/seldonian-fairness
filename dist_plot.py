import numpy as np
import matplotlib.pyplot as plt
import glob
import re

dists = []

# bin_path = 'experiment_results/bin/'
csv_path = 'experiment_results/csv/'


def get_existing_experiment_numbers(bin_path):
    result_files = glob.glob(bin_path + 'results*.npz')
    experiment_numbers = [
        re.search('.*results([0-9]*).*', fn, re.IGNORECASE) for fn in result_files]
    experiment_numbers = [int(i.group(1)) for i in experiment_numbers]
    experiment_numbers.sort()
    return experiment_numbers


def genFilename(n,bin_path):
    return bin_path + 'results%d.npz' % n


def stderror(v, axis=None):
    # number of valid (non NaN) elements in the vector
    non_nan = np.count_nonzero(~np.isnan(v), axis=axis)
    return np.nanstd(v, ddof=1, axis=axis) / np.sqrt(non_nan)


def addMoreResults(newFileId, ms, dists, bin_path):

    newFile = np.load(genFilename(newFileId, bin_path))
    new_ms = newFile['ms']
    new_dists = newFile['dists']

    if type(ms) == type(None):
        return [new_ms, new_dists]
    else:
        dists = np.vstack([dists,            new_dists])

        return [ms, dists]


def gather_results(bin_path):
    ms = None
    dists = None

    experiment_numbers = get_existing_experiment_numbers(bin_path)

    for file_idx in experiment_numbers:
        res = addMoreResults(file_idx,
                             ms,
                             dists,
                             bin_path)
        [ms, dists] = res
    return ms, dists

if __name__ == '__main__':

    #for strat
    ms_strat, dists_strat = gather_results('code/experiment_results/bin/')
    ms, dists = gather_results('python_tutorial/experiment_results/bin/')
    # print(ms_strat.shape, dists_strat.shape)
    plt.figure(figsize=(12,5), dpi=150)
    plt.plot(ms_strat, np.mean(dists_strat, axis=0), label="STRAT mean distance")
    plt.errorbar(ms_strat, np.mean(dists_strat, axis=0),yerr=stderror(dists_strat, axis=0), fmt='.k')
    
    plt.plot(ms, np.mean(dists, axis=0), label="NOSTRAT mean distance")
    plt.errorbar(ms, np.mean(dists, axis=0),yerr=stderror(dists, axis=0), fmt='.k')
    plt.legend()
    plt.xscale('log')
    plt.savefig('distance_plots.png')
    plt.show()

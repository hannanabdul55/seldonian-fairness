import numpy as np
import matplotlib.pyplot as plt
import glob
import re

dists = []

bin_path = 'experiment_results/bin/'
csv_path = 'experiment_results/csv/'


def get_existing_experiment_numbers():
    result_files = glob.glob(bin_path + 'results*.npz')
    experiment_numbers = [
        re.search('.*results([0-9]*).*', fn, re.IGNORECASE) for fn in result_files]
    experiment_numbers = [int(i.group(1)) for i in experiment_numbers]
    experiment_numbers.sort()
    return experiment_numbers


def genFilename(n):
    return bin_path + 'results%d.npz' % n


def stderror(v):
    # number of valid (non NaN) elements in the vector
    non_nan = np.count_nonzero(~np.isnan(v))
    return np.nanstd(v, ddof=1) / np.sqrt(non_nan)


def addMoreResults(newFileId, ms, dists):

    newFile = np.load(genFilename(newFileId))
    new_ms = newFile['ms']
    new_dists = newFile['dists']

    if type(ms) == type(None):
        return [new_ms, new_dists]
    else:
        dists = np.vstack([dists,            new_dists])

        return [ms, dists]


def gather_results():
    ms = None
    dists = None

    experiment_numbers = get_existing_experiment_numbers()

    for file_idx in experiment_numbers:
        res = addMoreResults(file_idx,
                             ms,
                             dists)
        [ms, dists] = res
    return ms, dists

if __name__ == '__main__':
    ms, dists = gather_results()
    print(ms.shape, dists.shape)
    plt.figure()
    plt.plot(ms, np.mean(dists, axis=0), label="mean dists")
    plt.xscale('log')
    plt.show()

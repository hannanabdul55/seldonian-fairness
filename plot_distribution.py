import glob
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

import argparse
from distance_utils import *

import json

data_dir = '../seldonian-data'
control_dir = 'res_control'
test_dir = 'res_test'
ext = '.txt.npz'
cand_file_pat = 'c_'
safe_file_pat = 's_'

def get_m(filename):
    return int(filename.split('/')[-1].split('.')[0].split('_')[-1])

output_dir = Path(os.path.join(data_dir, "output_plots"))

parser = argparse.ArgumentParser()
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir/control_dir, exist_ok=True)
    os.makedirs(output_dir/test_dir, exist_ok=True)
    idx = 0
    nostrat_dist = {}
    strat_dist = {}

    plot=False

    # baseline data
    for file in glob.glob(f"{data_dir}/{control_dir}/{cand_file_pat}*{ext}"):
      cand = np.load(file)['arr_0']
      safe = np.load(file.replace(cand_file_pat, safe_file_pat))['arr_0']
    #   print(cand['arr_0'])
      m = get_m(file)
      if m not in nostrat_dist:
        nostrat_dist[m] = []
      print(f"[CONTROL]Calculating metric for {file} with {m} samples")
      nostrat_dist[m].append(float(calc_dist(cand, safe)))
      # if m>1000:
      #   continue
      if plot:
        print(f"[CONTROL]Plotting {file} with {m} samples to {output_dir/control_dir}/{m}.png")

        plt.scatter(cand[:,0], cand[:,1],s=5, cmap="Blues", marker='.', alpha=0.5, label="Candidate set")
        plt.scatter(safe[:,0], safe[:,1],s=5, cmap="Reds", marker='.', alpha=0.5, label="safety set")
        plt.legend()
        plt.title(f"[CONTROL]Distribution for {m} datapoints")
        plt.savefig(f"{output_dir/control_dir}/{m}.png")
        plt.close()

    for file in glob.glob(f"{data_dir}/{test_dir}/{cand_file_pat}*{ext}"):
      cand = np.load(file)['arr_0']
      safe = np.load(file.replace(cand_file_pat, safe_file_pat))['arr_0']
    #   print(cand['arr_0'])
      m = get_m(file)

      print(f"[TEST]Calculating metric for {file} with {m} samples")
      if m not in strat_dist:
        strat_dist[m] = []
      strat_dist[m].append(float(calc_dist(cand, safe)))
      # if m>1000:
      #     continue
      if plot:
        print(f"[TEST]Plotting {file} with {m} samples to {output_dir/test_dir}/{m}.png")

        plt.scatter(cand[:,0], cand[:,1],s=5, cmap="Blues", marker='.', alpha=0.5, label="Candidate set")
        plt.scatter(safe[:,0], safe[:,1],s=5, cmap="Reds", marker='.', alpha=0.5, label="safety set")
        plt.legend()
        plt.title(f"[TEST]Distribution for {m} datapoints")
        plt.savefig(f"{output_dir/test_dir}/{m}.png")
        plt.close()
    
    # save results
    json.dump(strat_dist, open('results_strat.json', 'w'))
    json.dump(nostrat_dist, open('results_nostrat.json', 'w'))
    
    res_dir = Path("distribution_results")
    os.makedirs(res_dir, exist_ok=True)

    c_k = []
    c_v = []

    t_k=[]
    t_v = []
    for k,v in nostrat_dist.items():
      c_k.append(k)
      c_v.append(v)
    
    for k,v in strat_dist.items():
      t_k.append(k)
      t_v.append(v)
    
    c_k = np.array(c_k, dtype=int)
    c_v = np.array(c_v)
    t_k = np.array(t_k, dtype=int)
    t_v = np.array(t_v)


    c_idx = np.argsort(c_k)
    t_idx = np.argsort(t_k)

    print(f"c_k: {c_k.shape} | c_v: {c_v.shape} | c_idx: {c_idx.shape}")
    print(f"t_k: {t_k.shape} | t_v: {t_v.shape}")
    c_keys = np.take_along_axis(c_k, c_idx, 0)
    c_vals = c_v[c_idx, :]

    t_keys = np.take_along_axis(t_k, t_idx, 0)
    t_vals = t_v[t_idx, :]
    plt.figure()

    # print(c_keys)
    # print(np.mean(c_vals, axis=1))
    plt.plot(
      c_keys,
      np.mean(c_vals, axis=1),
      label='Baseline'
      )
    plt.errorbar(
      c_keys,
      np.mean(c_vals, axis=1),
      yerr=stderror(c_vals, axis=1),
      fmt='.k'
    )
    plt.plot(
      t_keys,
      np.mean(t_vals, axis=1),
      label='Stratified'
      )
    plt.errorbar(
      t_keys,
      np.mean(t_vals, axis=1),
      yerr=stderror(t_vals, axis=1),
      fmt='.k'
    )
    plt.title('directed hausdorff distance metric evaluation on candidate-safety split')
    plt.legend()
    plt.savefig(res_dir/'results.png')
    plt.show()








import glob
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

import argparse

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
    for file in glob.glob(f"{data_dir}/{control_dir}/{cand_file_pat}*{ext}"):
      cand = np.load(file)['arr_0']
      safe = np.load(file.replace(cand_file_pat, safe_file_pat))['arr_0']
    #   print(cand['arr_0'])
      m = get_m(file)
      if m<1000:
          continue
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
      if m<1000:
          continue
      print(f"[TEST]Plotting {file} with {m} samples to {output_dir/test_dir}/{m}.png")

      plt.scatter(cand[:,0], cand[:,1],s=5, cmap="Blues", marker='.', alpha=0.5, label="Candidate set")
      plt.scatter(safe[:,0], safe[:,1],s=5, cmap="Reds", marker='.', alpha=0.5, label="safety set")
      plt.legend()
      plt.title(f"[TEST]Distribution for {m} datapoints")
      plt.savefig(f"{output_dir/test_dir}/{m}.png")
      plt.close()
      








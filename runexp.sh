#!/bin/bash
#
#SBATCH --job-name=exp_run_akanji
#SBATCH --output=stdoutput/res.txt  # output file
#SBATCH -e stdoutput/res.err        # File to which STDERR will be written
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-20:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=200    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
#
#SBATCH --array=1-100

export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export OMP_NUM_THREADS=5


python experiment_synthetic.py

sleep 1

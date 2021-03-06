#!/bin/bash
#
#SBATCH --job-name=exp_run_cm
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=1080ti-long   # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-12:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=6gb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
export PYTHONPATH=/home/akanji/seldonian-fairness:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
/home/akanji/miniconda3/bin/python  experiment/experiment_s_p.py "$@"

sleep 1

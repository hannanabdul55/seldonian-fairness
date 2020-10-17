#!/bin/bash
#
#SBATCH --job-name=exp_run_pow
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res%j.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-22:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=4gb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu

export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
/home/akanji/miniconda3/envs/seldnian-pre/bin/python  experiment_s_p.py "$@"

sleep 1

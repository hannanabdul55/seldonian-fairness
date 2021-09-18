#!/bin/bash
#
#SBATCH --job-name=exp_run
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-22:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=4gb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
# export PYTHONPATH=/home/akanji/seldonian-fairness:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
cd code
/home/akanji/miniconda3/envs/seldnian-pre/bin/python  main_plotting.py "$@"
cd ..

sleep 1

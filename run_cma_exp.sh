#!/bin/bash
#
#SBATCH --job-name=exp_cma_run
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-22:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=500    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
#SBATCH --array=1-1000%100
export PYTHONPATH=/home/akanji/seldonian-fairness:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
/home/akanji/miniconda3/envs/seldnian-pre/bin/python experiment/run_experiment.py --worker-id $SLURM_ARRAY_JOB_ID "$@"

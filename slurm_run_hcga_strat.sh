#!/bin/bash
#
#SBATCH --job-name=exp_run
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-22:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=500    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
#SBATCH --array=0-99
export PYTHONPATH=/home/akanji/seldonian-fairness:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
cd seldonian/rl
/home/akanji/miniconda3/envs/seldnian-pre/bin/python run_hcga.py --workerid $SLURM_ARRAY_JOB_ID --index $SLURM_ARRAY_TASK_ID --strat true "$@"
cd ../..
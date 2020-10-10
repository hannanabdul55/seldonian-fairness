#!/bin/bash
#
#SBATCH --job-name=exp_run
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-11:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=400mb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu

export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
/home/akanji/miniconda3/envs/seldnian-pre/bin/python  experiment_synthetic.py $1

sleep 1

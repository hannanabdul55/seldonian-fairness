#!/bin/bash
#
#SBATCH --job-name=exp_run
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to

#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-11:59         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=300mb    # Memory in MB per cpu allocated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanji@cs.umass.edu
export PYTHONPATH=`pwd`:$PYTHONPATH
python  experiment/experiment_s_p.py "$@"

sleep 1

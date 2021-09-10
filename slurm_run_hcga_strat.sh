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
#SBATCH --array=3,20,50,80,  104,  206,  308,  410,  512,  614,  716,  818,  920, 1022,1124, 1226, 1328, 1430, 1532, 1634, 1736, 1838, 1940, 2042, 2144,2246, 2348, 2450, 2552, 2654, 2756, 2858, 2960, 3062, 3164, 3266,3368, 3470, 3572, 3674, 3776, 3878, 3980, 4082, 4184, 4286, 4388,4490, 4592, 4694, 4796, 4898, 5000
export PYTHONPATH=/home/akanji/seldonian-fairness:$PYTHONPATH
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7
export OMP_NUM_THREADS=7
cd seldonian/rl
/home/akanji/miniconda3/envs/seldnian-pre/bin/python run_hcga.py --workerid $SLURM_ARRAY_TASK_ID --num-training $SLURM_ARRAY_TASK_ID --out-dir output_swarm_strat_v3 --strat true 
cd ../..
#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --time=36:00:00

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 python/3.11.2
python run_comparison.py
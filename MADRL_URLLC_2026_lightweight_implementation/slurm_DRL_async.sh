#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

module load scicomp-python-env
python DRL_async_ci_train.py

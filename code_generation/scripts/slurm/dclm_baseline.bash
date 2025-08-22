#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=60:00:00 
#SBATCH --mem=236gb
#SBATCH --output=logs/dclm_baseline_%j.out
#SBATCH --error=logs/dclm_baseline_%j.err
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
source .venv/bin/activate
python -u ./code_generation/scripts/dclm/filter_dclm_baseline.py
#!/bin/bash
#SBATCH --job-name=deduplication
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%A_%a.out

echo "Working Directory: $PWD"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE CPUs per node with JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

export TRANSFORMERS_CACHE="/pfs/work9/workspace/scratch/fr_aa502-code_eval/models"
export HF_HOME="/pfs/work9/workspace/scratch/fr_aa502-code_eval/models"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "SLURM node list: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python instruction_deduplication.py

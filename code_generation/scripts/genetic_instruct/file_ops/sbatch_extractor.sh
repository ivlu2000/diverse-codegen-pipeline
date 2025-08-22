#!/bin/bash
#SBATCH --job-name=deduplication
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%A_%a.out

echo "Working Directory: $PWD"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE CPUs per node with JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

echo "SLURM node list: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python  subset_extractor.py

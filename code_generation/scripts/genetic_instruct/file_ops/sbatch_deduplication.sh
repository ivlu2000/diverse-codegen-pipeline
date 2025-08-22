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

# Optional: set a custom Hugging Face cache directory (models, tokenizers, etc.).
# If not set, defaults to ~/.cache/huggingface on your machine.
export HF_HOME="PATH_TO_CUSTOM_CACHE/models"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "SLURM node list: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python instruction_deduplication.py

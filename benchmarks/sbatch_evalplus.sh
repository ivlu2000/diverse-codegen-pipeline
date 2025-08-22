#!/bin/bash
#SBATCH --job-name=qwen_evalplus
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/evalplus_%j.out
#SBATCH --error=logs/evalplus_%j.err

module load devel/cuda

echo "Working Directory: $PWD"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE CPUs per node with JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"


source .venv/bin/activate

# Add evaluation here
echo "Running evaluation..."
evalplus.evaluate --model "ivlu2000/5k-subset-instructions_5" \
                  --dataset mbpp \
                  --backend hf \
                  --greedy




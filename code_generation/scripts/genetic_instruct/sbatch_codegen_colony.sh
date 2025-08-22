#!/bin/bash
#SBATCH --job-name=generation_1
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-100
#SBATCH --output=logs/slurm_%A_%a.out

echo "Working Directory: $PWD"
echo "Running job $SLURM_JOB_NAME with array task ID $SLURM_ARRAY_TASK_ID"

export TRANSFORMERS_CACHE="/pfs/work9/workspace/scratch/fr_aa502-code_gen/models"
export HF_HOME="/pfs/work9/workspace/scratch/fr_aa502-code_gen/models"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export APPTAINER_CACHEDIR="/pfs/work9/workspace/scratch/$USER/apptainer_cache"

echo "SLURM node list: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Start vLLM server in background
echo "Launching vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 &

VLLM_PID=$!

# Wait for vLLM to become available
echo "Waiting for vLLM API to start..."
for i in {1..60}; do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health | grep -q "200"; then
        echo "vLLM API is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

# Use SLURM_ARRAY_TASK_ID as the colony ID
COLONY_ID=$SLURM_ARRAY_TASK_ID

echo "Starting refinement script with colony ID: $COLONY_ID"
CUDA_VISIBLE_DEVICES=0 python genetic_instruction_generator.py --colony-id "$COLONY_ID" --start 0 --end 219399

echo "Shutting down vLLM server..."
kill $VLLM_PID

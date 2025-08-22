#!/bin/bash
#SBATCH --job-name=llm_refinement
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/llm_refinement_%j.out
#SBATCH --error=logs/llm_refinement_%j.err

# Get the offset parameter from command line
if [ $# -eq 0 ]; then
    OFFSET=0
else
    OFFSET=$1
fi

if [ $# -lt 2 ]; then
    TAKE=100
else
    TAKE=$2
fi

echo "Using offset: $OFFSET"
echo "Using take: $TAKE"

# export HF_HOME="/home/fr/fr_fr/fr_il72/work/gpfs/fr_il72-code-llm/synthetic_reasoning/.cache"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source .venv/bin/activate

# Start vLLM server in the background
vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" &
VLLM_PID=$!

# Wait for the API to start (adjust sleep time as needed)
echo "Waiting for vLLM API to start..."
sleep 300  # Give the server time to initialize

# Run the refinement script in the foreground
echo "Starting refinement script..."
python -u code_generation/scripts/fasttext/llm_refinement.py --offset "$OFFSET" --take "$TAKE"

# Cleanup: kill the vLLM server when the script finishes
kill $VLLM_PID


# python -u code_generation/src/fasttext/llm_refinement.py


echo "DONE";
echo "Finished at $(date)";
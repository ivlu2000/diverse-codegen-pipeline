#!/bin/bash
#SBATCH --job-name=finetune_model
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/finetune_model_%j.out
#SBATCH --error=logs/finetune_model_%j.err

# Parse command line arguments
NUM_SAMPLES=${1:-50000}  # Default to 50000 if not provided
DATASET=${2:-"amal-abed/combined_dataset"}  # Dataset to use

echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "DATASET: $DATASET"

module load devel/cuda

source .venv/bin/activate

# Create dataset name and output path
DATASET_NAME="${DATASET}_$((NUM_SAMPLES / 1000))k"
OUTPUT_PATH="code_generation/datasets/final_dataset_${DATASET_NAME}.jsonl"

echo "DATASET_NAME: $DATASET_NAME"
echo "OUTPUT_PATH: $OUTPUT_PATH"

# Load and prepare the dataset
python -u code_generation/scripts/instruction_tuning/load_data.py \
  --dataset $DATASET \
  --num_samples $NUM_SAMPLES \
  --output_path $OUTPUT_PATH

# Pass NUM_SAMPLES as environment variable to the finetune script
export NUM_SAMPLES=$NUM_SAMPLES
export DATA_PATH="./$OUTPUT_PATH"
export OUTPUT_MODEL_PATH="code_generation/models/codegemma-2b-finetuned-${DATASET_NAME}"
bash code_generation/scripts/instruction_tuning/finetune_codegemma.sh

echo "DONE";
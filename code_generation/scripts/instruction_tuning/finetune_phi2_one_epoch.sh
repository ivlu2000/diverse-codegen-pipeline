#!/bin/bash

# Default values - these can be overridden by environment variables
NUM_SAMPLES=${NUM_SAMPLES:-5000}
DATA_PATH=${DATA_PATH:-"code_generation/datasets/final_dataset_${NUM_SAMPLES}k.jsonl"}
OUTPUT_MODEL_PATH=${OUTPUT_MODEL_PATH:-"code_generation/models/phi2-finetuned-final-dataset-${NUM_SAMPLES}k"}
MODEL_PATH="microsoft/phi-2"

echo "DATA_PATH: $DATA_PATH"
echo "OUTPUT_MODEL_PATH: $OUTPUT_MODEL_PATH"

python code_generation/scripts/instruction_tuning/finetune_and_eval_phi2_one_epoch.py \
    --model_name $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_MODEL_PATH \
    --max_length 1024 \
    --batch_size 2 \
    --lr 5e-5 \
    --eval_greedy
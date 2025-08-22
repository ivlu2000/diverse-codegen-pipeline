# Import Libraries
import argparse
import subprocess
import csv
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the .jsonl dataset file"
)
parser.add_argument(
    "--output_dir", type=str, default="./phi2-finetuned", help="Where to save the model"
)
parser.add_argument(
    "--model_name", type=str, default="microsoft/phi-2", help="Base model name"
)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)  # Increased default learning rate
parser.add_argument("--overfit_test", action="store_true", help="Run overfit test on small batch")
parser.add_argument("--eval_greedy", action="store_true", help="Use greedy decoding in evaluation")
args = parser.parse_args()

# 1. Load dataset
print(f"Loading dataset from {args.data_path}")
dataset = load_dataset("json", data_files=args.data_path, split="train")

# Dataset analysis
print(f"Dataset size: {len(dataset)} samples")
print(f"Sample data structure: {dataset[0] if len(dataset) > 0 else 'Empty dataset'}")

# Check for empty or malformed data
empty_instructions = sum(1 for item in dataset if not item.get("instruction", "").strip())
empty_reasonings = sum(1 for item in dataset if not item.get("reasoning", "").strip())
empty_solutions = sum(1 for item in dataset if not item.get("solution", "").strip())
print(f"Empty instructions: {empty_instructions}")
print(f"Empty reasonings: {empty_reasonings}")
print(f"Empty solutions: {empty_solutions}")

if len(dataset) < 10:
    print("WARNING: Very small dataset! Consider using more data.")

# Overfit test setup
if args.overfit_test:
    print("\n=== RUNNING OVERFIT TEST ===")
    # Take only first 5 samples for overfit test
    dataset = dataset.select(range(min(5, len(dataset))))
    print(f"Overfit test dataset size: {len(dataset)}")
    args.epochs = 10  # More epochs for overfit test



# 2. Load tokenizer and model
print(f"Loading model and tokenizer: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

#3. Prompt formatting function
def formatting_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['solution']}"
    return text

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map={"": 0},
    trust_remote_code=True,
    quantization_config=config,
    token=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,         
    target_modules=["q_proj", "v_proj", "k_proj", "dense"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# Print trainable parameters
print("\n=== MODEL ANALYSIS ===")
model.print_trainable_parameters()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Percentage trainable: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.2f}%")

# # 6. Training arguments
# training_args = TrainingArguments(
#     output_dir=args.output_dir,
#     per_device_train_batch_size=args.batch_size,
#     num_train_epochs=args.epochs,
#     learning_rate=args.lr,
#     optim="paged_adamw_32bit",
#     logging_steps=1,
#     save_strategy="epoch",
#     bf16=True,
#     gradient_accumulation_steps=2,
#     lr_scheduler_type="constant",
#     warmup_ratio=0.03,
#     report_to="tensorboard",
# )

dataset = dataset.map(generate_and_tokenize_prompt)

# Create SFTConfig to avoid deprecation warnings
training_args = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    max_steps=10,
    optim="paged_adamw_32bit",
    logging_steps=1,  # Log every step
    logging_first_step=True,  # Log the first step
    logging_dir=f"{args.output_dir}/logs",  # Specify logging directory
    save_strategy="epoch",
    bf16=True,
    gradient_accumulation_steps=16,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="tensorboard",
    max_seq_length=args.max_length,
    packing=False,
    log_level="info",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
)

# 8. Train
print("Starting training...")
print("Training configuration:")
print(f"  - Batch size: {args.batch_size}")
print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {args.batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Max sequence length: {args.max_length}")
print(f"  - Learning rate: {args.lr}")
print(f"  - Epochs: {args.epochs}")
print(f"  - Dataset size: {len(dataset)}")
print(f"  - Logging every {training_args.logging_steps} steps")
print(f"  - Total steps per epoch: {len(dataset) // (args.batch_size * training_args.gradient_accumulation_steps)}")
print("=" * 60)

trainer.train()

# 10. Save model and tokenizer
print(f"\nSaving model to {args.output_dir}")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# -------------------------------
# 11. Evaluate on Humaneval & MBPP
# -------------------------------
print("\n=== EVALUATING MODEL ===")
eval_datasets = ["humaneval"]
eval_outputs = {}

for dataset_name in eval_datasets:
    cmd = [
        "evalplus.evaluate",
        "--model", args.output_dir,
        "--dataset", dataset_name,
        "--backend", "hf"
    ]
    if args.eval_greedy:
        cmd.append("--greedy")

    print(f"Running EvalPlus on {dataset_name}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        eval_outputs[dataset_name] = result.stdout.replace("\n", " ")
        print(f"{dataset_name} results: {eval_outputs[dataset_name]}")
    except subprocess.CalledProcessError as e:
        eval_outputs[dataset_name] = f"EvalPlus failed: {e.stderr}"
        print(f"{dataset_name} evaluation failed: {eval_outputs[dataset_name]}")

# -------------------------------
# 12. Save Evaluation Results
# -------------------------------
eval_results = {
"r": 16,
"alpha": 16,
"target_modules": ",".join(["q_proj", "v_proj", "k_proj", "dense"]),
"save_head": False,
"run_dir": str(args.output_dir),
**{f"eval_{d}": eval_outputs[d] for d in eval_datasets}
}

# Save results to CSV
output_path = Path(args.output_dir)
eval_csv_file = output_path / "evaluation_results.csv"
with open(eval_csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=eval_results.keys())
    writer.writeheader()
    writer.writerow(eval_results)

print(f"\nEvaluation results saved to {eval_csv_file}")
print("=== EVALUATION COMPLETE ===")

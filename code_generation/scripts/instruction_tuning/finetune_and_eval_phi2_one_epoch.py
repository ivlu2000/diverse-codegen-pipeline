#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
import csv

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# -------------------------------
# 1. Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the .jsonl dataset file")
parser.add_argument("--output_dir", type=str, default="./sweep_results", help="Where to save models")
parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Base model name")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=156)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--eval_greedy", action="store_true", help="Use greedy decoding in evaluation")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# -------------------------------
# 2. Load Dataset
# -------------------------------
print(f"Loading dataset from {args.data_path}")
dataset = load_dataset("json", data_files=args.data_path, split="train")
print(f"Dataset size: {len(dataset)}")



# -------------------------------
# 3. Tokenizer & Model Setup
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def formatting_func(example):
    return f"### Question: {example['instruction']}\n### Answer: {example['solution']}"

def generate_and_tokenize_prompt(example):
    return tokenizer(formatting_func(example))

dataset = dataset.map(generate_and_tokenize_prompt)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map={"": 0},
    trust_remote_code=True,
    quantization_config=config,
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# -------------------------------
# 4. LoRA Hyperparameter Sweep
# -------------------------------
r_values = [8, 16]
alpha_factors = [2]
target_module_sets = [
    ["q_proj", "v_proj"],
    ["q_proj", "v_proj", "k_proj"],
]

save_head_options = [False, True]

sweep_results = []

for r in r_values:
    for alpha_factor in alpha_factors:
        alpha = r * alpha_factor
        for target_modules in target_module_sets:
            for save_head in save_head_options:
                run_name = f"lora_r{r}_a{alpha}_{'_'.join(target_modules)}_savehead{save_head}"
                run_dir = output_dir / run_name
                run_dir.mkdir(exist_ok=True)

                print(f"\n=== RUNNING: {run_name} ===\n")

                peft_config = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    target_modules=target_modules,
                    bias="none",
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )

                if save_head:
                    peft_config.fan_in_fan_out = True  # adjust if save_head should modify model

                model = get_peft_model(base_model, peft_config)

                sft_args = SFTConfig(
                    output_dir=str(run_dir),
                    per_device_train_batch_size=args.batch_size,
                    num_train_epochs=args.epochs,
                    learning_rate=args.lr,
                    optim="paged_adamw_32bit",
                    logging_steps=1,
                    logging_first_step=True,
                    logging_dir=str(run_dir / "logs"),
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
                    args=sft_args,
                    tokenizer=tokenizer,
                    dataset_text_field="text",
                )

                # -------------------------------
                # 5. Train
                # -------------------------------
                trainer.train()
                trainer.save_model(run_dir)
                tokenizer.save_pretrained(run_dir)

                # -------------------------------
                # 6. Evaluate on Humaneval & MBPP
                # -------------------------------
                eval_datasets = ["humaneval", "mbpp"]
                eval_outputs = {}

                for dataset_name in eval_datasets:
                    cmd = [
                        "evalplus.evaluate",
                        "--model", str(run_dir),
                        "--dataset", dataset_name,
                        "--backend", "hf"
                    ]
                    if args.eval_greedy:
                        cmd.append("--greedy")

                    print(f"Running EvalPlus for {run_name} on {dataset_name}...")
                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        eval_outputs[dataset_name] = result.stdout.replace("\n", " ")
                        print(eval_outputs[dataset_name])
                    except subprocess.CalledProcessError as e:
                        eval_outputs[dataset_name] = f"EvalPlus failed: {e.stderr}"
                        print(eval_outputs[dataset_name])

                sweep_results.append({
                    "r": r,
                    "alpha": alpha,
                    "target_modules": ",".join(target_modules),
                    "save_head": save_head,
                    "run_dir": str(run_dir),
                    **{f"eval_{d}": eval_outputs[d] for d in eval_datasets}
                })

# -------------------------------
# 7. Save Sweep Results CSV
# -------------------------------
csv_file = output_dir / "sweep_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
    writer.writeheader()
    writer.writerows(sweep_results)

print(f"\nSweep finished! Results saved to {csv_file}")

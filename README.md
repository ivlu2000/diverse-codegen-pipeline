# 🧠 Diverse Synthetic Code Generation Pipeline

## 📖 Overview

This repository contains the official implementation of the paper:
**“Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks” (NeurIPS 2025 Workshop)**.

We provide the full data generation pipeline, training scripts, and evaluation setup needed to reproduce the results.
Our pipeline produces nearly **800k instruction–reasoning–code–test quadruplets**, and fine-tuning **Phi-2 (2.7B)** and **CodeGemma-2B** on this dataset yields consistent improvements on **HumanEval** and **MBPP**.

---

## ⚙️ Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for dependency management.

Following command is for HPC environments or Linux machines with NVidia GPUs.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
```

For systems without NVidia GPUs, you can use the following command to install the dependencies:

```bash
uv sync
```

However, you wont have access to packages like `faiss-gpu` and `bitsandbytes` which are required for the training.

---

## 0️⃣ Note
Typically all all sh or bash scripts have been written to be run on the HPC cluster.
If you have a gpu locally, you can run the python scripts directly and dont have to use the slurm scripts.

## 1️⃣ Dataset Curation & Expansion

Collect curated seed problems (\~40k tasks):

* [LeetCode dataset](https://huggingface.co/datasets/greengerong/leetcode)
* Codeforces, AtCoder, Advent of Code, CodeNet, etc.

Scripts:

* `scripts/seed_data_collection/*`
* Example:

  ```bash
  uv run code_generation/scripts/seed_data_collection/hugging_face/leetcode.py
  uv run code_generation/scripts/seed_data_collection/filter_tasks.py
  ```

This results in a dataset of 40k tasks in the `code_generation/datasets/` folder.

---

## 2️⃣ Relevance Filtering with FastText

Train a classifier to filter **DCLM-Baseline (\~3B docs)** into coding-relevant subsets (\~4M docs).

```bash
uv run code_generation/scripts/fasttext/train_model.py
sbatch code_generation/scripts/slurm/dclm_baseline.bash
```

Dataset available: [Filtered DCLM-Baseline](https://huggingface.co/datasets/ivlu2000/dclm-baseline-fasttext)

---

## 3️⃣ Structuring into Instruction–Reasoning–Solution–Test Quadruplets

Use **Qwen2.5-Coder-7B-Instruct** + `vLLM` to transform raw problems into structured quadruplets.

Desired format:

```json
{"instruction": "...", "reasoning": "...", "solution": "...", "test_cases": "..."}
```

For that an apptainer image is needed. You can build it using the following command:

```bash
apptainer pull python_3.11.sif docker://python:3.11
```


Run with offsets in parallel:

```bash
sbatch scripts/slurm/llm_refinement.bash 0 100
```

Result: \~220k validated quadruplets.
Dataset available: [Refined Dataset](https://huggingface.co/datasets/ivlu2000/dclm-refined-tested)

---

## 4️⃣ Execution-Based Validation

Generated solutions are executed in **isolated Apptainer containers** with time/memory limits. Only passing solutions are retained.

This ensures reasoning, solution, and test cases remain consistent.

---

## 5️⃣ Evolutionary Expansion with Genetic-Instruct

Enhance diversity with mutation and crossover:

```bash
sbatch code_generation/scripts/genetic_instruct/sbatch_codegen_colony.sh
sbatch code_generation/scripts/genetic_instruct/file_ops/sbatch_deduplication.sh
```

Still have to merge them right? #TODO

Final dataset size: \~800k unique quadruplets.

---

## 6️⃣ Fine-Tuning

We fine-tune **Phi-2 (2.7B)**  for 10 epochs with QLoRA (r=16, α=16, \[q\_proj, v\_proj, k\_proj, dense]).

```bash
sbatch code_generation/scripts/slurm/finetune_model_phi2.bash 25000
```

Runs on a single **A100 80GB GPU** (\~12h for 25k samples).
Model checkpoints saved under `code_generation/models/`.

The results are stored in a csv file called `evaluation_results.csv` in the sepcified output path.

To parse the results, you can use the following script:

```bash
python code_generation/scripts/instruction_tuning/parse_eval_results.py  output_path/evaluation_results.csv
```


For the other experiments, we fine-tune **CodeGemma-2B** and **Phi-2** for 1 epoch with QLoRA using different configurations and take the best performing model.
The configurations are the combinations of the following:

```
r_values = [8, 16]
alpha_factors = [2]
target_module_sets = [
    ["q_proj", "v_proj"],
    ["q_proj", "v_proj", "k_proj"],
]
save_head_options = [False, True]
```

To run the experiments, you can use the following script:

```bash
sbatch code_generation/scripts/slurm/finetune_model_codegemma.bash 5000 amal-abed/combined_dataset
```

For the Phi-2 experiments, you can use the following script:

```bash
sbatch code_generation/scripts/slurm/finetune_model_phi2_one_epoch.bash 5000 amal-abed/combined_dataset
```

We also finetuned **Phi-2** with other datasets. The datasets sepcified in the paper are from EpiCoder, SelfCodeAlign and our homogenous dataset.
Those are their names:
 * microsoft/EpiCoder-func-380k
 * bigcode/self-oss-instruct-sc2-exec-filter-50k
 * amal-abed/5k-subset-instructions

To run the experiments, you can use the following script:

```bash
sbatch code_generation/scripts/slurm/finetune_model_phi2_one_epoch.bash 5000 microsoft/EpiCoder-func-380k
sbatch code_generation/scripts/slurm/finetune_model_phi2_one_epoch.bash 5000 bigcode/self-oss-instruct-sc2-exec-filter-50k
sbatch code_generation/scripts/slurm/finetune_model_phi2_one_epoch.bash 5000 amal-abed/5k-subset-instructions
```

All results are stored in the `code_generation/models/` folder.
The results are stored in a csv file called `sweep_results.csv` in the sepcified output path.
You can parse the results using the following script:
```bash
python code_generation/scripts/instruction_tuning/parse_eval_results.py  output_path/sweep_results.csv
```

---

## 7️⃣ Evaluation
Benchmarks: **HumanEval** & **MBPP** using **EvalPlus**.
A local model or a huggingface model can also evaluated using this scrip:

```bash
sbatch benchmarks/sbatch_evalplus.sh
```

Results stored in `evalplus_results/`.

Expected performance (Phi-2):

* Base: 45.7% → Fine-tuned (25k samples): 56.1% on HumanEval
* Base: 62.7% → Fine-tuned (25k samples): 65.6% on MBPP

---

## 📊 Results Summary

| Model                 | HumanEval Base | HumanEval+ | MBPP Base | MBPP+    |
| --------------------- | -------------- | ---------- | --------- | -------- |
| Phi-2 (Base)          | 45.7           | 40.9       | 62.7      | 51.6     |
| Phi-2 + LeetCode      | 47.6           | 42.1       | 63.0      | 51.6     |
| Phi-2 + 25k synthetic | **56.1**       | **51.8**   | **65.6**  | **55.3** |


---

## 📂 Resources

* [Final 800k Dataset](https://huggingface.co/datasets/amal-abed/combined_dataset)
* [Filtered DCLM-Baseline](https://huggingface.co/datasets/ivlu2000/dclm-baseline-fasttext)
* [Refined Dataset](https://huggingface.co/datasets/ivlu2000/dclm-refined-tested)


import os
import json
import glob
import random
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO
import traceback
import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from functools import lru_cache

@lru_cache(maxsize=None)
def load_prompt_cached(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from src.shared.llm import LLMClient
import vllm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR = SCRIPT_DIR / "apptainer_runs"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Start index of the dataset slice")
parser.add_argument("--end", type=int, default=200000, help="End index of the dataset slice")
parser.add_argument("--colony-id", type=int, required=True, help="Unique colony ID (e.g., 0, 1, 2, 3)")
args = parser.parse_args()

NUM_SAMPLES = 50000
BATCH_SIZE = 128
SCRIPT_DIR = Path(__file__).resolve().parent
APPTAINER_IMAGE = SCRIPT_DIR / "apptainer" / "python.sif"

llm = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="no-key-needed",
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
load_prompt = lambda path: open(path, "r", encoding="utf-8").read().strip()

cross_over_prompt = load_prompt_cached(os.path.join(current_dir, "prompts/cross_over.txt"))
mutation_prompt = load_prompt_cached(os.path.join(current_dir, "prompts/mutation/mutation.txt"))
codegen_prompt = load_prompt_cached(os.path.join(current_dir, "prompts/code_generation.txt"))
judge_prompt = load_prompt_cached(os.path.join(current_dir, "prompts/fitness.txt"))

mutation_ops = {
    os.path.splitext(os.path.basename(fp))[0]: load_prompt_cached(fp)
    for fp in glob.glob(os.path.join(current_dir, "prompts/mutation/operations/*.txt"))
}

task_ids = {0: "cross-over", 1: "mutation"}
task_prob = [0.5, 0.5]

def encode_inst_prompts(seed_insts, task_id):
    if task_id == 0:
        subset = random.sample(seed_insts, min(5, len(seed_insts)))
        tasks = "\n\n".join(f"{i+1}.\n{inst['instruction']}" for i, inst in enumerate(subset))
        return [cross_over_prompt.format(tasks=tasks)]
    else:
        inst = random.choice(seed_insts)["instruction"]
        _, method_text = random.choice(list(mutation_ops.items()))
        return [mutation_prompt.format(method=method_text, instruction=inst)]

def parse_inst(task_id, resp):
    instruction_match = re.search(r'<instruction>(.*?)</instruction>', resp.strip(), re.DOTALL | re.IGNORECASE)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', resp.strip(), re.DOTALL | re.IGNORECASE)

    if instruction_match and reasoning_match:
        return [{
            "task_id": task_id,
            "instruction": instruction_match.group(1).strip(),
            "reasoning": reasoning_match.group(1).strip()
        }]

    return []

def make_codegen(insts):
    return [codegen_prompt.format(instruction=i, reasoning="N/A") for i in insts]

def format_few_shots(seed_insts, n=5):
    few_shots = random.sample(seed_insts, min(n, len(seed_insts)))
    formatted = []
    for i, fs in enumerate(few_shots, 1):
        if 'instruction' in fs and 'code' in fs:
            example_text = (
                f"Sample {i}:\n"
                f"Instruction:\n{fs['instruction']}\n\n"
                f"<code>\n{fs['code']}\n</code>\n\n"
                f"Decision: \\boxed{{Yes}}"
            )
            formatted.append(example_text)
    return "\n\n".join(formatted)

def make_judge(pairs, seed_insts):
    few_shot_examples = format_few_shots(seed_insts)
    return [
        judge_prompt.format(
            instruction=p['instruction'],
            code=p['solution'],
            reasoning=p['reasoning'],
            few_shot_examples=few_shot_examples
        )
        for p in pairs
    ]

def write_parquet(data, path_prefix):
    path = f"{path_prefix}_colony_{args.colony_id}.parquet"
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    print(f"Saved {len(data)} samples to {path}")

def generate_batches(llm, prompts):
    outs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        batch_responses = llm.batched_inference(
            batch,
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            num_concurrent=len(batch)
        )
        # Assuming batch_responses is a list of strings (the generated texts)
        outs.extend(batch_responses)
    return outs

def generate_batches_parallel(llm, prompts, batch_size=BATCH_SIZE, max_workers=8):
    outs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            futures.append(executor.submit(
                llm.batched_inference,
                batch,
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                num_concurrent=len(batch)
            ))
        for future in futures:
            outs.extend(future.result())
    return outs

def extract_solution_tests(generated_code):
    pattern = re.compile(
        r"<solution_(\d+)>\s*```python(.*?)```\s*</solution_\1>\s*"
        r"<test_\1>\s*```python(.*?)```\s*</test_\1>",
        re.DOTALL
    )
    return pattern.findall(generated_code)

def run_code_in_apptainer(code_str):
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=str(TEMP_DIR), delete=False
    )
    try:
        tmp_file.write(code_str)
        tmp_file.flush()
        tmp_filename = tmp_file.name

        cmd = [
            "apptainer", "exec", "--nv",
            "--pwd", str(TEMP_DIR),
            APPTAINER_IMAGE,
            "python3", tmp_filename
        ]

        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            text=True
        )
        success = (completed.returncode == 0)
        return success, completed.stdout

    except Exception as e:
        return False, f"Exception running apptainer: {e}"

    finally:
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

def test_and_select_solution(generated_code):
    candidates = extract_solution_tests(generated_code)
    for idx, (num, solution_code, test_code) in enumerate(candidates, 1):
        combined_code = f"{solution_code}\n\n{test_code}"
        success, output = run_code_in_apptainer(combined_code)
        if success:
            print(f"Solution {idx} passed inside container.")
            return {
                "solution_number": idx,
                "solution_code": solution_code.strip(),
                "test_code": test_code.strip()
            }
        else:
            print(f"Solution {idx} failed inside container:\n{output}")
    print("No valid solution passed the containerized tests.")
    return None

def parallel_test_and_select_batch(code_resps, max_workers=64):
    results = [None] * len(code_resps)

    def wrapper(i, code):
        return i, test_and_select_solution(code)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(wrapper, i, code) for i, code in enumerate(code_resps)]
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

    return results

def generate_samples(llm, seed, samples):

    generated_samples = []
    all_samples = list(seed)
    pbar = tqdm(total=NUM_SAMPLES, desc="Producing samples")

    while len(generated_samples) < NUM_SAMPLES:
        inst_prompts = []
        inst_task_ids = []
        for _ in range(BATCH_SIZE):
            task_id = np.random.choice(list(task_ids), p=task_prob)
            prompt = encode_inst_prompts(all_samples, task_id)[0]
            inst_prompts.append(prompt)
            inst_task_ids.append(task_id)

        inst_resps = generate_batches_parallel(llm, inst_prompts)

        new_insts, new_reasons, new_inst_task_ids = [], [], []
        for resp, task_id in zip(inst_resps, inst_task_ids):
            parsed = parse_inst(task_id, resp)
            for item in parsed:
                new_insts.append(item["instruction"])
                new_reasons.append(item["reasoning"])
                new_inst_task_ids.append(task_id)

        if not new_insts:
            continue

        code_prompts = make_codegen(new_insts)
        code_resps = generate_batches_parallel(llm, code_prompts)

        selected_batch = parallel_test_and_select_batch(code_resps)

        valid_samples = []
        for i, selected in enumerate(selected_batch):
            if selected:
                valid_samples.append({
                    "instruction": new_insts[i],
                    "reasoning": new_reasons[i],
                    "solution": selected["solution_code"],
                    "tests": selected["test_code"],
                    "task_type": task_ids[new_inst_task_ids[i]]
                })

        if valid_samples:
            judge_prompts = make_judge(valid_samples, all_samples)
            judge_responses = generate_batches_parallel(llm, judge_prompts)

            for i, response in enumerate(judge_responses):
                if "yes" in response.strip().lower():
                    passed_sample = valid_samples[i]
                    generated_samples.append(passed_sample)
                    all_samples.append(passed_sample)
                    write_parquet(generated_samples, "checkpoints/checkpoint_generated_instructions")

        pbar.n = len(generated_samples)
        pbar.last_print_n = len(generated_samples)
        pbar.refresh()

    pbar.close()
    write_parquet(generated_samples, "checkpoints/final_generated_instructions")
    return generated_samples


def main():
    dataset = load_dataset("amal-abed/test2", split="train")
    subset = dataset.select(range(args.start, args.end))

    seed = [{"instruction": item["instruction"]} for item in subset if "instruction" in item]

    samples = [
        {
            "instruction": item["instruction"],
            "code": item.get("code", "<missing code>"),
            "reasoning": item.get("reasoning", "No reasoning provided."),
        }
        for item in subset if "instruction" in item
    ]

    generate_samples(llm, seed, samples)
    print("Done generating all distinct samples")

if __name__ == "__main__":
    main()

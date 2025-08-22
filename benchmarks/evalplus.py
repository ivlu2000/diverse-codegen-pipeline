import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run evalplus.evaluate with dynamic dataset input.")
parser.add_argument("--dataset", required=True, help="Specify the dataset name")
args = parser.parse_args()

model = "Qwen2.5-Coder-32B-Instruct"
base_url = "http://localhost:8090/v1"
backend = "openai"

command = [
    "evalplus.evaluate",
    "--model", model,
    "--dataset", args.dataset,
    "--base-url", base_url,
    "--backend", backend,
    "--greedy"
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
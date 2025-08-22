from datasets import load_dataset
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ivlu2000/dclm-refined", 
                   help="Dataset to load")
parser.add_argument("--num_samples", type=int, default=5000, 
                   help="Number of samples in the final dataset")
parser.add_argument("--output_path", type=str, default=None,
                   help="Path to save the final dataset")
args = parser.parse_args()

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

seed = 42
random.seed(seed)

# Load dataset
print(f"Loading dataset: {args.dataset}")
dataset = load_dataset(args.dataset, split="train")

# Check if the dataset has a "code" column and rename it to "solution" if it exists
if "code" in dataset.column_names or "output" in dataset.column_names or "response" in dataset.column_names:
    print("Found 'code', 'output', or 'response' column in dataset, renaming to 'solution'")
    dataset = dataset.rename_column("code" if "code" in dataset.column_names else "output" if "output" in dataset.column_names else "response", "solution")

if "content" in dataset.column_names and "python" in dataset.column_names:
    print("Found 'content' and 'python' columns in dataset, renaming to 'instruction' and 'solution'")
    dataset = dataset.rename_column("content", "instruction")
    dataset = dataset.rename_column("python", "solution")

dataset = dataset.select_columns(["instruction", "solution"])

# Sample from dataset
final_dataset = dataset.shuffle(seed=seed).select(range(args.num_samples))

# Create filename
dataset_name = args.dataset.split('/')[-1]
output_filename = f"final_dataset_{dataset_name}_{args.num_samples}k.jsonl"

# Save to json line format
output_path = args.output_path if args.output_path else os.path.join(root_dir, "datasets", output_filename)
final_dataset.to_json(output_path, orient="records", lines=True)

print(f"Dataset saved to: {output_path}")
print(f"Total samples: {len(final_dataset)}")


# This dataset contains coding questions from
# - Codeforces
# - Codechef
# - Aizu
# - Atcoder
# - HackerEarth

import os
import pandas as pd
import dask.dataframe as dd


splits = {
    "train": "data/train-*-of-*.parquet",
    "test": "data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
    "valid": "data/valid-00000-of-00001-5e672c5751f060d3.parquet",
}
df = dd.read_parquet("hf://datasets/deepmind/code_contests/" + splits["train"])

# Extract unique tasks using Dask
tasks = df["description"].unique().compute()

print("Size of DeepMind Code Contest dataset: ", len(tasks))

# Print the tasks
print(tasks[0])
print(len(tasks))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


tasks_path = os.path.join(root_dir, "datasets/tasks.parquet")

# Handle existing tasks file using Dask
if os.path.exists(tasks_path):
    # Read existing tasks
    existing_df = dd.read_parquet(tasks_path)
    existing_tasks = existing_df["task"].compute().values

    # Check if new tasks need to be added
    new_tasks = [task for task in tasks if task not in existing_tasks]
    if new_tasks:
        # Create new dataframe with only the new tasks
        new_df = pd.DataFrame({"task": new_tasks})
        # Convert to parquet and append to existing file
        new_df.to_parquet(tasks_path, append=True, engine="fastparquet")
else:
    # Create a new Dask dataframe with the tasks
    new_df = dd.from_pandas(pd.DataFrame({"task": tasks}), npartitions=1)
    new_df.to_parquet(tasks_path)

# Load and print the final result
final_df = dd.read_parquet(tasks_path)
print("Size of combined dataset: ", len(final_df.compute()))

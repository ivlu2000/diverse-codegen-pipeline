import os
import pandas as pd
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("MatrixStudio/Codeforces-Python-Submissions")

# Extract unique tasks
tasks_train = list(set(ds["train"]["prompt"]))
tasks_test = list(set(ds["test"]["prompt"]))

unique_tasks = list(set(tasks_train + tasks_test))

print("Size of Codeforces dataset: ", len(unique_tasks))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# check if tasks.parquet exists
if os.path.exists(os.path.join(root_dir, "datasets/tasks.parquet")):
    # load the parquet file
    df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))
    # check if the tasks are already in the dataframe
    if not all(task in df["task"].values for task in unique_tasks):
        # append the new tasks to the dataframe
        df = pd.concat([df, pd.DataFrame({"task": unique_tasks})])
else:
    # create a pandas dataframe with the tasks
    df = pd.DataFrame({"task": unique_tasks})

# save the dataframe to a parquet file
df.to_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# load the parquet file
df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# print the dataframe
print("Size of combined dataset: ", len(df))

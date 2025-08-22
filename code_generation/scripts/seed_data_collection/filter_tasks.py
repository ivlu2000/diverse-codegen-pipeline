# load the tasks.parquet file and keep only unique tasks
import os
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))
df = df.drop_duplicates(subset="task")
df.to_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# load the tasks.parquet file
df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# print the dataframe
print(len(df))

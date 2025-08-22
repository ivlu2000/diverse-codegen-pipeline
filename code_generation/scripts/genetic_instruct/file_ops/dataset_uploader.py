from datasets import Dataset
import pandas as pd

df = pd.read_parquet("final_deduplicated_samples_MiniLM.parquet")
ds = Dataset.from_pandas(df)

ds.push_to_hub("amal-abed/combined_dataset")

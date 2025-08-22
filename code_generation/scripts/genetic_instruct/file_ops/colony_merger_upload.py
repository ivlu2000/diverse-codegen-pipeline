import os
import glob
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

# ---------- Paths & IDs ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "checkpoints"))
CHECKPOINT_GLOB = os.path.join(CHECKPOINT_DIR, "final_generated_instructions_colony_*.parquet")

HUB_DATASET_ID = "amal-abed/test2"
SPLIT_NAME = "train"

OUTPUT_PARQUET = os.path.join(CHECKPOINT_DIR, "combined_dataset.parquet")
OUTPUT_MERGED_LOCAL = os.path.join(CHECKPOINT_DIR, "final_samples.parquet")

# ---------- Step 1: Merge checkpoint parquet files locally ----------
checkpoint_files = sorted(glob.glob(CHECKPOINT_GLOB))
if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files matched: {CHECKPOINT_GLOB}")

all_dfs = [pd.read_parquet(f) for f in checkpoint_files]
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_parquet(OUTPUT_MERGED_LOCAL, index=False)
print(f"Merged {len(checkpoint_files)} files into {OUTPUT_MERGED_LOCAL} with {len(merged_df)} samples")

# ---------- Step 2: Load merged checkpoint dataset as Hugging Face Dataset ----------
ds_checkpoint = Dataset.from_pandas(merged_df, preserve_index=False)

# ---------- Step 3: Load existing dataset from HF Hub ----------
ds_seed = load_dataset(HUB_DATASET_ID, split=SPLIT_NAME)

# ---------- Step 4: Align columns for concatenation ----------
def _align_union(a, b):
    union = sorted(set(a.column_names) | set(b.column_names))

    def add_missing(ds):
        missing = [c for c in union if c not in ds.column_names]
        if missing:
            def _add(batch):
                n = len(next(iter(batch.values()))) if batch else 0
                for m in missing:
                    batch[m] = [None] * n
                return batch
            ds = ds.map(_add, batched=True)
        return ds.select_columns(union)

    return add_missing(a), add_missing(b)

ds_checkpoint, ds_seed = _align_union(ds_checkpoint, ds_seed)

# ---------- Step 5: Concatenate ----------
combined_dataset = concatenate_datasets([ds_checkpoint, ds_seed])

# ---------- Step 6: Save locally ----------
combined_dataset.to_parquet(OUTPUT_PARQUET)
print(f"Merged dataset saved as: {os.path.abspath(OUTPUT_PARQUET)}")
print(f"Total samples after merge: {len(combined_dataset)}")

# ---------- Step 7: Push merged dataset back to the SAME Hub repo ----------
combined_dataset.push_to_hub(HUB_DATASET_ID, split=SPLIT_NAME)
print(f"Final merged dataset pushed to: {HUB_DATASET_ID} (split='{SPLIT_NAME}')")

# ---------- Step 8: Clean up checkpoint files ----------
for f in checkpoint_files:
    try:
        os.remove(f)
        print(f"Deleted checkpoint file: {f}")
    except Exception as e:
        print(f"Warning: could not delete {f} ({e})")

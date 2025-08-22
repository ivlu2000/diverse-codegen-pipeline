import pandas as pd
import glob

checkpoint_files = sorted(glob.glob("/pfs/work9/workspace/scratch/fr_aa502-code_eval/synthetic_reasoning/code_generation/src/methods/genetic_instruct/checkpoints/checkpoint_generated_instructions_colony_*.parquet"))

all_dfs = []

for file in checkpoint_files:
    df = pd.read_parquet(file)
    all_dfs.append(df)

merged_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

merged_df.to_parquet("final_samples.parquet", index=False)

print(f"✅ Merged {len(checkpoint_files)} files into final_samples.parquet with {len(merged_df)} samples")


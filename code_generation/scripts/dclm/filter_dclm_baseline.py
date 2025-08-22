import os
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
import fasttext_parallel as ft


# Global model variable to ensure it's loaded only once
_model = None


def get_model(model_path):
    global _model
    if _model is None:
        # Load the model only once
        _model = ft.load_model(model_path)
    return _model


def main():
    # Determine root directory and set paths

    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    parquet_path = os.path.join(root_dir, "datasets/dclm_baseline_1.0.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    # Load or create the dataset with only the needed "text" column
    start = time.time()
    if not os.path.exists(parquet_path):
        print("Loading dataset from Hugging Face")
        df = dd.read_parquet(
            "hf://datasets/mlfoundations/dclm-baseline-1.0-parquet/**/*_train/**",
        )
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        print(f"Dataset saved in {time.time() - start:.2f}s")
    else:
        print("Loading local Parquet dataset")
        df = dd.read_parquet(parquet_path, engine="pyarrow", columns=["text"])
        print(f"Dataset loaded in {time.time() - start:.2f}s")

    # Create output directory
    output_dir = os.path.join(root_dir, "datasets/fasttext/dclm/predictions")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(root_dir, "models/fasttext_model.bin")

    def process_partition(df_partition, index=None, partition_info=None):
        part_start = time.time()
        partition_number = partition_info["number"] + index
        print("Partition info", partition_info)
        print(f"Processing partition number {partition_number}")

        # Load model for this worker
        model = get_model(model_path)

        # Process in smaller chunks to reduce memory usage
        chunk_size = 1000  # Smaller than current batch_size
        total_rows = len(df_partition)
        filtered_data = []

        # Process in chunks to avoid loading all texts into memory at once
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = df_partition.iloc[chunk_start:chunk_end]

            # Process text directly without creating an intermediate list
            texts = chunk["text"].str.replace("\n", " ", regex=False)

            labels, scores = model.batch(texts.tolist(), k=1)

            # Filter and append results directly
            for i, (label, score) in enumerate(zip(labels, scores)):
                if label[0] == 1 and score[0] > 0.9:
                    filtered_data.append(
                        {"score": score[0], "label": label[0], "text": texts.iloc[i]}
                    )

        # Create DataFrame only after all processing is done
        res_df = pd.DataFrame(filtered_data)
        positive_samples = len(filtered_data)

        # Save this partition's results immediately
        part_output_path = os.path.join(
            output_dir,
            f"part-{partition_number}.parquet",
        )
        res_df.to_parquet(part_output_path, engine="pyarrow", compression="snappy")

        print(
            f"Processed partition number {partition_number} with {total_rows} rows in {time.time() - part_start:.2f}s"
        )
        print(
            f"Found {len(res_df)} positive samples in partition number {partition_number}"
        )

        return pd.DataFrame(
            {
                "processed_rows": [total_rows],
                "positive_samples": [positive_samples],
            }
        )

    # Process each partition in parallel using Dask distributed
    meta = pd.DataFrame(
        {
            "processed_rows": pd.Series([], dtype=np.int64),
            "positive_samples": pd.Series([], dtype=np.int64),
        }
    )

    # print number of partitions
    number_of_partitions = df.npartitions
    step_size = 300
    print("Number of partitions", number_of_partitions)

    dfs = [
        df.partitions[i : i + step_size]
        for i in range(0, number_of_partitions, step_size)
    ]

    index = 0
    summaries = dd.DataFrame({"processed_rows": [], "positive_samples": []})
    for df in dfs:
        summary = df.map_partitions(
            lambda df, partition_info: process_partition(df, index, partition_info),
            meta=meta,
        ).compute(scheduler="processes")
        index += step_size
        summaries = dd.concat([summaries, summary])

    # sum up columns
    summaries = summaries.sum().compute()
    final_summary = pd.DataFrame(summaries)

    # Save the summary
    summary_path = os.path.join(output_dir, "summary.parquet")
    final_summary.to_parquet(summary_path, engine="pyarrow", compression="snappy")

    # Print overall statistics
    total_processed = summaries["processed_rows"].sum()
    total_positive = summaries["positive_samples"].sum()
    print(f"Total processed: {total_processed} rows")
    print(f"Total positive samples: {total_positive}")


if __name__ == "__main__":
    main()

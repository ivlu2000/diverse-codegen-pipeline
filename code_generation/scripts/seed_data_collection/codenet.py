import os
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_text_from_html(html_path):
    """Extract text content from HTML file."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text and clean it up
        text = soup.get_text(separator=" ").strip()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text


def process_problem_descriptions(input_dir, output_file):
    """Process all HTML files in the input directory and save as parquet."""
    # Get all HTML files
    html_files = list(Path(input_dir).glob("**/*.html"))

    print(f"Processing {len(html_files)} HTML files")

    # Process each file
    data = []
    for html_path in tqdm(html_files, desc="Processing HTML files"):
        problem_id = html_path.stem  # Get filename without extension
        text_content = extract_text_from_html(html_path)
        data.append({"problem_id": problem_id, "description": text_content})

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(data)
    tasks = df["description"]
    print("Size of CodeNet dataset: ", len(tasks))
    if os.path.exists(output_file):
        # load the parquet file
        df = pd.read_parquet(output_file)
        # check if the tasks are already in the dataframe
        if not all(task in df["task"].values for task in tasks):
            # append the new tasks to the dataframe
            df = pd.concat([df, pd.DataFrame({"task": tasks})])
    else:
        # create a pandas dataframe with the tasks
        df = pd.DataFrame({"task": tasks})

    df.to_parquet(output_file, index=False)
    print(f"Saved {len(data)} problem descriptions to {output_file}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    input_directory = os.path.join(root_dir, "datasets/codenet/problem_descriptions")
    output_file = os.path.join(root_dir, "datasets/tasks.parquet")
    process_problem_descriptions(input_directory, output_file)
    # load the parquet file and print the first 5 rows
    df = pd.read_parquet(output_file)
    print("Size of combined dataset: ", len(df))

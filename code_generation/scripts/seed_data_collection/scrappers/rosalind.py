import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Base URL of the Rosalind website
base_url = "https://rosalind.info"

tasks = []

# URL of the Rosalind problem list page
problem_list_url = f"{base_url}/problems/list-view/"

# Send a GET request to fetch the problem list page content
response = requests.get(problem_list_url)
response.raise_for_status()  # Ensure the request was successful

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Find the table containing the problems
table = soup.find("table", {"class": "problem-list"})


# Iterate over each table row (tr) in the table body
for row in table.tbody.find_all("tr"):
    # Find the table data (td) cell containing the problem title
    title_cell = row.find("a", {"class": ["not-accessible", "accessible"]})
    if title_cell:
        # Extract the problem title
        title = title_cell.text.strip()
        # Extract the relative URL and construct the full URL
        relative_url = title_cell["href"]
        problem_url = f"{base_url}{relative_url}"
        print(f"Title: {title}")
        print(f"URL: {problem_url}")

        # Fetch the problem page to extract the description
        problem_response = requests.get(problem_url)
        problem_response.raise_for_status()
        problem_soup = BeautifulSoup(problem_response.text, "html.parser")
        # Find the div containing the problem content
        problem_content = problem_soup.find("div", {"class": "problem-statement"})
        if problem_content:
            # Extract the text content of the problem description
            description = problem_content.get_text(separator="\n").strip()
            print("Description:")
            print(description)
            tasks.append(f"{title}\nDescription: \n{description}")
        else:
            print("Description not found.")
        print("-" * 80)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Size of Rosalind dataset: ", len(tasks))

# check if tasks.parquet exists
if os.path.exists(os.path.join(root_dir, "datasets/tasks.parquet")):
    # load the parquet file
    df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))
    # check if the tasks are already in the dataframe
    if not all(task in df["task"].values for task in tasks):
        # append the new tasks to the dataframe
        df = pd.concat([df, pd.DataFrame({"task": tasks})])
else:
    # create a pandas dataframe with the tasks
    df = pd.DataFrame({"task": tasks})

# save the dataframe to a parquet file
df.to_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# load the parquet file
df = pd.read_parquet(os.path.join(root_dir, "datasets/tasks.parquet"))

# print the dataframe
print("Size of combined dataset: ", len(df))

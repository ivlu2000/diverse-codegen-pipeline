import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

PAGES = 19
# URL of the Project Euler archives page
tasks = []
for page in range(1, PAGES + 1):
    url = f"https://projecteuler.net/archives;page={page}"

    # Send a GET request to fetch the page content
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table containing the problems
    table = soup.find("table", {"id": "problems_table"})

    # Iterate over each table row (tr) in the table body
    for row in table.find_all("tr"):
        # Find all table data (td) cells in the row
        cells = row.find_all("td")
        if len(cells) >= 2:
            # Extract the problem ID and title
            problem_id = cells[0].text.strip()
            title = cells[1].text.strip()
            # Construct the URL to the problem
            problem_url = f"https://projecteuler.net/problem={problem_id}"
            print(f"Problem {problem_id}: {title}")
            print(f"URL: {problem_url}\n")
            response = requests.get(problem_url)
            response.raise_for_status()  # Ensure the request was successful

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the div containing the problem content
            problem_content = soup.find("div", {"class": "problem_content"})

            # Extract the text content without HTML tags
            if problem_content:
                problem_text = problem_content.get_text(separator=" ", strip=True)
                # remove dollar signs
                problem_text = problem_text.replace("$", "")
                tasks.append(problem_text)
            else:
                print("Problem content not found.")

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Size of Project Euler dataset: ", len(tasks))

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

import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd

# Base URL of the Kattis website
BASE_URL = "https://open.kattis.com"

NUM_PAGES = 47
# URL of the Kattis problem list page
PROBLEM_LIST_URL = f"{BASE_URL}/problems?f_language=en&show_more_filters=off&f_min_difficulty=&f_max_difficulty=&order=-1"
tasks = []


# Function to fetch problem list page
def fetch_problem_list(url, page):
    response = requests.get(f"{url}&page={page}")
    response.raise_for_status()
    return response.text


# Function to fetch individual problem page
def fetch_problem_description(problem_url):
    response = requests.get(problem_url)
    response.raise_for_status()
    return response.text


# Function to parse problem list and extract problem titles and URLs
def parse_problem_list(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # find all a tags which begin with /problems/ but not end with /statistics
    problem_rows = soup.find_all("a", {"href": re.compile(r"^/problems/[^/]+$")})
    problems = []
    for row in problem_rows:
        title = row.text.strip()
        relative_url = row["href"]
        problem_url = f"{BASE_URL}{relative_url}"
        problems.append((title, problem_url))
    return problems


# Function to parse problem description
def parse_problem_description(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    problem_body = soup.find("div", {"class": "problembody"})
    if not problem_body:
        return "Description not found."

    # Process each element while preserving markdown-like formatting
    formatted_text = []
    for element in problem_body.descendants:
        if element.name == "h1":
            formatted_text.append(f"# {element.get_text().strip()}\n")
        elif element.name == "h2":
            formatted_text.append(f"## {element.get_text().strip()}\n")
        elif element.name == "h3":
            formatted_text.append(f"### {element.get_text().strip()}\n")
        elif element.name == "p":
            formatted_text.append(f"{element.get_text().strip()}\n")
        elif element.name == "th":
            formatted_text.append(f"**{element.get_text().strip()}**\n")
        elif element.name == "pre":
            # Preserve code blocks with proper formatting
            formatted_text.append(f"```\n{element.get_text().strip()}\n```\n")
        elif element.name == "ul" or element.name == "ol":
            for li in element.find_all("li"):
                formatted_text.append(f"* {li.get_text().strip()}\n")

    return "\n".join(formatted_text)


# Main script
if __name__ == "__main__":
    # Fetch and parse problem list page
    for page in range(1, NUM_PAGES + 1):
        list_html = fetch_problem_list(PROBLEM_LIST_URL, page)
        problems = parse_problem_list(list_html)

        for title, problem_url in problems:
            print(f"Fetching problem: {title}")
            try:
                # Fetch and parse individual problem page
                problem_html = fetch_problem_description(problem_url)
                description = parse_problem_description(problem_html)
                print(f"Page: {page}")
                print(f"Title: {title}")
                print(f"URL: {problem_url}")
                print("Description:")
                print(description)
                print("-" * 80)
                tasks.append(f"{title}\nDescription: \n{description}")

            except requests.HTTPError as e:
                print(f"Failed to fetch problem {title}: {e}")

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Size of Kattis dataset: ", len(tasks))

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

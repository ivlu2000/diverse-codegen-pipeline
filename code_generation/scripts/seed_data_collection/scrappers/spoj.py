import requests
from bs4 import BeautifulSoup
import os
import pandas as pd

# Base URLs
base_url = "https://www.spoj.com"
spoj_problem_categories = [
    {
        "name": "classical",
        "url": f"{base_url}/problems/classical/sort=0,start=",
        "end": 3950,
    },
    {
        "name": "challenge",
        "url": f"{base_url}/problems/challenge/sort=0,start=",
        "end": 150,
    },
    {
        "name": "partial",
        "url": f"{base_url}/problems/partial/sort=0,start=",
        "end": 150,
    },
    {"name": "riddle", "url": f"{base_url}/problems/riddle/sort=0,start=", "end": 50},
    {"name": "basics", "url": f"{base_url}/problems/basics/sort=0,start=", "end": 300},
]

tasks = []


# Function to fetch problem list page
def fetch_problem_list(start, category):
    url = f"{category['url']}{start}"
    response = requests.get(url)
    response.encoding = "utf-8"  # Set UTF-8 encoding
    response.raise_for_status()
    return response.text


# Function to fetch individual problem page
def fetch_problem_description(
    problem_code,
):
    url = f"{base_url}/problems/{problem_code}/"
    response = requests.get(url)
    response.encoding = "utf-8"  # Set UTF-8 encoding
    response.raise_for_status()
    return response.text


# Function to parse problem list and extract problem codes and titles
def parse_problem_list(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    problem_links = soup.select('td > a[href^="/problems/"]')
    problems = []
    for link in problem_links:
        problem_code = link["href"].split("/")[-1]
        title = link.text.strip()
        problems.append((problem_code, title))
    return problems


# Function to parse problem description
def parse_problem_description(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    problem_body = soup.find("div", {"id": "problem-body"})
    if problem_body:
        return problem_body.get_text(separator="\n").strip()
    return "Description not found."


# Main script
if __name__ == "__main__":
    for category in spoj_problem_categories:
        start = 0
        while True:
            # Fetch and parse problem list page
            list_html = fetch_problem_list(start, category)
            problems = parse_problem_list(list_html)
            if start >= category["end"]:
                break  # No more problems to process

            for problem_code, title in problems:
                print(f"Fetching problem: {title} ({problem_code}) {category['name']}")
                try:
                    # Fetch and parse individual problem page
                    problem_html = fetch_problem_description(problem_code)
                    description = parse_problem_description(problem_html)
                    print(f"Title: {title}")
                    print(f"Code: {problem_code}")
                    print("Description:")
                    print(description)
                    print("-" * 80)
                    task = f"{title} ({problem_code})\n {description}"
                    tasks.append(task)
                except requests.HTTPError as e:
                    print(f"Failed to fetch problem {problem_code}: {e}")

            start += 50  # Move to the next page of problems

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Size of SPOJ dataset: ", len(tasks))

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

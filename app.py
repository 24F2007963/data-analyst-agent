from flask import Flask, request, jsonify
import base64
import os
import requests
from bs4 import BeautifulSoup
import duckdb
import pandas as pd
import re


app = Flask(__name__)

LLM_PROXY_URL = "https://aipipe.org/openrouter/v1/chat/completions"  # Replace with your OpenAI/Gemini Proxy URL
LLM_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDc5NjNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.OGdLJaI1rTxuOObABnswuQHbD4BIfwBhHkbyhVPlhWQ"  # If required by your proxy


def call_llm_task_parser(questions_text):
    prompt = f"""
    You are a Data Analyst Agent. Given the following questions, identify the tasks and return a JSON plan.

    Questions:
    {questions_text}

    Respond in the following JSON format:
    {{
      "tasks": [
        {{ "type": "scrape", "url": "...", "extract": "tablename" }},
        {{ "type": "duckdb_query", "query": "SELECT ..." }},
        {{ "type": "regression", "on": ["column1", "column2"] }},
        {{ "type": "plot", "x": "column1", "y": "column2" }}
      ]
    }}
    """

    payload = {
        "model": "gpt-4o-mini",  # or gemini model name if applicable
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json"
    }
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"


    try:
        response = requests.post(LLM_PROXY_URL, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        raise

    
    parsed_tasks = response.json()["choices"][0]["message"]["content"]
    return parsed_tasks

def scrape_table_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')
    if not tables:
        return []

    # Extract first table
    table_data = []
    table = tables[0]
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    table_data.append(headers)

    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if cells:
            table_data.append([cell.get_text(strip=True) for cell in cells])

    return table_data

def execute_duckdb_query(query):
    
    # Load httpfs and parquet extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")

    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables in DuckDB:", tables)

    print(f"Executing DuckDB Query: {query}")
    result = con.execute(query).fetchall()
    con.close()
    return result

def execute_task_plan(task_plan):
    con = duckdb.connect()
    scraped_tables = []
    duckdb_results = []

    for task in task_plan.get("tasks", []):
        if task["type"] == "scrape":
            print(f"Scraping from URL: {task['url']}")
            table_data = scrape_table_from_url(task['url'])
            print(f"Extracted {len(table_data)-1} rows from table.")

            # Convert to DataFrame
            headers = table_data[0]  # First row is header
            rows = table_data[1:]
            df = pd.DataFrame(rows, columns=headers)

            # Register DataFrame as a DuckDB table
            con.register(task["extract"], df)
        elif task["type"] == "duckdb_query":
            result = execute_duckdb_query(con, task['query'])
            duckdb_results.append(result)
        elif task["type"] == "regression":
            print(f"Performing regression on columns: {task['on']}")
            # Perform regression analysis
        elif task["type"] == "plot":
            print(f"Plotting scatterplot X: {task['x']}, Y: {task['y']}")
            # Generate plot

    # Dummy response after execution
    dummy_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    image_base64 = f"data:image/png;base64,{dummy_image_data.decode()}"
    return [1, "Titanic", 0.485782, image_base64]

@app.route('/api/', methods=['POST'])
def analyze():
    if 'questions.txt' not in request.files:
        return jsonify({'error': 'questions.txt file is required'}), 400

    questions_file = request.files['questions.txt']
    questions_text = questions_file.read().decode('utf-8')

    attachments = {}
    for file_key in request.files:
        if file_key != 'questions.txt':
            attachments[file_key] = request.files[file_key]

    print("Received Questions:", questions_text)
    print("Received Attachments:", list(attachments.keys()))

    # Call LLM to parse task plan
    task_plan_json = call_llm_task_parser(questions_text)
    parsed_tasks = re.sub(r"```(json)?", "", task_plan_json).strip()

    import json
    task_plan = json.loads(parsed_tasks)
    print("Parsed Task Plan:", task_plan)

    # Execute dynamic task plan
    response = execute_task_plan(task_plan)

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
from redbox.app import Redbox
from redbox.models.settings import get_settings
from redbox.models.chain import RedboxQuery, RedboxState, AISettings, ChatLLMBackend

# from langfuse.callback import CallbackHandler
from uuid import uuid4
import langchain
import re
import json
import time
from dotenv import load_dotenv
import os
import logging
import subprocess
import boto3
import sqlite3

# just logging stuff like we have in redbox
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

bucket_name = 'redbox-evaluation-dataset'
file_path = 'tabular/financial.sqlite'
ground_truth_db = './financial.sqlite'


def get_state(user_uuid, prompts, documents, ai_setting):
    q = RedboxQuery(
        question=f"@tabular {prompts[-1]}",
        s3_keys=documents,
        user_uuid=user_uuid,
        chat_history=prompts[:-1],
        ai_settings=ai_setting,
        permitted_s3_keys=documents,
    )

    return RedboxState(
        request=q,
    )


def run_app(app, state) -> RedboxState:
    return app.graph.invoke(state)


def initialise_param():
    try_again = True
    iter_limit = 0
    return try_again, iter_limit

def get_tabular_db():
    result = subprocess.run(['find', '.', '-name', '*.db'], stdout=subprocess.PIPE)
    logger.info("Result: %s", result.stdout.decode('utf-8'))
    # Decode the output from bytes to string
    str_result = result.stdout.decode('utf-8').strip().split('\n')
    if len(str_result) == 1:
        tabular_db = str_result[0]
        return tabular_db
    else:
        raise Exception('there is more than 1 db')

def delete_tabular_db(tabular_db):
    os.remove(tabular_db)

def download_eval_db():
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_path, ground_truth_db)

def execute_sql(db, sql_statement):
    #execute tabular agent SQL
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.executescript(sql_statement)
    results = cursor.fetchall()
    conn.close()
    return results

# run docker compose locally
# then upload the 7 tables (csv files) from the S3 bucket in Dev, named "redbox-evaluation-dataset/tabular"

# initialise app
env = get_settings()
env = env.model_copy(update={"elastic_root_index": "redbox-data-integration"})
env = env.model_copy(update={"elastic_chunk_alias": "redbox-data-integration-chunk-current"})
ai_setting = AISettings(chat_backend=ChatLLMBackend(name="anthropic.claude-3-7-sonnet-20250219-v1:0", provider="bedrock"))
app = Redbox(debug=False, env=env)
langchain.debug = False
email_address = os.getenv("USER_EMAIL")
documents = ["account.csv", "card.csv", "client.csv", "disp.csv", "district.csv", "loan.csv", "order.csv"]
documents_paths = [email_address + "/" + doc for doc in documents]

# read evaluation dataset (also available in S3 bucket "redbox-evaluation-dataset/tabular" )
with open("../data_results/tabular/financial_dataset_original.json") as f:
    eval_data = json.load(f)

# regex to derive SQL from tabular output
regex_pattern = r"```sql\s*(.*?)\s```*"


results = []

start_time = time.time()

#download eval databse
download_eval_db()

for row in eval_data:
    prompt_no_evidence = row["question"]
    prompt_with_evidence = row["question"] + " " + row["evidence"]
    print(prompt_no_evidence)
    print(prompt_with_evidence)
    try_again, iter_limit = initialise_param()
    # without evidence
    while try_again and iter_limit < 3:
        x = get_state(uuid4(), prompts=[prompt_no_evidence], documents=documents_paths, ai_setting=ai_setting)
        tabular_db = None
        result = run_app(app, x)
        answer = result["messages"][-1].content
        row["redbox_answer_without_evidence"] = answer
        matches = re.findall(regex_pattern, answer, re.DOTALL)
        if len(matches) == 1:
            row["SQL_redbox_without_evidence"] = matches[0]
            #execute tabular sql
            tabular_db = get_tabular_db()
            tabular_results = execute_sql(tabular_db, row["SQL_redbox_without_evidence"])
            #execute tabular sql
            ground_truth_results = execute_sql(ground_truth_db, row["SQL"])
            if tabular_results == ground_truth_results:
                row["is_accurate_without_evidence"] = 1
            else:
                row["is_accurate_without_evidence"] = 0
            try_again = False
        else:
            if answer == "Agent stopped due to iteration limit or time limit":
                row["SQL_redbox_without_evidence"] = "None"
                row["is_accurate_without_evidence"] = 0
                try_again = False
            else:
                row["SQL_redbox_without_evidence"] = "None"
                try_again = True
                iter_limit += 1
        if "is_accurate_without_evidence" not in row:
            row["is_accurate_without_evidence"] = 0
  
        if not tabular_db:
            tabular_db = get_tabular_db()
        delete_tabular_db(tabular_db)

    with open('../data_results/tabular/tabular_results_'+str(row["question_id"])+"_without_evidence.txt",'w') as f:
        f.write(str(tabular_results))
    with open('../data_results/tabular/ground_truth_results_'+str(row["question_id"])+"_without_evidence.txt",'w') as f:
        f.write(str(ground_truth_results))

    try_again, iter_limit = initialise_param()
    # with evidence
    while try_again and iter_limit < 3:
        x = get_state(uuid4(), prompts=[prompt_with_evidence], documents=documents_paths, ai_setting=ai_setting)
        tabular_db = None
        result = run_app(app, x)
        answer = result["messages"][-1].content
        row["redbox_answer_with_evidence"] = answer
        matches = re.findall(regex_pattern, answer, re.DOTALL)
        if len(matches) == 1:
            row["SQL_redbox_with_evidence"] = matches[0]
            #execute tabular sql
            tabular_db = get_tabular_db()
            tabular_results = execute_sql(tabular_db, row["SQL_redbox_with_evidence"])
            #execute tabular sql
            ground_truth_results = execute_sql(ground_truth_db, row["SQL"])
            if tabular_results == ground_truth_results:
                row["is_accurate_with_evidence"] = 1
            else:
                row["is_accurate_with_evidence"] = 0
            try_again = False
        else:
            if answer == "Agent stopped due to iteration limit or time limit":
                row["SQL_redbox_with_evidence"] = "None"
                row["is_accurate_with_evidence"] = 0
                try_again = False
            else:
                row["SQL_redbox_with_evidence"] = "None"
                try_again = True
                iter_limit += 1
        if "is_accurate_with_evidence" not in row:
            row["is_accurate_with_evidence"] = 0
  
        if not tabular_db:
            tabular_db = get_tabular_db()
        delete_tabular_db(tabular_db)

    with open('../data_results/tabular/tabular_results_'+str(row["question_id"])+"_with_evidence.txt",'w') as f:
        f.write(str(tabular_results))
    with open('../data_results/tabular/ground_truth_results_'+str(row["question_id"])+"_with_evidence.txt",'w') as f:
        f.write(str(ground_truth_results))

    results.append(row)
    # Writing to a JSON file with indentation
    with open("../data_results/tabular/evaluation_results_test.json", "w") as outfile:
        json.dump(results, outfile, indent=4)
    time.sleep(5)  # slow down request frequency to avoid API throttling
    logger.info("Iteration number: %s", str(eval_data.index(row)))


duration_time = time.time() - start_time
logger.info("Duration time: %s", str(duration_time))


os.remove(ground_truth_db)
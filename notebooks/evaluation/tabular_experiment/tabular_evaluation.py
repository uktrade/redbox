import json
import logging
import os
import re
import sqlite3
import subprocess
import time

# from langfuse.callback import CallbackHandler
from uuid import uuid4

import boto3
import langchain
from dotenv import load_dotenv

from redbox.app import Redbox
from redbox.models.chain import AISettings, ChatLLMBackend, RedboxQuery, RedboxState
from redbox.models.settings import get_settings

# just logging stuff like we have in redbox
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def get_state(user_uuid, prompts, documents, ai_setting):
    """Get state for Redbox.
    Args:
    - user_uuid: user id
    - promots: user input / question
    - documents: list of documents selected in redbox
    - ai setting: AI settings including model name
    Returns Redbox state
    """
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
    """Invokes Redbox graph
     Args:
     - app: Redbox app
     - state: Redbox state
    Returns: Results from redbox graph
    """
    return app.graph.invoke(state)


def initialise_param():
    """Inialise params
     Args:
     - try_again: True for Tabular agent retry, or False
     - iter_limit: the current number of iterations
    Returns: inialised params
    """
    try_again = True
    iter_limit = 0
    return try_again, iter_limit


def get_tabular_db():
    """Fetch name of tabular database created by tabular agent

    Returns: full path of tabular database
    """
    result = subprocess.run(["find", ".", "-name", "*.db"], check=False, stdout=subprocess.PIPE)
    logger.info("Result: %s", result.stdout.decode("utf-8"))
    # Decode the output from bytes to string
    str_result = result.stdout.decode("utf-8").strip().split("\n")
    if len(str_result) == 1:
        return str_result[0]
    else:
        msg = "there is more than 1 db"
        raise Exception(msg)


def delete_tabular_db(tabular_db):
    """Deletes tabular database from local directory
    Args:
    - tabular_db: full path of tabular database
    """
    os.remove(tabular_db)


def download_eval_db(bucket_name, file_path, ground_truth_db):
    """Downloads evaluation database
    Args:
    - bucket_name: name of s3 bucket
    - file_path:  filename in s3 bucket
    - ground_truth_db: local filename
    """
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, file_path, ground_truth_db)


def execute_sql(db, sql_statement):
    """Executes a SQL statement using a database
    Args:
    - db: database name
    - sql_statement:  SQL query
    Returns: results from the SQL query
    """
    # execute tabular agent SQL
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(sql_statement)
    results = cursor.fetchall()
    conn.close()
    return results


def get_tabular_results(row, ground_truth_results, documents, ai_setting, env, question, suffix):
    """Derive results from tabular agent on a specific question
    Args:
    - row: row from the evaluation dataset
    - ground_truth_results:  results from the ground truth SQL query
    - documents: full paths of tables selected for tabular agent
    - ai setting: AI settings including model name
    - env: local setting
    - question: question sent to tabular agent
    - suffix: suffix name used to stored results in JSON. Can be "without_evidence" for questions without evidence, and "with_evidence" for questions containing evidence.

    Returns: results the row from the evaluation dataset with the saved results from tabular agent.
    """

    try_again, iter_limit = initialise_param()
    while try_again and iter_limit < 3:
        x = get_state(uuid4(), prompts=[question], documents=documents, ai_setting=ai_setting)
        app = Redbox(debug=False, env=env)
        result = run_app(app, x)
        answer = result["messages"][-1].content
        row["redbox_answer_" + suffix] = answer
        # regex to derive SQL from tabular output
        regex_pattern = r"```sql\s*(.*?)\s```*"
        matches = re.findall(regex_pattern, answer, re.DOTALL)
        tabular_db = get_tabular_db()
        if len(matches) == 1:
            row["SQL_redbox_" + suffix] = matches[0]
            try:
                # execute tabular sql
                tabular_results = execute_sql(tabular_db, row["SQL_redbox_" + suffix])
                if tabular_results == ground_truth_results:
                    row["is_accurate_" + suffix] = 1
                else:
                    row["is_accurate_" + suffix] = 0
            except Exception as e:
                row["is_accurate_" + suffix] = 0
                row["redbox_sql_error_" + suffix] = str(e)
                tabular_results = "None"
            try_again = False

        else:  # either the answer is Agent stopped due to iteration limit or time limit or there is no SQL output
            row["SQL_redbox_" + suffix] = "None"
            row["is_accurate_" + suffix] = 0
            tabular_results = "None"
            try_again = True
            iter_limit += 1

        delete_tabular_db(tabular_db)

    with open("../data_results/tabular/tabular_results_" + str(row["question_id"]) + "_" + suffix + ".txt", "w") as f:
        f.write(str(tabular_results))
    return row


def main():
    # initialise app
    env = get_settings()
    env = env.model_copy(update={"elastic_root_index": "redbox-data-integration"})
    env = env.model_copy(update={"elastic_chunk_alias": "redbox-data-integration-chunk-current"})
    ai_setting = AISettings(
        chat_backend=ChatLLMBackend(name="anthropic.claude-3-7-sonnet-20250219-v1:0", provider="bedrock")
    )
    langchain.debug = False
    email_address = os.getenv("USER_EMAIL")
    documents = ["account.csv", "card.csv", "client.csv", "disp.csv", "district.csv", "loan.csv", "order.csv"]
    documents_paths = [email_address + "/" + doc for doc in documents]

    # read evaluation dataset (also available in S3 bucket "redbox-evaluation-dataset/tabular" )
    with open("../data_results/tabular/financial_dataset_original.json") as f:
        eval_data = json.load(f)

    results = []

    start_time = time.time()

    # download eval databse
    bucket_name = "redbox-evaluation-dataset"
    file_path = "tabular/financial.sqlite"
    ground_truth_db = "./financial.sqlite"
    download_eval_db(bucket_name, file_path, ground_truth_db)

    for row in eval_data:
        prompt_no_evidence = row["question"]
        prompt_with_evidence = row["question"] + " " + row["evidence"]
        # execute ground truth sql and save results
        ground_truth_results = execute_sql(ground_truth_db, row["SQL"])
        with open("../data_results/tabular/ground_truth_results_" + str(row["question_id"]) + ".txt", "w") as f:
            f.write(str(ground_truth_results))
        # execute tabular results and save them (without evidence)
        row = get_tabular_results(
            row,
            ground_truth_results,
            documents_paths,
            ai_setting,
            env,
            question=prompt_no_evidence,
            suffix="without_evidence",
        )
        # execute tabular results and save them (with evidence)
        row = get_tabular_results(
            row,
            ground_truth_results,
            documents_paths,
            ai_setting,
            env,
            question=prompt_with_evidence,
            suffix="with_evidence",
        )
        results.append(row)
        # Writing to a JSON file with indentation
        with open("../data_results/tabular/evaluation_results.json", "w") as outfile:
            json.dump(results, outfile, indent=4)
        time.sleep(5)  # slow down request frequency to avoid API throttling
        logger.info("Iteration number: %s", str(eval_data.index(row)))
    # clean up
    os.remove(ground_truth_db)

    duration_time = time.time() - start_time
    logger.info("Duration time: %s", str(duration_time))


if __name__ == "__main__":
    main()

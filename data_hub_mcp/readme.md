# Overview

Data Hub MCP and ETL

Provides API like access to LLM agents via the MCP protocol.
Also provides local seed data for testing the MCP

## Quick start

### Install dependencies

locally you can simply install additional requirements into your existing venv by runing the following

pip install -r requirements.txt

### database setup

Setup the mcp database: 

docker compose up db_data_hub

run basic seed data and table setup (from within {project_root}/data_hub_mcp) 

python run_ingest.py 

### run server locally

this will run the mcp server in docker 

docker compose up data_hub_mcp

note: you can also run locally with 

python run_mcp_server.py

test in browser

go to http://localhost:8100/mcp

you should see an MCP response

### Call client locally with examples

python run_mcp_client.py

# Overview

Data Hub MCP and ETL

Provides API like access to LLM agents via the MCP protocol.
Also provides local seed data for testing the MCP

## Quick start

### Install dependencies

locally you can simply install additional requirements into your existing venv by runing the following

pip install -r requirements.txt

### Database setup

Create a database and users as per details specified in data_hub_mcp/.env.local


### run server locally

python run_mcp_server.py

run docker server

docker build --tag 'data_hub_mcp_server' .  

then

docker run -d -p 9000:8000 --name local_mcp data_hub_mcp_server

or

you can just run :

docker compose up data_hub_mcp

test in browser

go to http://localhost:8000/mcp

you should see an MCP response

### Call client locally with examples

python run_mcp_client.py

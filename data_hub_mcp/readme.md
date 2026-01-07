# Overview

Data Hub MCP

Provides API like access to LLM agents via the MCP protocol.

more details : todo

## Quick start

### Install dependencies

locally you can simply install additional requirements into your existing venv by runing the following

pip install -r requirements.txt

### Database setup

todo ...

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

import os

import mcp_client
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

import mcp_client
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

if __name__ == "__main__":
    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")

    client = Client(f"http://{mcp_host}:{mcp_port}/mcp")
    mcp_client.run_examples(client)

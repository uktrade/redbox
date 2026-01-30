import os

import mcp_client_prop
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

if __name__ == "__main__":
    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")

    client = Client(f"http://{mcp_host}:{mcp_port}/mcp")
    mcp_client_prop.run_examples(client)

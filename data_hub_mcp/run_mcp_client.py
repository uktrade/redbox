import asyncio
import os
from pathlib import Path

import mcp_client_prop
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

if __name__ == "__main__":
    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")

    client = Client(f"http://{mcp_host}:{mcp_port}/mcp")

    json_output = asyncio.run(mcp_client_prop.run_examples(client))
    # print(json_output)

    with Path.open("output.json", "w") as f:
        f.write(json_output)

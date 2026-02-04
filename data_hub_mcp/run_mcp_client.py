import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client

from data_hub_mcp import mcp_client, mcp_client_parameterised

load_dotenv()

SHOULD_EXPORT = os.getenv("MCP_CLIENT_SHOULD_EXPORT", "false").lower() in ("true", "1", "yes")

if __name__ == "__main__":
    mcp_host = os.getenv("MCP_HOST")
    mcp_port = os.getenv("MCP_PORT")

    client = Client(f"http://{mcp_host}:{mcp_port}/mcp")

    if SHOULD_EXPORT:
        json_output = asyncio.run(mcp_client_parameterised.run_examples(client))
        with Path.open("output.json", "w") as f:
            f.write(json_output)

    else:
        mcp_client.run_examples(client)

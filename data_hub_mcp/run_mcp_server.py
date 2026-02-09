import os

import mcp_server
import mcp_server_test
from dotenv import load_dotenv

load_dotenv()

USE_STATIC_DATA = os.getenv("MCP_USE_STATIC_DATA", "false").lower() in ("true", "1", "yes")

if __name__ == "__main__":
    if USE_STATIC_DATA:
        mcp_server_test.mcp.run(
            transport="http",
            port=8100,
            host="0.0.0.0",  # noqa: S104
        )
    else:
        mcp_server.mcp.run(
            transport="http",
            port=8100,
            host="0.0.0.0",  # noqa: S104
        )

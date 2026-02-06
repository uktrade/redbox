import logging

import mcp_server
from dotenv import load_dotenv
from util.logging import setup_colored_logging

load_dotenv()
setup_colored_logging(level=logging.INFO)

if __name__ == "__main__":
    mcp_server.mcp.run(
        transport="http",
        port=8100,
        host="0.0.0.0",  # noqa: S104
    )

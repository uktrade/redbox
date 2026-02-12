import mcp_server
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    mcp_server.mcp.run(
        transport="http",
        port=8100,
        host="0.0.0.0",  # noqa: S104
    )

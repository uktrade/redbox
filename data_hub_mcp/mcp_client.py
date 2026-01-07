import asyncio
import logging
import sys
from dataclasses import asdict

from fastmcp import Client

client = Client("http://localhost:8100/mcp")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


async def call_tool(tool_name: str, name: str):
    async with client:
        result = await client.call_tool(tool_name, {"name": name})
        logger.info(result)
        return asdict(result)


async def call_companies(company_name: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "companies", {"company_name": company_name, "page": page, "page_size": page_size}
        )
        logger.info(result)

        return asdict(result)


def run_examples():
    asyncio.run(call_tool("greet", "Ford"))
    asyncio.run(call_tool("greet", "Ford"))
    asyncio.run(call_tool("greet", "Ford"))

    # # asyncio.run(call_tool('greet', "Doug"))
    # asyncio.run(call_companies("FIREWORK"))
    # asyncio.run(call_companies("1ST", 1, 2))
    # asyncio.run(call_companies("1ST", 2, 2))
    # asyncio.run(call_companies("bmw", 0, 6))
    # asyncio.run(call_companies("bmw", 1, 6))

import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from mcp.types import TextContent

from data_hub_mcp.mcp_server import mcp as data_hub_mcp_server


@pytest.mark.data_hub_mcp
@pytest.fixture
def mcp_server():
    mcp = FastMCP(name="CalculationServer")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        return a + b

    return mcp


# A straightforward test of our tool
@pytest.mark.data_hub_mcp
@pytest.mark.asyncio
async def test_add_tool(mcp_server: FastMCP):
    async with Client(mcp_server) as client:  # Client uses the mcp_server instance
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "3"


@pytest.mark.asyncio
async def test_server_tools():
    tools = await data_hub_mcp_server.get_tools()
    assert len(tools.items()) == 8

    assert "companies" in tools
    assert "company_details" in tools
    assert "company_details_extended" in tools
    assert "companies_or_interactions" in tools
    assert "company_interactions" in tools
    assert "account_management_objectives" in tools
    assert "investment_projects" in tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "expected_properties"),
    [
        ("companies", ["companies", "page", "page_size", "total"]),
        ("company_details", ["result"]),
        ("companies_or_interactions", ["result"]),
        ("company_interactions", ["result"]),
        ("account_management_objectives", ["result"]),
        ("investment_projects", ["result"]),
    ],
)
async def test_server_tool(tool_name, expected_properties):
    tool = await data_hub_mcp_server.get_tool(tool_name)
    for expected_property in expected_properties:
        assert expected_property in tool.output_schema["properties"]

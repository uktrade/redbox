import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from mcp.types import TextContent

from mcp_server import mcp as data_hub_mcp_server


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
    assert len(tools.items()) == 7

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
        (
            "company_details",
            [
                "address_1",
                "address_2",
                "address_county",
                "address_postcode",
                "address_country",
                "address_area_name",
                "address_town",
                "company_number",
                "description",
                "name",
                "turnover_gbp",
                "business_type",
                "duns_number",
                "export_experience",
                "global_ultimate_duns_number",
                "headquarter_type",
                "id",
                "number_of_employees",
                "registered_address_1",
                "registered_address_2",
                "registered_address_country",
                "registered_address_county",
                "registered_address_postcode",
                "registered_address_area_name",
                "registered_address_town",
                "sector",
                "export_segment",
                "export_sub_segment",
                "trading_names",
                "turnover",
                "turnover_usd",
                "uk_region",
                "website",
                "is_out_of_business",
                "strategy",
            ],
        ),
        (
            "company_details_extended",
            ["company_details", "investment_projects", "account_management_objectives", "interactions"],
        ),
        ("companies_or_interactions", ["companies_search_result", "interactions_search_result"]),
        ("company_interactions", ["interactions", "total", "page", "page_size"]),
        ("account_management_objectives", ["account_management_objectives", "total", "page", "page_size"]),
        ("investment_projects", ["investment_projects", "total", "page", "page_size"]),
    ],
)
async def test_server_tool(tool_name, expected_properties):
    tool = await data_hub_mcp_server.get_tool(tool_name)
    schema = tool.output_schema

    properties = schema["properties"]

    if "result" in properties and "anyOf" in properties["result"]:
        ref = properties["result"]["anyOf"][0]["$ref"]
        ref_name = ref.split("/")[-1]
        properties = schema["$defs"][ref_name]["properties"]

    for expected_property in expected_properties:
        assert expected_property in properties

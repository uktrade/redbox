import os

from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyDetailsExtended,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)
from db_ops import (
    db_check,
)
from fastmcp import FastMCP
from repository import PostgresRepository, StaticRepository

# from fastmcp.server.auth.providers.auth0 import Auth0Provider
from starlette.responses import JSONResponse

# load_dotenv()

# auth_provider = Auth0Provider(
#     config_url=os.getenv("AUTHBROKER_CONFIG_URL"),  # Your Auth0 configuration URL
#     client_id=os.getenv("AUTHBROKER_CLIENT_ID"),  # Your Auth0 application Client ID
#     client_secret=os.getenv("AUTHBROKER_CLIENT_SECRET"),  # Your Auth0 application Client Secret
#     audience=os.getenv("AUTHBROKER_AUDIENCE"),  # Your Auth0 API audience
#     base_url=os.getenv("AUTHBROKER_BASE_URL"),  # Must match your application configuration
# )

# Static Data Override

USE_STATIC_DATA = os.getenv("MCP_USE_STATIC_DATA", "false").lower() in ("true", "1", "yes")
repo = PostgresRepository()
server_suffix = ""

if USE_STATIC_DATA:
    repo = StaticRepository()
    server_suffix = " [STATIC]"

# Setup Server

mcp = FastMCP(
    name=f"Data Hub companies MCP server{server_suffix}",
    stateless_http=True,
    json_response=True,
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check():
    db_status = "failed"
    if db_check():
        db_status = "success"

    return JSONResponse({"status": "healthy", "service": "mcp-server", "db_access_status": db_status})


@mcp.tool(
    name="greet",  # Custom tool name for the LLM
    description="Basic example tool useful for testing.",  # Custom description
    tags={"testing"},  # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "Doug Mills"},  # Custom metadata
)
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(
    name="companies",
    description="Query companies based on company name, returns a short overview of a list of companies",
    tags={"data_hub", "companies", "search"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    return await repo.companies(company_name, page_size, page)


@mcp.tool(
    name="company_details",
    description="Full details of a company",
    tags={"data_hub", "companies"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details(company_id: str) -> CompanyDetails | None:
    return await repo.company_details(company_id)


@mcp.tool(
    name="company_details_extended",
    description="Full details of a company, related interactions, account management "
    "objectives and investment projects",
    tags={"data_hub", "companies", "interactions", "objectives", "investment projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details_extended(company_id: str) -> CompanyDetailsExtended | None:
    return await repo.company_details(company_id)


@mcp.tool(
    name="companies_or_interactions",
    description="Query companies, and will return interactions on a single result, "
    "or a list of companies of there are multiple matches",
    tags={"data_hub", "companies_or_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies_or_interactions(
    company_name: str, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult | None:
    return await repo.companies_or_interactions(company_name, page_size, page)


@mcp.tool(
    name="company_interactions",
    description="Query company interactions based on company id",
    tags={"data_hub", "company_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_interactions(
    company_id: str, page_size: int = 10, page: int = 0
) -> CompanyInteractionSearchResult | None:
    return await repo.company_interactions(company_id, page_size, page)


@mcp.tool(
    name="account_management_objectives",
    description="Query account management objectives based on company id",
    tags={"data_hub", "company", "account_management_objectives"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def account_management_objectives(
    company_id: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult | None:
    return await repo.account_management_objectives(company_id, page_size, page)


@mcp.tool(
    name="investment_projects",
    description="Query investment projects based on company id",
    tags={"data_hub", "company", "investment_projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def investment_projects(
    company_id: str, page_size: int = 10, page: int = 0
) -> InvestmentProjectsSearchResult | None:
    return await repo.investment_projects(company_id, page_size, page)

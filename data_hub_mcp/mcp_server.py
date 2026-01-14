import os

from data_classes import CompanyDetails, CompanySearchResult, CompanyInteractionSearchResult, AccountManagementObjectivesSearchResult, InvestmentProjectsSearchResult
from db_ops import get_companies, get_company, db_check, get_company_interactions, get_account_management_objectives, get_investment_projects
from fastmcp import FastMCP
from starlette.responses import JSONResponse

mcp = FastMCP(
    name="Data Hub companies MCP server",
    stateless_http=True,
    json_response=True,
)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):

    db_status = "failed"
    if db_check():
        db_status = "success"

    return JSONResponse({"status": "healthy", "service": "mcp-server", "db_access_status": db_status})

@mcp.custom_route("/config", methods=["GET"])
async def config(request):
    return JSONResponse({
        'env': str(os.environ)
    })


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
    return get_companies(company_name, page_size, page)

@mcp.tool(
    name="company_details",
    description="Full details of a company",
    tags={"data_hub", "companies"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details(company_id: str) -> CompanyDetails | None:
    return get_company(company_id)

@mcp.tool(
    name="company_interactions",
    description="Query company interactions based on company id",
    tags={"data_hub", "company_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_interactions(company_id: str, page_size: int = 10, page: int = 0) -> CompanyInteractionSearchResult | None:
    return get_company_interactions(company_id, page_size, page)

@mcp.tool(
    name="account_management_objectives",
    description="Query account management objectives based on company id",
    tags={"data_hub", "company", 'account_management_objectives'},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def account_management_objectives(company_id: str, page_size: int = 10, page: int = 0) -> AccountManagementObjectivesSearchResult | None:
    return get_account_management_objectives(company_id, page_size, page)

@mcp.tool(
    name="investment_projects",
    description="Query investment projects based on company id",
    tags={"data_hub", "company", 'investment_projects'},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def investment_projects(company_id: str, page_size: int = 10, page: int = 0) -> InvestmentProjectsSearchResult | None:
    return get_investment_projects(company_id, page_size, page)

from data_hub_mcp.data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyDetailsExtended,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)
from data_hub_mcp.db_ops import (
    db_check,
    get_account_management_objectives,
    get_companies,
    get_companies_or_interactions,
    get_company,
    get_company_extended,
    get_company_interactions,
    get_investment_projects,
)
from fastmcp import FastMCP
from starlette.responses import JSONResponse

mcp = FastMCP(
    name="Data Hub companies MCP server",
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
    name="company_details_extended",
    description="Full details of a company, related interactions, account management "
    "objectives and investment projects",
    tags={"data_hub", "companies", "interactions", "objectives", "investment projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details_extended(
    company_id: str,
    include_interactions: bool = True,
    include_objectives: bool = True,
    include_investment_projects: bool = True,
) -> CompanyDetailsExtended:
    return get_company_extended(company_id, include_interactions, include_objectives, include_investment_projects)


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
    return get_companies_or_interactions(company_name, page_size, page)


@mcp.tool(
    name="company_interactions",
    description="Query company interactions based on company id",
    tags={"data_hub", "company_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_interactions(
    company_id: str, page_size: int = 10, page: int = 0
) -> CompanyInteractionSearchResult | None:
    return get_company_interactions(company_id, page_size, page)


@mcp.tool(
    name="account_management_objectives",
    description="Query account management objectives based on company id",
    tags={"data_hub", "company", "account_management_objectives"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def account_management_objectives(
    company_id: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult | None:
    return get_account_management_objectives(company_id, page_size, page)


@mcp.tool(
    name="investment_projects",
    description="Query investment projects based on company id",
    tags={"data_hub", "company", "investment_projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def investment_projects(
    company_id: str, page_size: int = 10, page: int = 0
) -> InvestmentProjectsSearchResult | None:
    return get_investment_projects(company_id, page_size, page)

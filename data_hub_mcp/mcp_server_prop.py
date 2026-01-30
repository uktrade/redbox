import os

from data_classes_prop import (
    CompanyEnrichmentSearchResult,
    # CompaniesOrInteractionSearchResult,
    SectorGroupedCompanySearchResult,
)
from db_ops_prop import (
    db_check,
    # get_account_management_objectives,
    # get_companies_or_interactions,
    # get_companies,
    get_company_details,
    get_related_company_details,
    # get_company_interactions,
    # get_investment_projects,
)
from fastmcp import FastMCP

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


@mcp.custom_route("/config", methods=["GET"])
async def config():
    return JSONResponse({"env": str(os.environ)})


@mcp.tool(
    name="greet",  # Custom tool name for the LLM
    description="Basic example tool useful for testing.",  # Custom description
    tags={"testing"},  # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "Doug Mills"},  # Custom metadata
)
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(
    name="company_details",
    description="Query companies based on company name",
    tags={"data_hub", "companies", "search"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details(
    company_name: str,
    page_size: int = 10,
    page: int = 0,
    fetch_interactions: bool = True,
    fetch_objectives: bool = True,
    fetch_investments: bool = True,
) -> CompanyEnrichmentSearchResult:
    """
    Async wrapper for fetching enriched company details.
    Allows optional fetching of related interactions, objectives, and investment projects.
    """
    return get_company_details(
        company_name=company_name,
        page_size=page_size,
        page=page,
        fetch_interactions=fetch_interactions,
        fetch_objectives=fetch_objectives,
        fetch_investments=fetch_investments,
    )


@mcp.tool(
    name="related_company_details",
    description="Fetch companies related by sector or a given company's sector, with optional enrichment",
    tags={"data_hub", "companies", "related", "search"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def related_company_details(
    sector_name: str | None = None,
    company_name: str | None = None,
    page: int = 0,
    page_size: int = 10,
    fetch_interactions: bool = True,
    fetch_objectives: bool = True,
    fetch_investments: bool = True,
) -> SectorGroupedCompanySearchResult:
    """
    Async wrapper for fetching companies in the same sector as a given company or a target sector.
    Optional flags control whether to fetch interactions, objectives, and investment projects.
    """
    return get_related_company_details(
        sector_name=sector_name,
        company_name=company_name,
        page=page,
        page_size=page_size,
        fetch_interactions=fetch_interactions,
        fetch_objectives=fetch_objectives,
        fetch_investments=fetch_investments,
    )


# @mcp.tool(
#     name="company_details",
#     description="Full details of a company",
#     tags={"data_hub", "companies"},
#     meta={"version": "1.0", "author": "Doug Mills"},
# )
# async def company_details(company_id: str) -> CompanyDetails | None:
#     return get_company(company_id)


# @mcp.tool(
#     name="companies_or_interactions",
#     description="""
# Query companies, and will return interactions on a single result,
# or a list of companies of there are multiple matches
# """,
#     tags={"data_hub", "companies_or_interactions"},
#     meta={"version": "1.0", "author": "Doug Mills"},
# )
# async def companies_or_interactions(
#     company_name: str, page_size: int = 10, page: int = 0
# ) -> CompaniesOrInteractionSearchResult | None:
#     return get_companies_or_interactions(company_name, page_size, page)


# @mcp.tool(
#     name="company_interactions",
#     description="Query company interactions based on company id",
#     tags={"data_hub", "company_interactions"},
#     meta={"version": "1.0", "author": "Doug Mills"},
# )
# async def company_interactions(
#     company_id: str, page_size: int = 10, page: int = 0
# ) -> CompanyInteractionSearchResult | None:
#     return get_company_interactions(company_id, page_size, page)


# @mcp.tool(
#     name="account_management_objectives",
#     description="Query account management objectives based on company id",
#     tags={"data_hub", "company", "account_management_objectives"},
#     meta={"version": "1.0", "author": "Doug Mills"},
# )
# async def account_management_objectives(
#     company_id: str, page_size: int = 10, page: int = 0
# ) -> AccountManagementObjectivesSearchResult | None:
#     return get_account_management_objectives(company_id, page_size, page)


# @mcp.tool(
#     name="investment_projects",
#     description="Query investment projects based on company id",
#     tags={"data_hub", "company", "investment_projects"},
#     meta={"version": "1.0", "author": "Doug Mills"},
# )
# async def investment_projects(
#     company_id: str, page_size: int = 10, page: int = 0
# ) -> InvestmentProjectsSearchResult | None:
#     return get_investment_projects(company_id, page_size, page)

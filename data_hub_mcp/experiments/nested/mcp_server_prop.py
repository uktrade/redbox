import os

from fastmcp import FastMCP
from fastmcp.exceptions import ValidationError
from starlette.responses import JSONResponse

from data_hub_mcp.experiments.nested.data_classes_prop import (
    CompanyEnrichmentSearchResult,
    SectorGroupedCompanySearchResult,
    SectorGroupedInvestmentSummary,
    SectorGroupedOverview,
)
from data_hub_mcp.experiments.nested.db_ops_prop import (
    db_check,
    get_company_details,
    get_related_company_details,
    get_sector_investment_projects,
    get_sector_overview,
)

NoCompanyOrSectorError = ValidationError("No company or sector provided")

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
    fetch_interactions: bool = False,
    fetch_objectives: bool = False,
    fetch_investments: bool = False,
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
    # Validate input
    if not sector_name and not company_name:
        raise NoCompanyOrSectorError

    return get_related_company_details(
        sector_name=sector_name,
        company_name=company_name,
        page=page,
        page_size=page_size,
        fetch_interactions=fetch_interactions,
        fetch_objectives=fetch_objectives,
        fetch_investments=fetch_investments,
    )


@mcp.tool(
    name="sector_overview",
    description="Aggregate high-level metrics for a sector (companies, turnover, employees, investments, GVA)",
    tags={"data_hub", "sector", "overview"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def sector_overview(
    sector_name: str | None = None,
    company_name: str | None = None,
) -> SectorGroupedOverview:
    """
    Async wrapper for fetching aggregated sector metrics.
    Can query either by sector_name directly or a company_name to find its sector.
    """
    if not sector_name and not company_name:
        raise NoCompanyOrSectorError

    return get_sector_overview(
        sector_name=sector_name,
        company_name=company_name,
    )


@mcp.tool(
    name="sector_investment_projects",
    description="Fetch investment projects in a sector, including their economic impact and status",
    tags={"data_hub", "sector", "investment"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def sector_investment_projects(
    sector_name: str | None = None,
    company_name: str | None = None,
    page: int = 0,
    page_size: int = 10,
) -> SectorGroupedInvestmentSummary:
    """
    Async wrapper for fetching sector-level investment projects.
    Can query either by sector_name directly or company_name to find its sector.
    """
    if not sector_name and not company_name:
        raise NoCompanyOrSectorError

    return get_sector_investment_projects(
        sector_name=sector_name,
        company_name=company_name,
        page=page,
        page_size=page_size,
    )

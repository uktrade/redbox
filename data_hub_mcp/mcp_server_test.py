import os

from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)
from fastmcp import FastMCP
from starlette.responses import JSONResponse
from static_data import (
    STATIC_COMPANY_DETAILS,
    STATIC_COMPANY_DETAILS_2,
    STATIC_COMPANY_SHORT,
    STATIC_COMPANY_SHORT_2,
    STATIC_INTERACTION,
    STATIC_INTERACTION_2,
    STATIC_INVESTMENT_PROJECT,
    STATIC_INVESTMENT_PROJECT_2,
    STATIC_OBJECTIVE,
    STATIC_OBJECTIVE_2,
)

MATCH_TYPE = os.getenv("MCP_RECORD_MATCHING_TYPE", "exact").lower()


def check_company_name(input_company_name: str, db_company_name: str) -> bool:
    match MATCH_TYPE:
        case "contains":
            return input_company_name.lower() in db_company_name.lower()
        case "contains_keyword":
            return set(input_company_name.lower().split()) & set(db_company_name.lower().split())
        case _:
            return input_company_name.lower() == db_company_name.lower()


# -- MCP Setup --

mcp = FastMCP(
    name="Static Data Hub companies MCP server",
    stateless_http=True,
    json_response=True,
)

# -- Routes --


@mcp.custom_route("/health", methods=["GET"])
async def health_check():
    return JSONResponse(
        {
            "status": "healthy",
            "service": "mcp-static-server",
            "db_access_status": "disabled",
        }
    )


@mcp.custom_route("/config", methods=["GET"])
async def config():
    return JSONResponse({"env": dict(os.environ)})


# -- Tools --


@mcp.tool(name="greet", description="Basic example tool useful for testing.")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(
    name="companies",
    description="""
Search for companies by name and retrieve a short overview of matching results.

This tool returns a list of companies containing:
- `id`: Internal company identifier
- `name`: Company name
- `description`: Brief description of the company
- `turnover_gbp`: Approximate turnover in GBP
- `address_1`, `address_2`, `address_postcode`, `address_country`: Address details
- `company_number`: Official registration number
""",
)
async def companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    if check_company_name(company_name, "acme"):
        return CompanySearchResult(
            companies=[STATIC_COMPANY_SHORT, STATIC_COMPANY_SHORT_2],
            total=2,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT.name):
        return CompanySearchResult(
            companies=[STATIC_COMPANY_SHORT],
            total=1,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT_2.name):
        return CompanySearchResult(
            companies=[STATIC_COMPANY_SHORT_2],
            total=1,
            page=page,
            page_size=page_size,
        )
    return CompanySearchResult(
        companies=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    name="company_details",
    description="""
Retrieve full details of a single company by its name.

Returned data includes:
- Full address and registered address
- Company identifiers: company_number, DUNS, VAT number
- Financials: turnover in GBP and USD, estimated flags
- Business information: sector, business type, trading names, export experience
- Operational status and strategy
- Employee information

Use this tool when you need complete, detailed information about a single company.
""",
)
async def company_details(company_name: str) -> CompanyDetails | None:
    if check_company_name(company_name, STATIC_COMPANY_DETAILS.name):
        return STATIC_COMPANY_DETAILS
    if check_company_name(company_name, STATIC_COMPANY_DETAILS_2.name):
        return STATIC_COMPANY_DETAILS_2
    return None


@mcp.tool(
    name="companies_or_interactions",
    description="""
Search for a company by name. If a single match is found, returns associated interactions.
If multiple matches are found, returns a list of company overviews.

Outputs include:
- `companies_search_result`: Short company info (as in `companies` tool)
- `interactions_search_result`: List of company interactions (date, subject, team, theme)

This tool is useful when you want either a short list of companies or the interactions of a single company.
""",
)
async def companies_or_interactions(
    company_name: str, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult:
    _ = company_name
    if check_company_name(company_name, "acme"):
        return CompaniesOrInteractionSearchResult(
            companies_search_result=CompanySearchResult(
                companies=[STATIC_COMPANY_SHORT, STATIC_COMPANY_SHORT_2],
                total=2,
                page=page,
                page_size=page_size,
            ),
            interactions_search_result=[STATIC_INTERACTION, STATIC_INTERACTION_2],
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT.name):
        return CompaniesOrInteractionSearchResult(
            companies_search_result=CompanySearchResult(
                companies=[STATIC_COMPANY_SHORT],
                total=1,
                page=page,
                page_size=page_size,
            ),
            interactions_search_result=[STATIC_INTERACTION],
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT_2.name):
        return CompaniesOrInteractionSearchResult(
            companies_search_result=CompanySearchResult(
                companies=[STATIC_COMPANY_SHORT_2],
                total=1,
                page=page,
                page_size=page_size,
            ),
            interactions_search_result=[STATIC_INTERACTION_2],
        )
    return CompaniesOrInteractionSearchResult()


@mcp.tool(
    name="company_interactions",
    description="""
Retrieve interactions for a given company name.

Each interaction includes:
- Team region and country
- Date of interaction and financial year
- Interaction subject and theme
- Company sector

Use this tool to see all historical interactions for a company.
""",
)
async def company_interactions(company_name: str, page_size: int = 10, page: int = 0) -> CompanyInteractionSearchResult:
    if check_company_name(company_name, "acme"):
        return CompanyInteractionSearchResult(
            interactions=[STATIC_INTERACTION, STATIC_INTERACTION_2],
            total=2,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_INTERACTION.company_name):
        return CompanyInteractionSearchResult(
            interactions=[STATIC_INTERACTION],
            total=1,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_INTERACTION_2.company_name):
        return CompanyInteractionSearchResult(
            interactions=[STATIC_INTERACTION_2],
            total=1,
            page=page,
            page_size=page_size,
        )
    return CompanyInteractionSearchResult(
        interactions=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    name="account_management_objectives",
    description="""
Retrieve account management objectives for a given company.

Each objective contains:
- `subject` and `detail`
- Target date and progress
- Blocker flags and descriptions
- Created/modified timestamps and user IDs

This tool is useful for tracking strategic objectives for a company's account.
""",
)
async def account_management_objectives(
    company_name: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult:
    if check_company_name(company_name, "acme"):
        return AccountManagementObjectivesSearchResult(
            account_management_objectives=[STATIC_OBJECTIVE, STATIC_OBJECTIVE_2],
            total=2,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT.name):
        return AccountManagementObjectivesSearchResult(
            account_management_objectives=[STATIC_OBJECTIVE],
            total=1,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT_2.name):
        return AccountManagementObjectivesSearchResult(
            account_management_objectives=[STATIC_OBJECTIVE_2],
            total=1,
            page=page,
            page_size=page_size,
        )
    return AccountManagementObjectivesSearchResult(
        account_management_objectives=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@mcp.tool(
    name="investment_projects",
    description="""
Retrieve investment projects associated with a company.

Each project includes:
- Name, description, and sector
- Investment type and value
- Project dates (estimated and actual land date)
- Locations, UK regions, and client details
- Number of new and safeguarded jobs
- Strategic drivers and stage

Use this tool to explore planned and ongoing investment projects for a company.
""",
)
async def investment_projects(company_name: str, page_size: int = 10, page: int = 0) -> InvestmentProjectsSearchResult:
    if check_company_name(company_name, "acme"):
        return InvestmentProjectsSearchResult(
            investment_projects=[STATIC_INVESTMENT_PROJECT, STATIC_INVESTMENT_PROJECT_2],
            total=2,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT.name):
        return InvestmentProjectsSearchResult(
            investment_projects=[STATIC_INVESTMENT_PROJECT],
            total=1,
            page=page,
            page_size=page_size,
        )
    if check_company_name(company_name, STATIC_COMPANY_SHORT_2.name):
        return InvestmentProjectsSearchResult(
            investment_projects=[STATIC_INVESTMENT_PROJECT_2],
            total=1,
            page=page,
            page_size=page_size,
        )
    return InvestmentProjectsSearchResult(
        investment_projects=[],
        total=0,
        page=page,
        page_size=page_size,
    )

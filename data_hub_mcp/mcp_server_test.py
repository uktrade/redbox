import os
from datetime import UTC, date, datetime

from data_classes import (
    AccountManagementObjective,
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteraction,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    CompanyShort,
    InvestmentProject,
    InvestmentProjectsSearchResult,
)
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# ------------------------------------------------------------------
# MCP SETUP
# ------------------------------------------------------------------

mcp = FastMCP(
    name="Static Data Hub companies MCP server",
    stateless_http=True,
    json_response=True,
)

# ------------------------------------------------------------------
# STATIC FIXTURE DATA
# ------------------------------------------------------------------

STATIC_COMPANY_ID = "cmp_001"

STATIC_COMPANY_SHORT = CompanyShort(
    id=STATIC_COMPANY_ID,
    address_1="1 Test Street",
    address_2=None,
    address_postcode="SW1A 1AA",
    address_country="UK",
    company_number="12345678",
    description="Static test company",
    name="Acme Industries Ltd",
    turnover_gbp=1_500_000,
)

STATIC_COMPANY_DETAILS = CompanyDetails(
    id=STATIC_COMPANY_ID,
    name="Acme Industries Ltd",
    description="Static test company for MCP testing",
    address_1="1 Test Street",
    address_2=None,
    address_county="Greater London",
    address_postcode="SW1A 1AA",
    address_country="UK",
    address_area_name="Westminster",
    address_town="London",
    company_number="12345678",
    turnover_gbp=1_500_000,
    business_type="Manufacturing",
    duns_number="123456789",
    export_experience="Experienced",
    global_ultimate_duns_number="987654321",
    headquarter_type="UK HQ",
    is_number_of_employees_estimated=False,
    is_turnover_estimated=False,
    number_of_employees=120,
    registered_address_1="1 Test Street",
    registered_address_2=None,
    registered_address_country="UK",
    registered_address_county="Greater London",
    registered_address_postcode="SW1A 1AA",
    registered_address_area_name="Westminster",
    registered_address_town="London",
    sector="Advanced Manufacturing",
    export_segment="Industrial",
    export_sub_segment="Machinery",
    trading_names=["Acme"],
    turnover="£1.5m",
    turnover_usd=1_900_000,
    uk_region="London",
    vat_number="GB123456789",
    website="https://example.com",
    is_out_of_business=False,
    strategy="Expand EU exports",
)

STATIC_INTERACTION = CompanyInteraction(
    team_region="Europe",
    team_country="UK",
    interaction_year=2025,
    interaction_financial_year="2024/25",
    company_name="Acme Industries Ltd",
    company_sector="Advanced Manufacturing",
    date_of_interaction=date(2025, 1, 15),
    interaction_subject="Quarterly export review",
    interaction_theme_investment_or_export="Export",
)

STATIC_OBJECTIVE = AccountManagementObjective(
    id="amo_001",
    subject="Increase EU exports",
    detail="Support company with EU distributor introductions",
    target_date=date(2025, 6, 30),
    has_blocker=False,
    blocker_description=None,
    progress=40,
    created_on=datetime(2024, 11, 1, tzinfo=UTC),
    modified_on=datetime(2025, 1, 10, tzinfo=UTC),
    created_by_id="user_001",
    modified_by_id="user_002",
    company_id=STATIC_COMPANY_ID,
)

STATIC_INVESTMENT_PROJECT = InvestmentProject(
    id="ip_001",
    name="UK Expansion Project",
    description="New manufacturing facility",
    sector="Advanced Manufacturing",
    stage="Active",
    status="In progress",
    investment_type="FDI",
    investor_type="Foreign",
    investor_company_id=STATIC_COMPANY_ID,
    investor_company_sector="Manufacturing",
    uk_company_id=STATIC_COMPANY_ID,
    total_investment=25_000_000,
    number_new_jobs=80,
    number_safeguarded_jobs=20,
    likelihood_to_land="High",
    created_on=datetime(2024, 10, 1, tzinfo=UTC),
    modified_on=datetime(2025, 1, 5, tzinfo=UTC),
    actual_land_date=None,
    estimated_land_date=date(2025, 9, 30),
    address_1="Industrial Park",
    address_2=None,
    address_town="Birmingham",
    address_county="West Midlands",
    address_country="UK",
    address_postcode="B1 1AA",
    actual_uk_regions=None,
    anonymous_description=None,
    associated_non_fdi_r_and_d_project_id=None,
    average_salary="£45,000",
    business_activities=["Manufacturing", "R&D"],
    client_relationship_manager_id="crm_001",
    client_requirements=None,
    client_contact_ids=["contact_001"],
    client_contact_names=["Jane Doe"],
    client_contact_emails=["jane.doe@example.com"],
    competing_countries=["Germany", "Poland"],
    country_investment_originates_from="USA",
    created_by_id="user_001",
    delivery_partners=["Local Authority"],
    export_revenue=True,
    first_active_on=datetime(2024, 10, 5, tzinfo=UTC),
    fdi_type="Expansion",
    fdi_value="High",
    foreign_equity_investment=60,
    government_assistance=True,
    gross_value_added=5_000_000,
    gva_multiplier=1.4,
    level_of_involvement="High",
    modified_by_id="user_002",
    new_tech_to_uk=True,
    non_fdi_r_and_d_budget=False,
    other_business_activity=None,
    project_arrived_in_triage_on=datetime(2024, 9, 15, tzinfo=UTC),
    project_assurance_adviser_id="paa_001",
    project_moved_to_won=None,
    project_manager_id="pm_001",
    project_reference="IP-2025-001",
    proposal_deadline=date(2025, 3, 31, tzinfo=UTC),
    r_and_d_budget=True,
    referral_source_activity="Trade show",
    referral_source_activity_marketing=None,
    referral_source_activity_website=None,
    specific_programme=["Levelling Up"],
    strategic_drivers=["Market access", "Cost efficiency"],
    team_member_ids=["tm_001", "tm_002"],
    uk_company_sector="Manufacturing",
    possible_uk_regions=["West Midlands"],
    eyb_lead_ids=None,
)

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# TOOLS (STATIC)
# ------------------------------------------------------------------


@mcp.tool(name="greet", description="Basic example tool useful for testing.")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(name="companies", description="Static company search")
async def companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    _ = company_name
    return CompanySearchResult(
        companies=[STATIC_COMPANY_SHORT],
        total=1,
        page=page,
        page_size=page_size,
    )


@mcp.tool(name="company_details", description="Static company details")
async def company_details(company_id: str) -> CompanyDetails:
    _ = company_id
    return STATIC_COMPANY_DETAILS


@mcp.tool(name="companies_or_interactions", description="Static combined search")
async def companies_or_interactions(
    company_name: str, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult:
    _ = company_name
    return CompaniesOrInteractionSearchResult(
        companies_search_result=CompanySearchResult(
            companies=[STATIC_COMPANY_SHORT],
            total=1,
            page=page,
            page_size=page_size,
        ),
        interactions_search_result=None,
    )


@mcp.tool(name="company_interactions", description="Static company interactions")
async def company_interactions(company_id: str, page_size: int = 10, page: int = 0) -> CompanyInteractionSearchResult:
    _ = company_id
    return CompanyInteractionSearchResult(
        interactions=[STATIC_INTERACTION],
        total=1,
        page=page,
        page_size=page_size,
    )


@mcp.tool(name="account_management_objectives", description="Static account objectives")
async def account_management_objectives(
    company_id: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult:
    _ = company_id
    return AccountManagementObjectivesSearchResult(
        account_management_objectives=[STATIC_OBJECTIVE],
        total=1,
        page=page,
        page_size=page_size,
    )


@mcp.tool(name="investment_projects", description="Static investment projects")
async def investment_projects(company_id: str, page_size: int = 10, page: int = 0) -> InvestmentProjectsSearchResult:
    _ = company_id
    return InvestmentProjectsSearchResult(
        investment_projects=[STATIC_INVESTMENT_PROJECT],
        total=1,
        page=page,
        page_size=page_size,
    )

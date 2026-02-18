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
    name="companies",
    tags={"data_hub", "companies", "search"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    """
    Query companies based on company name, returns a short overview of a list of companies

    Use this tool to find a company's unique ID or to get a high-level overview of multiple companies
    matching a specific name or keyword.

    Args:
        company_name: The name or partial name of the company to search for.
        page_size: Number of results to return per page (default: 10).
        page: Page number for paginated results (default: 0).

    Returns:
        CompanySearchResult: A structured object containing:
            - companies: List of CompanyShort objects, each with:
                - id (str): Unique UUID for the company
                - name (str): Full legal name
                - address_1, address_2, address_postcode, address_country (str): Location details
                - company_number (str): Registered company number
                - description (str): Brief summary or tagline
                - turnover_gbp (int): Annual turnover in GBP
            - total (int): Total matches found
            - page (int): Current page index
            - page_size (int): Number of items per page
    """
    return get_companies(company_name, page_size, page)


@mcp.tool(
    name="company_details",
    tags={"data_hub", "companies"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details(company_id: str) -> CompanyDetails | None:
    """
    Query full details of a company using its unique ID.

    Use this tool when you have a specific company ID (typically obtained from the 'companies'
    search tool) and need to access the full record.

    Args:
        company_id: The unique UUID of the company in the Data Hub database.
                  | Must be a valid ID string to return a result.

    Returns:
        CompanyDetails | None: A detailed object containing:
            - id (str): Unique identifier for the company
            - name (str): Full legal name
            - description (str): Brief summary of the business
            - website (str): Official company URL
            - address_1, address_2, address_town, address_county, address_postcode, address_country, address_area_name (str):
                Physical operating location details
            - registered_address_1, registered_address_2, registered_address_town, registered_address_county, registered_address_postcode, registered_address_country, registered_address_area_name (str):
                Official registered office location details
            - company_number (str): Companies House or equivalent registration number
            - uk_company_id (str): Internal UK-specific identifier
            - duns_number (str): The company's 9-digit D-U-N-S identification number
            - global_ultimate_duns_number (str): D-U-N-S number for the global ultimate parent
            - turnover (str), turnover_gbp (int), turnover_usd (int): Annual revenue in different currencies and formats
            - number_of_employees (int): Total staff count
            - sector (str): Primary industrial sector (e.g., 'Aerospace', 'Technology')
            - business_type (str): Legal structure (e.g., 'Private Limited Company')
            - headquarter_type (str): Indicates if it is a Global, UK, or Regional HQ
            - global_ultimate_hq (bool): Whether this is the top-level headquarters globally
            - uk_region (str): The primary UK region where the company is based
            - possible_uk_regions (list[str]): Other UK regions associated with the company
            - export_experience (str): Summary of historical export activity
            - export_segment, export_sub_segment (str): Classification for export potential
            - trading_names (list[str]): Other names the company operates under
            - is_out_of_business (bool): Status flag indicating if the company is no longer active
            - strategy (str): Internal notes on the engagement strategy for this company
            - eyb_lead_ids (str): Associated lead IDs from the Export Your Business system

    Examples:
        - Get full profile: company_id="550e8400-e29b-41d4-a716-446655440000"
    """
    return get_company(company_id)


@mcp.tool(
    name="company_details_extended",
    tags={"data_hub", "companies", "interactions", "objectives", "investment projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details_extended(
    company_id: str,
    include_interactions: bool = True,
    include_objectives: bool = True,
    include_investment_projects: bool = True,
) -> CompanyDetailsExtended:
    """
    Retrieve full details of a company, related interactions, account management
    objectives and investment projects

    Use this tool when you need a complete situational awareness of a company. It
    returns all related records for a single company ID.

    Args:
        company_id: The unique UUID of the company record in Data Hub.

    Returns:
        CompanyDetailsExtended | None: A unified object containing these nested components:

        1. company_details (CompanyDetails):
            - id (str): Unique identifier for the company
            - name (str): Full legal name
            - description (str): Brief summary of the business
            - website (str): Official company URL
            - address_1, address_2, address_town, address_county, address_postcode, address_country, address_area_name (str):
                Physical operating location details
            - registered_address_1, registered_address_2, registered_address_town, registered_address_county, registered_address_postcode, registered_address_country, registered_address_area_name (str):
                Official registered office location details
            - company_number (str): Companies House or equivalent registration number
            - uk_company_id (str): Internal UK-specific identifier
            - duns_number (str): The company's 9-digit D-U-N-S identification number
            - global_ultimate_duns_number (str): D-U-N-S number for the global ultimate parent
            - turnover (str), turnover_gbp (int), turnover_usd (int): Annual revenue in different currencies and formats
            - number_of_employees (int): Total staff count
            - sector (str): Primary industrial sector (e.g., 'Aerospace', 'Technology')
            - business_type (str): Legal structure (e.g., 'Private Limited Company')
            - headquarter_type (str): Indicates if it is a Global, UK, or Regional HQ
            - global_ultimate_hq (bool): Whether this is the top-level headquarters globally
            - uk_region (str): The primary UK region where the company is based
            - possible_uk_regions (list[str]): Other UK regions associated with the company
            - export_experience (str): Summary of historical export activity
            - export_segment, export_sub_segment (str): Classification for export potential
            - trading_names (list[str]): Other names the company operates under
            - is_out_of_business (bool): Status flag indicating if the company is no longer active
            - strategy (str): Internal notes on the engagement strategy for this company
            - eyb_lead_ids (str): Associated lead IDs from the Export Your Business system

        2. investment_projects (InvestmentProjectsSearchResult):
            - total (int), page (int), page_size (int): Pagination metadata.
            - investment_projects (list[InvestmentProject]):
                - id (str | None): Unique project identifier.
                - name (str | None): Title of the investment project.
                - project_reference (str | None): Internal reference string.
                - project_code (str | None): Official project tracking code.
                - status (str | None): Current project status (e.g., 'Won', 'Dormant').
                - stage (str | None): Current pipeline stage (e.g., 'Prospect', 'Active').
                - sector (str | None): Primary industrial sector of the project.
                - investment_type (str | None): The broad category of investment.
                - fdi_type (str | None): The specific type of Foreign Direct Investment.
                - fdi_value (str | None): Qualitative value of the FDI project.
                - total_investment (int | None): Total capital expenditure.
                - foreign_equity_investment (int | None): Value of foreign equity.
                - gross_value_added (int | None): Economic impact (GVA).
                - gva_multiplier (float | None): Multiplier used for GVA calculation.
                - number_new_jobs (int | None): Projected or actual new jobs created.
                - number_safeguarded_jobs (int | None): Number of existing jobs protected.
                - average_salary (str | None): Salary bracket for the created roles.
                - actual_land_date (date | None): Date the project officially landed.
                - estimated_land_date (date | None): Projected landing date.
                - proposal_deadline (date | None): Deadline for government proposals.
                - first_active_on (datetime | None): Date the project was first activated.
                - project_moved_to_won (datetime | None): Date the status changed to 'Won'.
                - project_arrived_in_triage_on (datetime | None): Date entered triage.
                - country_investment_originates_from (str | None): Source country.
                - competing_countries (list[str] | None): Other countries considered.
                - investor_company_id (str | None): ID of the investing entity.
                - investor_company_sector (str | None): Sector of the investor.
                - investor_type (str | None): Category of the investor.
                - uk_company_id (str | None): ID of the UK recipient company.
                - uk_company_sector (str | None): Sector of the UK company.
                - address_1, address_2, address_town, address_postcode (str | None): Project location.
                - actual_uk_regions (str | None): Final regional landing spots.
                - possible_uk_regions (list[str] | None): Regions under consideration.
                - description (str | None): Detailed project overview.
                - anonymous_description (str | None): Public-facing summary.
                - client_requirements (str | None): Specific needs of the investor.
                - strategic_drivers (list[str] | None): Key factors driving the investment.
                - business_activities (list[str] | None): Core activities involved.
                - delivery_partners (list[str] | None): Partner organizations involved.
                - specific_programme (list[str] | None): Named government programmes.
                - r_and_d_budget (bool | None): Whether the project has R&D focus.
                - new_tech_to_uk (bool | None): Whether it brings new tech to the UK.
                - export_revenue (bool | None): Whether the project generates export income.
                - government_assistance (bool | None): Whether gov funding was provided.
                - likelihood_to_land (str | None): Probability of project completion.
                - level_of_involvement (str | None): Level of gov support provided.
                - created_on, modified_on (datetime | None): System timestamps.

        3. account_management_objectives (AccountManagementObjectivesSearchResult):
            - total (int), page (int), page_size (int): Pagination metadata.
            - account_management_objectives (list[AccountManagementObjective]):
                - id (str | None): Unique identifier for the objective record.
                - company_id (str | None): The ID of the associated company.
                - subject (str | None): A brief title or summary of the objective.
                - detail (str | None): A comprehensive description of the goal and its requirements.
                - target_date (date | None): The deadline or expected completion date for the objective.
                - progress (int | None): Numerical percentage (0-100) representing completion status.
                - has_blocker (bool | None): Boolean flag indicating if progress is currently stalled.
                - blocker_description (str | None): Detailed explanation of what is obstructing the objective.
                - created_on (datetime | None): Timestamp when the objective was first recorded.
                - modified_on (datetime | None): Timestamp of the most recent update to the record.

        4. interactions (CompanyInteractionSearchResult):
            - total (int), page (int), page_size (int): Pagination metadata.
            - interactions (list[CompanyInteraction]):
                - id (str | None): Unique interaction record ID
                - company_id (str | None): ID of the associated company
                - interaction_date (date | None): When the event occurred
                - interaction_subject (str | None): Title or topic of the interaction
                - interaction_notes (str | None): Detailed meeting or call notes
                - interaction_type (str | None): Format (e.g., 'Meeting', 'Email')
                - interaction_kind (str | None): Category of engagement
                - theme (str | None): Strategic theme (e.g., 'Export', 'Investment')
                - sector (str | None): Industry sector discussed
                - communication_channel (str | None): Medium (e.g., 'Video call', 'Email')
                - created_on, modified_on (datetime | None): Record timestamps
                - policy_areas, policy_issue_types (list[str] | None): Policy-specific details
                - related_trade_agreement_names (list[str] | None): Linked trade deals
                - export_barrier_type_names (list[str] | None): Identified trade barriers
                - export_barrier_notes (str | None): Detailed barrier descriptions
                - were_countries_discussed (bool | None): Flag for geographic focus
                - service_delivery, service_delivery_status (str | None): Service tracking
                - investment_project_id (str | None): ID for linked FDI projects
                - dbt_initiative (str | None): Related government initiative
                - net_company_receipt (float | None): Financial impact value
    """
    return get_company_extended(company_id, include_interactions, include_objectives, include_investment_projects)


@mcp.tool(
    name="companies_or_interactions",
    tags={"data_hub", "companies_or_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies_or_interactions(
    company_name: str, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult:
    """
    Search for companies by name and retrieve either a consolidated list of matches or a single match and
    a detailed interaction history.

    Use this tool when a user provides a company name. It is optimized to return a list of 'CompanyShort'
    results if the search is broad, or 'CompanyInteraction' records if the search identifies a
    single company.

    Args:
        company_name: The company name or keyword to search across both profiles and interactions.
                    | Performs a search against company names and interaction subjects.
        page_size: The number of results to return per category (default: 10).
        page: The page number to retrieve for paginated results (default: 0).

    Returns:
        CompaniesOrInteractionSearchResult: A combined object containing:

        - companies_search_result (CompanySearchResult | None):
            Populated with matching company profiles. Contains:
            - companies (list[CompanyShort]):
                - id (str | None): Unique identifier
                - name (str): Registered company name
                - description (str | None): Brief business summary
                - address_1, address_2 (str | None): Primary location details
                - address_postcode, address_country (str | None): Geographic markers
                - company_number (str | None): Official registration number
                - turnover_gbp (int | None): Annual turnover in GBP
            - total (int): Total company matches found
            - page, page_size (int): Pagination metadata

        - interactions_search_result (CompanyInteractionSearchResult | None):
            Populated with relevant engagement history. Contains:
            - interactions (list[CompanyInteraction]):
                - id (str | None): Unique interaction record ID
                - company_id (str | None): ID of the associated company
                - interaction_date (date | None): When the event occurred
                - interaction_subject (str | None): Title or topic of the interaction
                - interaction_notes (str | None): Detailed meeting or call notes
                - interaction_type (str | None): Format (e.g., 'Meeting', 'Email')
                - interaction_kind (str | None): Category of engagement
                - theme (str | None): Strategic theme (e.g., 'Export', 'Investment')
                - sector (str | None): Industry sector discussed
                - communication_channel (str | None): Medium (e.g., 'Video call', 'Email')
                - created_on, modified_on (datetime | None): Record timestamps
                - policy_areas, policy_issue_types (list[str] | None): Policy-specific details
                - related_trade_agreement_names (list[str] | None): Linked trade deals
                - export_barrier_type_names (list[str] | None): Identified trade barriers
                - export_barrier_notes (str | None): Detailed barrier descriptions
                - were_countries_discussed (bool | None): Flag for geographic focus
                - service_delivery, service_delivery_status (str | None): Service tracking
                - investment_project_id (str | None): ID for linked FDI projects
                - dbt_initiative (str | None): Related government initiative
                - net_company_receipt (float | None): Financial impact value
            - total (int): Total interaction matches found
            - page, page_size (int): Pagination metadata
    """
    return get_companies_or_interactions(company_name, page_size, page)


@mcp.tool(
    name="company_interactions",
    tags={"data_hub", "company_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_interactions(company_id: str, page_size: int = 10, page: int = 0) -> CompanyInteractionSearchResult:
    """
    Query company interactions based on company id.

    Use this tool when you have a specific company ID (typically obtained from the 'companies'
    search tool) and need to access related interaction records.

    Args:
        company_id: The unique UUID of the company to retrieve interactions for.
                  | Obtained from 'companies' or 'companies_or_interactions' search.
        page_size: Number of interaction records to return per page (default: 10).
        page: Page number for paginated results (default: 0).

    Returns:
        CompanyInteractionSearchResult: A structured object containing:
            - interactions: A list of CompanyInteraction objects, each containing:
                - id (str | None): Unique identifier for the interaction record.
                - company_id (str | None): ID of the company involved.
                - interaction_date (date | None): The date the interaction took place.
                - interaction_subject (str | None): Short summary/title of the interaction.
                - interaction_notes (str | None): Detailed description or minutes of the interaction.
                - interaction_type (str | None): The format of the interaction (e.g., 'Meeting').
                - interaction_kind (str | None): The nature of the interaction.
                - theme (str | None): Strategic theme (e.g., 'Export', 'Investment').
                - sector (str | None): The industry sector discussed.
                - communication_channel (str | None): Medium used (e.g., 'Email', 'Video call').
                - contact (str | None): Name of the company representative involved.
                - adviser (str | None): Name of the government adviser who led the interaction.
                - created_on (datetime | None): Timestamp of record creation.
                - modified_on (datetime | None): Timestamp of last update.
                - policy_areas (list[str] | None): Specific policy sectors discussed.
                - policy_issue_types (list[str] | None): Types of policy issues raised.
                - policy_feedback_notes (str | None): Detailed notes regarding policy feedback.
                - were_countries_discussed (bool | None): Whether specific geographic markets were mentioned.
                - related_trade_agreement_names (list[str] | None): Trade agreements relevant to the discussion.
                - export_barrier_type_names (list[str] | None): Categories of trade barriers identified.
                - export_barrier_notes (str | None): Specific details on identified export hurdles.
                - export_challenge_type (str | None): Classification of export challenges.
                - dbt_initiative (str | None): Related government initiative or campaign.
                - service_delivery (str | None): The specific service provided during interaction.
                - service_delivery_status (str | None): Status of the service delivery.
                - investment_project_id (str | None): Linked ID if related to an investment project.
                - company_export_id (str | None): Linked ID for export-specific tracking.
                - interaction_link (str | None): URL to the full record in the Data Hub UI.
                - net_company_receipt (float | None): Financial value associated with the interaction.
            - total (int): Total number of records.
            - page (int): Current page index.
            - page_size (int): Number of items per page.
    """
    return get_company_interactions(company_id, page_size, page)


@mcp.tool(
    name="account_management_objectives",
    tags={"data_hub", "company", "account_management_objectives"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def account_management_objectives(
    company_id: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult:
    """
    Query account management objectives based on company id

    Use this tool when you have a specific company ID (typically obtained from the 'companies'
    search tool) and need to access related account management objective records.

    Args:
        company_id: The unique UUID of the company in Data Hub.
                  | Required to filter objectives to a specific entity.
        page_size: The number of records to return per page (default: 10).
        page: The page number for pagination (default: 0).

    Returns:
        AccountManagementObjectivesSearchResult: An object containing:
            - account_management_objectives: A list of AccountManagementObjective objects, each containing:
                - id (str | None): Unique identifier for the objective record.
                - company_id (str | None): The ID of the associated company.
                - subject (str | None): A brief title or summary of the objective.
                - detail (str | None): A comprehensive description of the goal and its requirements.
                - target_date (date | None): The deadline or expected completion date for the objective.
                - progress (int | None): Numerical percentage (0-100) representing completion status.
                - has_blocker (bool | None): Boolean flag indicating if progress is currently stalled.
                - blocker_description (str | None): Detailed explanation of what is obstructing the objective.
                - created_on (datetime | None): Timestamp when the objective was first recorded.
                - modified_on (datetime | None): Timestamp of the most recent update to the record.
            - total (int): Total number of records.
            - page (int): The current page index.
            - page_size (int): The number of results per page.
    """
    return get_account_management_objectives(company_id, page_size, page)


@mcp.tool(
    name="investment_projects",
    tags={"data_hub", "company", "investment_projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def investment_projects(company_id: str, page_size: int = 10, page: int = 0) -> InvestmentProjectsSearchResult:
    """
    Query investment projects based on company id

    Use this tool when you have a specific company ID (typically obtained from the 'companies'
    search tool) and need to access related investment project records.

    Args:
        company_id: The unique UUID of the investor or target company.
        page_size: Number of project records to return per page (default: 10).
        page: Page number for pagination (default: 0).

    Returns:
        InvestmentProjectsSearchResult: An object containing:
            - investment_projects: A list of InvestmentProject objects, each containing:
                - id (str | None): Unique project identifier.
                - name (str | None): Title of the investment project.
                - project_reference (str | None): Internal reference string.
                - project_code (str | None): Official project tracking code.
                - status (str | None): Current project status (e.g., 'Won', 'Dormant').
                - stage (str | None): Current pipeline stage (e.g., 'Prospect', 'Active').
                - sector (str | None): Primary industrial sector of the project.
                - investment_type (str | None): The broad category of investment.
                - fdi_type (str | None): The specific type of Foreign Direct Investment.
                - fdi_value (str | None): Qualitative value of the FDI project.
                - total_investment (int | None): Total capital expenditure.
                - foreign_equity_investment (int | None): Value of foreign equity.
                - gross_value_added (int | None): Economic impact (GVA).
                - gva_multiplier (float | None): Multiplier used for GVA calculation.
                - number_new_jobs (int | None): Projected or actual new jobs created.
                - number_safeguarded_jobs (int | None): Number of existing jobs protected.
                - average_salary (str | None): Salary bracket for the created roles.
                - actual_land_date (date | None): Date the project officially landed.
                - estimated_land_date (date | None): Projected landing date.
                - proposal_deadline (date | None): Deadline for government proposals.
                - first_active_on (datetime | None): Date the project was first activated.
                - project_moved_to_won (datetime | None): Date the status changed to 'Won'.
                - project_arrived_in_triage_on (datetime | None): Date entered triage.
                - country_investment_originates_from (str | None): Source country.
                - competing_countries (list[str] | None): Other countries considered.
                - investor_company_id (str | None): ID of the investing entity.
                - investor_company_sector (str | None): Sector of the investor.
                - investor_type (str | None): Category of the investor.
                - uk_company_id (str | None): ID of the UK recipient company.
                - uk_company_sector (str | None): Sector of the UK company.
                - address_1, address_2, address_town, address_postcode (str | None): Project location.
                - actual_uk_regions (str | None): Final regional landing spots.
                - possible_uk_regions (list[str] | None): Regions under consideration.
                - description (str | None): Detailed project overview.
                - anonymous_description (str | None): Public-facing summary.
                - client_requirements (str | None): Specific needs of the investor.
                - strategic_drivers (list[str] | None): Key factors driving the investment.
                - business_activities (list[str] | None): Core activities involved.
                - delivery_partners (list[str] | None): Partner organizations involved.
                - specific_programme (list[str] | None): Named government programmes.
                - r_and_d_budget (bool | None): Whether the project has R&D focus.
                - new_tech_to_uk (bool | None): Whether it brings new tech to the UK.
                - export_revenue (bool | None): Whether the project generates export income.
                - government_assistance (bool | None): Whether gov funding was provided.
                - likelihood_to_land (str | None): Probability of project completion.
                - level_of_involvement (str | None): Level of gov support provided.
                - created_on, modified_on (datetime | None): System timestamps.
            - total (int): Total projects found for this company.
            - page, page_size (int): Pagination metadata.
    """
    return get_investment_projects(company_id, page_size, page)

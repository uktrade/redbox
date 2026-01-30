from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

# ============================================================
# Search / Quality Metadata
# ============================================================


class MCPToolResultType(StrEnum):
    SINGLE_EXACT_MATCH = "A single result was returned with an exact match"
    MULTI_EXACT_MATCH = "Multiple results were returned with an exact match"
    SINGLE_SIM_MATCH = "A single result was returned with a similar ILIKE match"
    MULTI_SIM_MATCH = "Multiple results were returned with a similar ILIKE match"
    NONE = "No results found with exact match or similar ILIKE match"


# ============================================================
# Company Models
# ============================================================


@dataclass
class CompanyShort:
    """
    Short representation of a company (search result / list view)
    """

    id: str | None
    name: str
    description: str | None
    company_number: str | None
    turnover_gbp: int | None

    address_1: str | None
    address_2: str | None
    address_postcode: str | None
    address_country: str | None

    @staticmethod
    def populate_from_record(record: dict) -> "CompanyShort":
        return CompanyShort(
            id=record.get("id"),
            name=record.get("name"),
            description=record.get("description"),
            company_number=record.get("company_number"),
            turnover_gbp=record.get("turnover_gbp"),
            address_1=record.get("address_1"),
            address_2=record.get("address_2"),
            address_postcode=record.get("address_postcode"),
            address_country=record.get("address_country"),
        )


@dataclass
class CompanyDetails:
    """
    Full representation of a company (detail / enrichment view)
    """

    id: str | None
    name: str
    description: str | None

    company_number: str | None
    vat_number: str | None
    duns_number: str | None
    global_ultimate_duns_number: str | None

    business_type: str | None
    sector: str | None
    export_segment: str | None
    export_sub_segment: str | None
    export_experience: str | None

    turnover: str | None
    turnover_gbp: int | None
    turnover_usd: int | None
    is_turnover_estimated: bool | None

    number_of_employees: int | None
    is_number_of_employees_estimated: bool | None

    website: str | None
    trading_names: list[str] | None
    strategy: str | None

    headquarter_type: str | None
    uk_region: str | None
    is_out_of_business: bool | None

    address_1: str | None
    address_2: str | None
    address_town: str | None
    address_county: str | None
    address_postcode: str | None
    address_country: str | None
    address_area_name: str | None

    registered_address_1: str | None
    registered_address_2: str | None
    registered_address_town: str | None
    registered_address_county: str | None
    registered_address_postcode: str | None
    registered_address_country: str | None
    registered_address_area_name: str | None

    @staticmethod
    def populate_from_record(record: dict) -> "CompanyDetails":
        return CompanyDetails(
            id=record.get("id"),
            name=record.get("name"),
            description=record.get("description"),
            company_number=record.get("company_number"),
            vat_number=record.get("vat_number"),
            duns_number=record.get("duns_number"),
            global_ultimate_duns_number=record.get("global_ultimate_duns_number"),
            business_type=record.get("business_type"),
            sector=record.get("sector"),
            export_segment=record.get("export_segment"),
            export_sub_segment=record.get("export_sub_segment"),
            export_experience=record.get("export_experience"),
            turnover=record.get("turnover"),
            turnover_gbp=record.get("turnover_gbp"),
            turnover_usd=record.get("turnover_usd"),
            is_turnover_estimated=record.get("is_turnover_estimated"),
            number_of_employees=record.get("number_of_employees"),
            is_number_of_employees_estimated=record.get("is_number_of_employees_estimated"),
            website=record.get("website"),
            trading_names=record.get("trading_names"),
            strategy=record.get("strategy"),
            headquarter_type=record.get("headquarter_type"),
            uk_region=record.get("uk_region"),
            is_out_of_business=record.get("is_out_of_business"),
            address_1=record.get("address_1"),
            address_2=record.get("address_2"),
            address_town=record.get("address_town"),
            address_county=record.get("address_county"),
            address_postcode=record.get("address_postcode"),
            address_country=record.get("address_country"),
            address_area_name=record.get("address_area_name"),
            registered_address_1=record.get("registered_address_1"),
            registered_address_2=record.get("registered_address_2"),
            registered_address_town=record.get("registered_address_town"),
            registered_address_county=record.get("registered_address_county"),
            registered_address_postcode=record.get("registered_address_postcode"),
            registered_address_country=record.get("registered_address_country"),
            registered_address_area_name=record.get("registered_address_area_name"),
        )


# ============================================================
# Related Domain Models
# ============================================================


@dataclass
class CompanyInteraction:
    team_region: str | None
    team_country: str | None
    interaction_year: int | None
    interaction_financial_year: str | None
    company_name: str | None
    company_sector: str | None
    date_of_interaction: date | None
    interaction_subject: str | None
    interaction_theme_investment_or_export: str | None

    @staticmethod
    def populate_from_record(record: dict) -> "CompanyInteraction":
        return CompanyInteraction(
            team_region=record.get("team_region"),
            team_country=record.get("team_country"),
            interaction_year=record.get("interaction_year"),
            interaction_financial_year=record.get("interaction_financial_year"),
            company_name=record.get("company_name"),
            company_sector=record.get("company_sector"),
            date_of_interaction=record.get("date_of_interaction"),
            interaction_subject=record.get("interaction_subject"),
            interaction_theme_investment_or_export=record.get("interaction_theme_investment_or_export"),
        )


@dataclass
class AccountManagementObjective:
    id: str | None
    subject: str | None
    detail: str | None
    target_date: date | None
    has_blocker: bool | None
    blocker_description: str | None
    progress: int | None
    created_on: datetime | None
    modified_on: datetime | None
    created_by_id: str | None
    modified_by_id: str | None
    company_id: str | None

    @staticmethod
    def populate_from_record(record: dict) -> "AccountManagementObjective":
        return AccountManagementObjective(
            id=record.get("id"),
            subject=record.get("subject"),
            detail=record.get("detail"),
            target_date=record.get("target_date"),
            has_blocker=record.get("has_blocker"),
            blocker_description=record.get("blocker_description"),
            progress=record.get("progress"),
            created_on=record.get("created_on"),
            modified_on=record.get("modified_on"),
            created_by_id=record.get("created_by_id"),
            modified_by_id=record.get("modified_by_id"),
            company_id=record.get("company_id"),
        )


@dataclass
class InvestmentProject:
    id: str | None
    name: str | None
    description: str | None

    investment_type: str | None
    investor_type: str | None
    stage: str | None
    status: str | None
    likelihood_to_land: str | None

    sector: str | None
    uk_company_sector: str | None
    investor_company_sector: str | None

    uk_company_id: str | None
    investor_company_id: str | None

    estimated_land_date: date | None
    actual_land_date: date | None

    total_investment: int | None
    foreign_equity_investment: int | None
    gross_value_added: int | None

    number_new_jobs: int | None
    number_safeguarded_jobs: int | None

    created_on: datetime | None
    modified_on: datetime | None

    @staticmethod
    def populate_from_record(record: dict) -> "InvestmentProject":
        return InvestmentProject(
            id=record.get("id"),
            name=record.get("name"),
            description=record.get("description"),
            investment_type=record.get("investment_type"),
            investor_type=record.get("investor_type"),
            stage=record.get("stage"),
            status=record.get("status"),
            likelihood_to_land=record.get("likelihood_to_land"),
            sector=record.get("sector"),
            uk_company_sector=record.get("uk_company_sector"),
            investor_company_sector=record.get("investor_company_sector"),
            uk_company_id=record.get("uk_company_id"),
            investor_company_id=record.get("investor_company_id"),
            estimated_land_date=record.get("estimated_land_date"),
            actual_land_date=record.get("actual_land_date"),
            total_investment=record.get("total_investment"),
            foreign_equity_investment=record.get("foreign_equity_investment"),
            gross_value_added=record.get("gross_value_added"),
            number_new_jobs=record.get("number_new_jobs"),
            number_safeguarded_jobs=record.get("number_safeguarded_jobs"),
            created_on=record.get("created_on"),
            modified_on=record.get("modified_on"),
        )


# ============================================================
# Search Result Wrappers
# ============================================================


@dataclass
class CompanySearchResult:
    result_type: MCPToolResultType
    companies: list[CompanyShort]
    total: int
    page: int
    page_size: int


@dataclass
class CompanyInteractionSearchResult:
    interactions: list[CompanyInteraction]
    total: int
    page: int
    page_size: int

    @staticmethod
    def load(
        interactions: list[CompanyInteraction], page: int = 0, page_size: int | None = None
    ) -> "CompanyInteractionSearchResult":
        """
        Factory method to create a search result from a list of interactions.
        Automatically computes total and handles pagination.
        """
        total = len(interactions)
        if page_size is None:
            page_size = total  # default to all items if not specified

        start = page * page_size
        end = start + page_size
        paged_items = interactions[start:end]

        return CompanyInteractionSearchResult(
            interactions=paged_items,
            total=total,
            page=page,
            page_size=page_size,
        )


@dataclass
class AccountManagementObjectivesSearchResult:
    account_management_objectives: list[AccountManagementObjective]
    total: int
    page: int
    page_size: int

    @staticmethod
    def load(
        objectives: list[AccountManagementObjective], page: int = 0, page_size: int | None = None
    ) -> "AccountManagementObjectivesSearchResult":
        """
        Factory method to create a search result from a list of objectives.
        Automatically computes total and handles pagination.
        """
        total = len(objectives)
        if page_size is None:
            page_size = total  # default to all items if not specified

        start = page * page_size
        end = start + page_size
        paged_items = objectives[start:end]

        return AccountManagementObjectivesSearchResult(
            account_management_objectives=paged_items,
            total=total,
            page=page,
            page_size=page_size,
        )


@dataclass
class InvestmentProjectsSearchResult:
    investment_projects: list[InvestmentProject]
    total: int
    page: int
    page_size: int

    @staticmethod
    def load(
        investment_projects: list[InvestmentProject], page: int = 0, page_size: int | None = None
    ) -> "InvestmentProjectsSearchResult":
        """
        Factory method to create a search result from a list of investment projects.
        Automatically computes total and handles pagination.
        """
        total = len(investment_projects)
        if page_size is None:
            page_size = total  # default to all items if not specified

        start = page * page_size
        end = start + page_size
        paged_items = investment_projects[start:end]

        return InvestmentProjectsSearchResult(
            investment_projects=paged_items,
            total=total,
            page=page,
            page_size=page_size,
        )


# ============================================================
# Fully Enriched MCP Result
# ============================================================


@dataclass
class CompanyEnrichmentResult:
    # result_type: MCPToolResultType
    company: CompanyDetails
    interactions: CompanyInteractionSearchResult
    objectives: AccountManagementObjectivesSearchResult
    investment_projects: InvestmentProjectsSearchResult


@dataclass
class CompanyEnrichmentSearchResult:
    result_type: MCPToolResultType
    companies: list[CompanyEnrichmentResult]
    total: int
    page: int
    page_size: int


@dataclass
class SectorCompanyGroup:
    sector: str
    companies: list[CompanyEnrichmentResult]
    total: int


@dataclass
class SectorGroupedCompanySearchResult:
    result_type: MCPToolResultType
    sectors: list[SectorCompanyGroup]
    total_sectors: int
    total_companies: int
    page: int
    page_size: int


@dataclass
class SectorOverviewGroup:
    sector: str
    total_companies: int
    total_turnover_gbp: float | None
    total_employees: int | None
    total_investment_gbp: float | None
    total_gva: float | None
    average_turnover_gbp: float | None
    average_employees: float | None
    average_investment_gbp: float | None
    average_gva: float | None


@dataclass
class SectorGroupedOverview:
    sectors: list[SectorOverviewGroup]
    total_sectors: int


@dataclass
class SectorInvestmentProject:
    project_name: str
    company_name: str
    uk_company_id: str | None
    investor_company_id: str | None
    investment_type: str
    stage: str
    status: str
    likelihood_to_land: str
    total_investment: float | None
    foreign_equity_investment: float | None
    gross_value_added: float | None
    number_new_jobs: int | None
    number_safeguarded_jobs: int | None


@dataclass
class SectorInvestmentGroup:
    sector: str
    total_projects: int
    total_investment: float | None
    total_gva: float | None
    total_new_jobs: int | None
    total_safeguarded_jobs: int | None
    projects: list[SectorInvestmentProject]


@dataclass
class SectorGroupedInvestmentSummary:
    sectors: list[SectorInvestmentGroup]
    total_sectors: int
    total_projects: int
    page: int
    page_size: int


@dataclass
class SectorCompanyPerformance:
    company_name: str
    turnover_gbp: int | None
    turnover_usd: int | None
    number_of_employees: int | None
    export_segment: str | None
    export_sub_segment: str | None
    growth_turnover: float | None
    growth_employees: float | None

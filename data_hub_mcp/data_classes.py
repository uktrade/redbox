from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class CompanyShort:
    """
    Class for representing company, the short version
    """

    id: str | None
    address_1: str | None
    address_2: str | None
    address_postcode: str | None
    address_country: str | None
    company_number: str | None
    description: str | None
    name: str
    turnover_gbp: int | None

    @staticmethod
    def populate_from_record(record):
        return CompanyShort(
            id=record["id"],
            address_1=record["address_1"],
            address_2=record["address_2"],
            address_postcode=record["address_postcode"],
            address_country=record["address_country"],
            company_number=record["company_number"],
            description=record["description"],
            name=record["name"],
            turnover_gbp=record["turnover_gbp"],
        )


@dataclass
class CompanyDetails:
    """
    Class for representing a company, the detailed version
    """

    address_1: str | None
    address_2: str | None
    address_county: str | None
    address_postcode: str | None
    address_country: str | None
    address_area_name: str | None
    address_town: str | None
    company_number: str | None
    description: str | None
    name: str
    turnover_gbp: int | None
    business_type: str | None
    duns_number: str | None
    export_experience: str | None
    global_ultimate_duns_number: str | None
    headquarter_type: str | None
    id: str | None
    is_number_of_employees_estimated: bool | None
    is_turnover_estimated: bool | None
    number_of_employees: int | None
    registered_address_1: str | None
    registered_address_2: str | None
    registered_address_country: str | None
    registered_address_county: str | None
    registered_address_postcode: str | None
    registered_address_area_name: str | None
    registered_address_town: str | None
    sector: str | None
    export_segment: str | None
    export_sub_segment: str | None
    trading_names: list[str] | None
    turnover: str | None
    turnover_usd: int | None
    uk_region: str | None
    vat_number: str | None
    website: str | None
    is_out_of_business: bool | None
    strategy: str | None

    @staticmethod
    def populate_from_record(record):
        return CompanyDetails(
            address_1=record["address_1"],
            address_2=record["address_2"],
            address_county=record["address_county"],
            address_postcode=record["address_postcode"],
            address_country=record["address_country"],
            address_area_name=record["address_area_name"],
            address_town=record["address_town"],
            company_number=record["company_number"],
            description=record["description"],
            name=record["name"],
            turnover_gbp=record["turnover_gbp"],
            business_type=record["business_type"],
            duns_number=record["duns_number"],
            export_experience=record["export_experience"],
            global_ultimate_duns_number=record["global_ultimate_duns_number"],
            headquarter_type=record["headquarter_type"],
            id=record["id"],
            is_number_of_employees_estimated=record["is_number_of_employees_estimated"],
            is_turnover_estimated=record["is_turnover_estimated"],
            number_of_employees=record["number_of_employees"],
            registered_address_1=record["registered_address_1"],
            registered_address_2=record["registered_address_2"],
            registered_address_country=record["registered_address_country"],
            registered_address_county=record["registered_address_county"],
            registered_address_postcode=record["registered_address_postcode"],
            registered_address_area_name=record["registered_address_area_name"],
            registered_address_town=record["registered_address_town"],
            sector=record["sector"],
            export_segment=record["export_segment"],
            export_sub_segment=record["export_sub_segment"],
            trading_names=record["trading_names"],
            turnover=record["turnover"],
            turnover_usd=record["turnover_usd"],
            uk_region=record["uk_region"],
            vat_number=record["vat_number"],
            website=record["website"],
            is_out_of_business=record["is_out_of_business"],
            strategy=record["strategy"],
        )


@dataclass
class CompanyInteraction:
    """
    Class for representing a company interaction
    """

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
    def populate_from_record(record):
        return CompanyInteraction(
            team_region=record["team_region"],
            team_country=record["team_country"],
            interaction_year=record["interaction_year"],
            interaction_financial_year=record["interaction_financial_year"],
            company_name=record["company_name"],
            company_sector=record["company_sector"],
            date_of_interaction=record["date_of_interaction"],
            interaction_subject=record["interaction_subject"],
            interaction_theme_investment_or_export=record["interaction_theme_investment_or_export"],
        )


@dataclass
class AccountManagementObjective:
    """
    Class for representing an account management objective
    """

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
    def populate_from_record(record):
        return AccountManagementObjective(
            id=record["id"],
            subject=record["subject"],
            detail=record["detail"],
            target_date=record["target_date"],
            has_blocker=record["has_blocker"],
            blocker_description=record["blocker_description"],
            progress=record["progress"],
            created_on=record["created_on"],
            modified_on=record["modified_on"],
            created_by_id=record["created_by_id"],
            modified_by_id=record["modified_by_id"],
            company_id=record["company_id"],
        )


@dataclass
class InvestmentProject:
    """
    Class for representing an account management objective
    """

    actual_land_date: date | None
    actual_uk_regions: str | None
    address_1: str | None
    address_2: str | None
    address_county: str | None
    address_country: str | None
    address_town: str | None
    address_postcode: str | None
    anonymous_description: str | None
    associated_non_fdi_r_and_d_project_id: str | None
    average_salary: str | None
    business_activities: list[str] | None

    client_relationship_manager_id: str | None
    client_requirements: str | None
    client_contact_ids: list[str] | None
    client_contact_names: list[str] | None
    client_contact_emails: list[str] | None
    competing_countries: list[str] | None
    country_investment_originates_from: str | None
    created_by_id: str | None
    created_on: datetime | None
    delivery_partners: list[str] | None
    description: str | None
    estimated_land_date: date | None

    export_revenue: bool | None
    first_active_on: datetime | None
    fdi_type: str | None
    fdi_value: str | None
    foreign_equity_investment: int | None
    government_assistance: bool | None
    gross_value_added: int | None
    gva_multiplier: float | None
    id: str | None
    investment_type: str | None
    investor_company_id: str | None
    investor_company_sector: str | None
    investor_type: str | None
    level_of_involvement: str | None
    likelihood_to_land: str | None

    modified_by_id: str | None
    modified_on: datetime | None
    name: str | None
    new_tech_to_uk: bool | None
    non_fdi_r_and_d_budget: bool | None
    number_new_jobs: int | None
    number_safeguarded_jobs: int | None
    other_business_activity: str | None
    project_arrived_in_triage_on: datetime | None
    project_assurance_adviser_id: str | None
    project_moved_to_won: datetime | None
    project_manager_id: str | None
    project_reference: str | None
    proposal_deadline: date | None
    r_and_d_budget: bool | None
    referral_source_activity: str | None
    referral_source_activity_marketing: str | None
    referral_source_activity_website: str | None
    sector: str | None
    specific_programme: list[str] | None
    stage: str | None
    status: str | None
    strategic_drivers: list[str] | None
    team_member_ids: list[str] | None
    total_investment: int | None
    uk_company_sector: str | None
    possible_uk_regions: list[str] | None
    eyb_lead_ids: str | None

    uk_company_id: str | None

    @staticmethod
    def populate_from_record(record):
        return InvestmentProject(
            actual_land_date=record["actual_land_date"],
            actual_uk_regions=record["actual_uk_regions"],
            address_1=record["address_1"],
            address_2=record["address_2"],
            address_county=record["address_county"],
            address_country=record["address_country"],
            address_town=record["address_town"],
            address_postcode=record["address_postcode"],
            anonymous_description=record["anonymous_description"],
            associated_non_fdi_r_and_d_project_id=record["associated_non_fdi_r_and_d_project_id"],
            average_salary=record["average_salary"],
            business_activities=record["business_activities"],
            client_relationship_manager_id=record["client_relationship_manager_id"],
            client_requirements=record["client_requirements"],
            client_contact_ids=record["client_contact_ids"],
            client_contact_names=record["client_contact_names"],
            client_contact_emails=record["client_contact_emails"],
            competing_countries=record["competing_countries"],
            country_investment_originates_from=record["country_investment_originates_from"],
            created_by_id=record["created_by_id"],
            created_on=record["created_on"],
            delivery_partners=record["delivery_partners"],
            description=record["description"],
            estimated_land_date=record["estimated_land_date"],
            export_revenue=record["export_revenue"],
            first_active_on=record["first_active_on"],
            fdi_type=record["fdi_type"],
            fdi_value=record["fdi_value"],
            foreign_equity_investment=record["foreign_equity_investment"],
            government_assistance=record["government_assistance"],
            gross_value_added=record["gross_value_added"],
            gva_multiplier=record["gva_multiplier"],
            id=record["id"],
            investment_type=record["investment_type"],
            investor_company_id=record["investor_company_id"],
            investor_company_sector=record["investor_company_sector"],
            investor_type=record["investor_type"],
            level_of_involvement=record["level_of_involvement"],
            likelihood_to_land=record["likelihood_to_land"],
            modified_by_id=record["modified_by_id"],
            modified_on=record["modified_on"],
            name=record["name"],
            new_tech_to_uk=record["new_tech_to_uk"],
            non_fdi_r_and_d_budget=record["non_fdi_r_and_d_budget"],
            number_new_jobs=record["number_new_jobs"],
            number_safeguarded_jobs=record["number_safeguarded_jobs"],
            other_business_activity=record["other_business_activity"],
            project_arrived_in_triage_on=record["project_arrived_in_triage_on"],
            project_assurance_adviser_id=record["project_assurance_adviser_id"],
            project_moved_to_won=record["project_moved_to_won"],
            project_manager_id=record["project_manager_id"],
            project_reference=record["project_reference"],
            proposal_deadline=record["proposal_deadline"],
            r_and_d_budget=record["r_and_d_budget"],
            referral_source_activity=record["referral_source_activity"],
            referral_source_activity_marketing=record["referral_source_activity_marketing"],
            referral_source_activity_website=record["referral_source_activity_website"],
            sector=record["sector"],
            specific_programme=record["specific_programme"],
            stage=record["stage"],
            status=record["status"],
            strategic_drivers=record["strategic_drivers"],
            team_member_ids=record["team_member_ids"],
            total_investment=record["total_investment"],
            uk_company_sector=record["uk_company_sector"],
            possible_uk_regions=record["possible_uk_regions"],
            eyb_lead_ids=record["eyb_lead_ids"],
            uk_company_id=record["uk_company_id"],
        )


@dataclass
class CompanySearchResult:
    """Company search result"""

    companies: list[CompanyShort]
    total: int
    page: int
    page_size: int


@dataclass
class CompanyInteractionSearchResult:
    """Company Interactions search result"""

    interactions: list[CompanyInteraction]
    total: int
    page: int
    page_size: int


@dataclass
class CompaniesOrInteractionSearchResult:
    """Company Interactions search result"""

    companies_search_result: CompanySearchResult | None
    interactions_search_result: CompanyInteractionSearchResult | None


@dataclass
class AccountManagementObjectivesSearchResult:
    """Account Management Objectives search result"""

    account_management_objectives: list[AccountManagementObjective]
    total: int
    page: int
    page_size: int


@dataclass
class InvestmentProjectsSearchResult:
    """Investment Projects search result"""

    investment_projects: list[InvestmentProject]
    total: int
    page: int
    page_size: int


@dataclass
class CompanyDetailsExtended(CompanyDetails):
    investment_projects: list[InvestmentProject]
    interactions: list[CompanyInteraction]
    account_management_objectives: list[AccountManagementObjective]

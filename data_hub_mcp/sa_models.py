from sqlalchemy import (
    ARRAY,
    UUID,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def get_schema():
    return "data_hub"


class Company(Base):
    __tablename__ = "companies"
    __table_args__ = {"schema": get_schema()}  # noqa: RUF012

    address_1 = Column("address_1", String)
    address_2 = Column("address_2", String)
    address_county = Column("address_county", String(100))
    address_country = Column("address_country", String(200))
    address_postcode = Column("address_postcode", String(200))
    address_area_name = Column("address_area_name", String(200))
    address_town = Column("address_town", String(200))
    archived = Column("archived", Boolean)
    archived_on = Column("archived_on", DateTime, nullable=True)
    business_type = Column("business_type", String(200))
    company_number = Column("company_number", String(200))
    created_on = Column("created_on", Date)
    description = Column("description", String)
    duns_number = Column("duns_number", String(30))
    export_experience = Column("export_experience", String(200))
    global_headquarters_id = Column("global_headquarters_id", UUID, nullable=True)
    global_ultimate_duns_number = Column("global_ultimate_duns_number", String(200))
    headquarter_type = Column("headquarter_type", String(200))
    id = Column("id", UUID, primary_key=True)
    is_number_of_employees_estimated = Column("is_number_of_employees_estimated", Boolean, nullable=True)
    is_turnover_estimated = Column("is_turnover_estimated", Boolean, nullable=True)
    modified_on = Column("modified_on", DateTime, nullable=True)
    name = Column("name", String(250))
    number_of_employees = Column("number_of_employees", Integer, nullable=True)
    one_list_account_owner_id = Column("one_list_account_owner_id", UUID, nullable=True)
    one_list_core_team_adviser_ids = Column("one_list_core_team_adviser_ids", ARRAY(UUID))
    one_list_tier = Column("one_list_tier", String(100))
    cdms_reference_code = Column("cdms_reference_code", String(200))
    registered_address_1 = Column("registered_address_1", String(100))
    registered_address_2 = Column("registered_address_2", String(100))
    registered_address_country = Column("registered_address_country", String(100))
    registered_address_county = Column("registered_address_county", String(50))
    registered_address_postcode = Column("registered_address_postcode", String(100))
    registered_address_area_name = Column("registered_address_area_name", String(50))
    registered_address_town = Column("registered_address_town", String(100))
    sector = Column("sector", String(300))
    export_segment = Column("export_segment", String(200))
    export_sub_segment = Column("export_sub_segment", String(200))
    trading_names = Column("trading_names", ARRAY(String))
    turnover = Column("turnover", String(50))
    turnover_usd = Column("turnover_usd", BigInteger, nullable=True)
    turnover_gbp = Column("turnover_gbp", BigInteger, nullable=True)
    uk_region = Column("uk_region", String(100))
    vat_number = Column("vat_number", String(300))
    website = Column("website", String(300))
    archived_reason = Column("archived_reason", String, nullable=True)
    created_by_id = Column("created_by_id", UUID, nullable=True)
    is_out_of_business = Column("is_out_of_business", Boolean)
    strategy = Column("strategy", String, nullable=True)
    source = Column("source", String(200))


class CompanyInteraction(Base):
    __tablename__ = "company_interactions"
    __table_args__ = {"schema": get_schema()}  # noqa: RUF012

    id = Column("id", Integer, primary_key=True, autoincrement=True)
    team_region = Column("team_region", String(100))
    team_country = Column("team_country", String(100))
    interaction_year = Column("interaction_year", Integer, nullable=True)
    interaction_financial_year = Column("interaction_financial_year", String(50))
    company_name = Column("company_name", String(300))
    company_sector = Column("company_sector", String(300))
    company_id = Column("company_id", UUID, ForeignKey(f"{get_schema()}.companies.id"))
    date_of_interaction = Column("date_of_interaction", Date)
    interaction_subject = Column("interaction_subject", String())
    interaction_theme_investment_or_export = Column("interaction_theme_investment_or_export", String)
    adviser_name = Column("adviser_name", String)
    team_name = Column("team_name", String)


class InvestmentProjects(Base):
    __tablename__ = "investment_projects"
    __table_args__ = {"schema": get_schema()}  # noqa: RUF012

    actual_land_date = Column("actual_land_date", Date, nullable=True)
    actual_uk_regions = Column("actual_uk_regions", String(100))
    address_1 = Column("address_1", String)
    address_2 = Column("address_2", String)
    address_county = Column("address_county", String(100))
    address_country = Column("address_country", String(100))
    address_town = Column("address_town", String(100))
    address_postcode = Column("address_postcode", String(100))
    anonymous_description = Column("anonymous_description", String)
    associated_non_fdi_r_and_d_project_id = Column("associated_non_fdi_r_and_d_project_id", String)
    average_salary = Column("average_salary", String(100))
    business_activities = Column("business_activities", ARRAY(String(100)))
    client_relationship_manager_id = Column("client_relationship_manager_id", UUID, nullable=True)
    client_requirements = Column("client_requirements", String, nullable=True)
    client_contact_ids = Column("client_contact_ids", ARRAY(UUID))
    client_contact_names = Column("client_contact_names", ARRAY(String(100)))
    client_contact_emails = Column("client_contact_emails", ARRAY(String(200)))
    competing_countries = Column("competing_countries", ARRAY(String(100)), nullable=True)
    country_investment_originates_from = Column("country_investment_originates_from", String(100), nullable=True)
    created_by_id = Column("created_by_id", UUID, nullable=True)
    created_on = Column("created_on", DateTime, nullable=True)
    delivery_partners = Column("delivery_partners", ARRAY(String(100)))
    description = Column("description", String)
    estimated_land_date = Column("estimated_land_date", Date, nullable=True)
    export_revenue = Column("export_revenue", Boolean, nullable=True)
    first_active_on = Column("first_active_on", DateTime, nullable=True)
    fdi_type = Column("fdi_type", String(100))
    fdi_value = Column("fdi_value", String(100))
    foreign_equity_investment = Column("foreign_equity_investment", BigInteger, nullable=True)
    government_assistance = Column("government_assistance", Boolean, nullable=True)
    gross_value_added = Column("gross_value_added", Integer, nullable=True)
    gva_multiplier = Column("gva_multiplier", Float, nullable=True)
    id = Column("id", UUID, primary_key=True)
    investment_type = Column("investment_type", String(100), nullable=True)
    investor_company_id = Column("investor_company_id", UUID, nullable=True)
    investor_company_sector = Column("investor_company_sector", String(100), nullable=True)
    investor_type = Column("investor_type", String(100), nullable=True)
    level_of_involvement = Column("level_of_involvement", String(100), nullable=True)
    likelihood_to_land = Column("likelihood_to_land", String(100), nullable=True)
    modified_by_id = Column("modified_by_id", UUID, nullable=True)
    modified_on = Column("modified_on", DateTime, nullable=True)
    name = Column("name", String(300), nullable=True)
    new_tech_to_uk = Column("new_tech_to_uk", Boolean, nullable=True)
    non_fdi_r_and_d_budget = Column("non_fdi_r_and_d_budget", Boolean)
    number_new_jobs = Column("number_new_jobs", Integer, nullable=True)
    number_safeguarded_jobs = Column("number_safeguarded_jobs", Integer, nullable=True)
    other_business_activity = Column("other_business_activity", String, nullable=True)
    project_arrived_in_triage_on = Column("project_arrived_in_triage_on", DateTime, nullable=True)
    project_assurance_adviser_id = Column("project_assurance_adviser_id", UUID, nullable=True)
    project_moved_to_won = Column("project_moved_to_won", DateTime, nullable=True)
    project_manager_id = Column("project_manager_id", UUID, nullable=True)
    project_reference = Column("project_reference", String(100), nullable=True)
    proposal_deadline = Column("proposal_deadline", Date, nullable=True)
    r_and_d_budget = Column("r_and_d_budget", Boolean, nullable=True)
    referral_source_activity = Column("referral_source_activity", String(100), nullable=True)
    referral_source_activity_marketing = Column("referral_source_activity_marketing", String(100), nullable=True)
    referral_source_activity_website = Column("referral_source_activity_website", String(100), nullable=True)
    sector = Column("sector", String, nullable=True)
    specific_programme = Column("specific_programme", ARRAY(String(100)))
    stage = Column("stage", String(100), nullable=True)
    status = Column("status", String(100), nullable=True)
    strategic_drivers = Column("strategic_drivers", ARRAY(String(100)))
    team_member_ids = Column("team_member_ids", ARRAY(UUID))
    total_investment = Column("total_investment", BigInteger, nullable=True)
    uk_company_id = Column("uk_company_id", UUID, ForeignKey(f"{get_schema()}.companies.id"), nullable=True)
    uk_company_sector = Column("uk_company_sector", String(100), nullable=True)
    possible_uk_regions = Column("possible_uk_regions", ARRAY(String(100)))
    eyb_lead_ids = Column("eyb_lead_ids", String(100), nullable=True)


class AccountManagementObjectives(Base):
    __tablename__ = "account_management_objectives"
    __table_args__ = {"schema": get_schema()}  # noqa: RUF012

    id = Column("id", UUID, primary_key=True)
    company_id = Column("company_id", UUID, ForeignKey(f"{get_schema()}.companies.id"))
    subject = Column("subject", String(200))
    detail = Column("detail", String)
    target_date = Column("target_date", Date, nullable=True)
    has_blocker = Column("has_blocker", Boolean, nullable=True)
    blocker_description = Column("blocker_description	", String(200))
    progress = Column("progress", Integer, nullable=True)
    created_on = Column("created_on", DateTime, nullable=True)
    modified_on = Column("modified_on", DateTime, nullable=True)
    created_by_id = Column("created_by_id", UUID, nullable=True)
    modified_by_id = Column("modified_by_id", UUID, nullable=True)

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
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    address_1 = Column("address_1", String)
    address_2 = Column("address_2", String)
    address_county = Column("address_county", String(100))
    address_country = Column("address_country", String(200))
    address_postcode = Column("address_postcode", String(200))
    address_area_name = Column("address_area_name", String(200))
    address_town = Column("address_town", String(200))
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
    modified_on = Column("modified_on", DateTime, nullable=True)
    name = Column("name", String(250))
    number_of_employees = Column("number_of_employees", Integer, nullable=True)
    one_list_tier = Column("one_list_tier", String(100))
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
    website = Column("website", String(300))
    is_out_of_business = Column("is_out_of_business", Boolean)
    strategy = Column("strategy", String, nullable=True)
    source = Column("source", String(200))


class InvestmentProject(Base):
    __tablename__ = "investment_projects"

    actual_land_date = Column("actual_land_date", Date, nullable=True)
    actual_uk_regions = Column("actual_uk_regions", String(100))
    address_1 = Column("address_1", String)
    address_2 = Column("address_2", String)
    # address_county = Column("address_county", String(100))
    # address_country = Column("address_country", String(100))
    address_town = Column("address_town", String(100))
    address_postcode = Column("address_postcode", String(100))
    anonymous_description = Column("anonymous_description", String)
    associated_non_fdi_r_and_d_project_id = Column("associated_non_fdi_r_and_d_project_id", String)
    average_salary = Column("average_salary", String(100))
    business_activities = Column("business_activities", ARRAY(String(100)))
    client_requirements = Column("client_requirements", String, nullable=True)
    competing_countries = Column("competing_countries", ARRAY(String(100)), nullable=True)
    country_investment_originates_from = Column("country_investment_originates_from", String(100), nullable=True)
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
    # investor_company_id = Column("investor_company_id", UUID, nullable=True)
    investor_company_sector = Column("investor_company_sector", String(100), nullable=True)
    investor_type = Column("investor_type", String(100), nullable=True)
    level_of_involvement = Column("level_of_involvement", String(100), nullable=True)
    likelihood_to_land = Column("likelihood_to_land", String(100), nullable=True)
    modified_on = Column("modified_on", DateTime, nullable=True)
    name = Column("name", String(300), nullable=True)
    new_tech_to_uk = Column("new_tech_to_uk", Boolean, nullable=True)
    non_fdi_r_and_d_budget = Column("non_fdi_r_and_d_budget", Boolean)
    number_new_jobs = Column("number_new_jobs", Integer, nullable=True)
    number_safeguarded_jobs = Column("number_safeguarded_jobs", Integer, nullable=True)
    other_business_activity = Column("other_business_activity", String, nullable=True)
    project_arrived_in_triage_on = Column("project_arrived_in_triage_on", DateTime, nullable=True)
    project_moved_to_won = Column("project_moved_to_won", DateTime, nullable=True)
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
    total_investment = Column("total_investment", BigInteger, nullable=True)
    uk_company_sector = Column("uk_company_sector", String(100), nullable=True)
    possible_uk_regions = Column("possible_uk_regions", ARRAY(String(100)))
    eyb_lead_ids = Column("eyb_lead_ids", String(100), nullable=True)

    uk_company_id = Column("uk_company_id", UUID, ForeignKey(Company.id), nullable=True)
    company = relationship("Company", foreign_keys="InvestmentProject.uk_company_id")

    investor_company_id = Column("investor_company_id", UUID, ForeignKey(Company.id), nullable=True)
    investor_company = relationship("Company", foreign_keys="InvestmentProject.investor_company_id")


class Interaction(Base):
    __tablename__ = "interactions"

    communication_channel = Column("communication_channel", String, nullable=True)
    company_export_id = Column("company_export_id", UUID, nullable=True)
    created_on = Column("created_on", DateTime)
    interaction_date = Column("interaction_date", Date)
    id = Column("id", UUID, primary_key=True)
    interaction_link = Column("interaction_link", String, nullable=True)

    interaction_kind = Column("interaction_kind", String, nullable=True)
    modified_on = Column("modified_on", DateTime)
    net_company_receipt = Column("net_company_receipt", Float, nullable=True)
    interaction_notes = Column("interaction_notes", String, nullable=True)
    sector = Column("sector", String)
    service_delivery_status = Column("service_delivery_status", String)
    service_delivery = Column("service_delivery", String)
    interaction_subject = Column("interaction_subject", String)
    theme = Column("theme", String)
    policy_feedback_notes = Column("policy_feedback_notes", String)
    policy_areas = Column("policy_areas", ARRAY(String), nullable=True)
    policy_issue_types = Column("policy_issue_types", ARRAY(String), nullable=True)
    were_countries_discussed = Column("were_countries_discussed", Boolean, nullable=True)
    related_trade_agreement_names = Column("related_trade_agreement_names", ARRAY(String), nullable=True)
    export_barrier_type_names = Column("export_barrier_type_names", ARRAY(String), nullable=True)
    export_barrier_notes = Column("export_barrier_notes", String, nullable=True)
    dbt_initiative = Column("dbt_initiative", String, nullable=True)
    export_challenge_type = Column("export_challenge_type", String, nullable=True)
    interaction_type = Column("interaction_type", String, nullable=True)

    investment_project_id = Column("investment_project_id", UUID, ForeignKey(InvestmentProject.id), nullable=True)
    investment_project = relationship("InvestmentProject", foreign_keys="Interaction.investment_project_id")

    company_id = Column("company_id", UUID, ForeignKey(Company.id))
    company = relationship("Company", foreign_keys="Interaction.company_id")


class AccountManagementObjective(Base):
    __tablename__ = "account_management_objectives"

    id = Column("id", UUID, primary_key=True)
    subject = Column("subject", String(200))
    detail = Column("detail", String)
    target_date = Column("target_date", Date, nullable=True)
    has_blocker = Column("has_blocker", Boolean, nullable=True)
    blocker_description = Column("blocker_description", String(200))
    progress = Column("progress", Integer, nullable=True)
    created_on = Column("created_on", DateTime, nullable=True)
    modified_on = Column("modified_on", DateTime, nullable=True)

    company_id = Column("company_id", UUID, ForeignKey(Company.id))
    company = relationship("Company", foreign_keys="AccountManagementObjective.company_id")

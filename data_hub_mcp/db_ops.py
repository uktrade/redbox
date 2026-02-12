import os

import sa_models as sa_models
from data_classes import (
    AccountManagementObjective,
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyDetailsExtended,
    CompanyInteraction,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    CompanyShort,
    InvestmentProject,
    InvestmentProjectsSearchResult,
)
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session


def object_as_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}


def get_engine():
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT")

    # creds need extracting to local env
    return create_engine(
        f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    )


def get_session(engine=None) -> Session:
    if engine is None:
        engine = get_engine()

    return Session(engine)


def get_companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    session = get_session()

    companies_all = session.query(sa_models.Company).filter(sa_models.Company.name.ilike("%" + company_name + "%"))
    total_count = len(companies_all.all())
    companies = companies_all.offset(page * page_size).limit(page_size).all()
    # Return list

    companies_search_result = CompanySearchResult(companies=[], total=total_count, page=page, page_size=page_size)

    for company in companies:
        companies_search_result.companies.append(CompanyShort.populate_from_record(object_as_dict(company)))

    session.close()

    return companies_search_result


def get_company(company_id) -> CompanyDetails:
    session = get_session()
    company = session.query(sa_models.Company).filter(sa_models.Company.id == company_id).one_or_none()
    company_details = CompanyDetails.populate_from_record(object_as_dict(company)) if company else None
    session.close()
    return company_details


def get_company_extended(
    company_id,
    include_interactions: bool = True,
    include_objectives: bool = True,
    include_investment_projects: bool = True,
) -> CompanyDetailsExtended:
    session = get_session()
    company = session.query(sa_models.Company).filter(sa_models.Company.id == company_id).one_or_none()
    company_details = CompanyDetails.populate_from_record(object_as_dict(company)) if company else None
    company_details_extended = CompanyDetailsExtended(
        company_details=company_details, interactions=None, account_management_objectives=None, investment_projects=None
    )

    if company_details_extended.company_details is not None:
        if include_interactions:
            company_details_extended.interactions = get_company_interactions(company_id, 1000)

        if include_objectives:
            company_details_extended.account_management_objectives = get_account_management_objectives(company_id, 1000)

        if include_investment_projects:
            company_details_extended.investment_projects = get_investment_projects(company_id, 1000)

    session.close()
    return company_details_extended


def get_companies_or_interactions(
    company_name, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult:
    result = CompaniesOrInteractionSearchResult(companies_search_result=None, interactions_search_result=None)
    result.companies_search_result = get_companies(company_name, page_size, page)

    if len(result.companies_search_result.companies) == 1:
        result.interactions_search_result = get_company_interactions(
            result.companies_search_result.companies[0].id, 100, 0
        )

    return result


def get_company_interactions(company_id, page_size: int = 10, page: int = 0) -> CompanyInteractionSearchResult:
    company_interactions_search_result = CompanyInteractionSearchResult(
        interactions=[], total=0, page=page, page_size=page_size
    )
    session = get_session()
    company = session.query(sa_models.Company).filter(sa_models.Company.id == company_id).one_or_none()
    if company:
        company_interactions_all = (
            session.query(sa_models.Interaction)
            .filter(sa_models.Interaction.company_id == company.id)
            .order_by(sa_models.Interaction.interaction_date.desc())
        )
        total_count = len(company_interactions_all.all())
        company_interactions = company_interactions_all.offset(page * page_size).limit(page_size).all()

        company_interactions_search_result.total = total_count

        for company_interaction in company_interactions:
            company_interactions_search_result.interactions.append(
                CompanyInteraction.populate_from_record(object_as_dict(company_interaction))
            )

    session.close()

    return company_interactions_search_result


def get_account_management_objectives(
    company_id, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult:
    account_management_objectives_search_result = AccountManagementObjectivesSearchResult(
        account_management_objectives=[], total=0, page=page, page_size=page_size
    )
    session = get_session()
    company = session.query(sa_models.Company).filter(sa_models.Company.id == company_id).one_or_none()
    if company:
        account_management_objectives_all = (
            session.query(sa_models.AccountManagementObjective)
            .filter(sa_models.AccountManagementObjective.company_id == company.id)
            .order_by(sa_models.AccountManagementObjective.created_on.desc())
        )
        total_count = len(account_management_objectives_all.all())
        account_management_objectives_search_result.total = total_count
        account_management_objectives = (
            account_management_objectives_all.offset(page * page_size).limit(page_size).all()
        )

        for account_management_objective in account_management_objectives:
            account_management_objectives_search_result.account_management_objectives.append(
                AccountManagementObjective.populate_from_record(object_as_dict(account_management_objective))
            )

    session.close()

    return account_management_objectives_search_result


def get_investment_projects(company_id, page_size: int = 10, page: int = 0) -> InvestmentProjectsSearchResult:
    investment_projects_search_result = InvestmentProjectsSearchResult(
        investment_projects=[], total=0, page=page, page_size=page_size
    )
    session = get_session()
    company = session.query(sa_models.Company).filter(sa_models.Company.id == company_id).one_or_none()
    if company:
        investment_projects_all = (
            session.query(sa_models.InvestmentProject)
            .filter(sa_models.InvestmentProject.uk_company_id == company.id)
            .order_by(sa_models.InvestmentProject.created_on.desc())
        )
        total_count = len(investment_projects_all.all())
        investment_projects = investment_projects_all.offset(page * page_size).limit(page_size).all()

        investment_projects_search_result.total = total_count

        for investment_project in investment_projects:
            investment_projects_search_result.investment_projects.append(
                InvestmentProject.populate_from_record(object_as_dict(investment_project))
            )

    session.close()

    return investment_projects_search_result


def db_check():
    session = get_session()
    # We just want to check we can query at this point
    session.query(sa_models.Company).count()
    return True

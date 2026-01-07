from data_classes import CompanyDetails, CompanySearchResult, CompanyShort
from sa_models import Company
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session


def object_as_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}


def get_session() -> Session:
    # creds need extracting to local env
    engine = create_engine("postgresql+psycopg2://user:pass@localhost/data_hub_data")
    return Session(engine)


def get_companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    session = get_session()

    companies_all = session.query(Company).filter(Company.name.ilike("%" + company_name + "%"))
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

    company = session.query(Company).filter(Company.id == company_id).one_or_none()

    company_details = CompanyDetails.populate_from_record(object_as_dict(company)) if company else None

    session.close()

    return company_details

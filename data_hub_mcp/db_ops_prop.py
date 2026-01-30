import logging
import os

from data_classes_prop import (
    AccountManagementObjective,
    AccountManagementObjectivesSearchResult,
    # CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyEnrichmentResult,
    CompanyEnrichmentSearchResult,
    CompanyInteraction,
    CompanyInteractionSearchResult,
    InvestmentProject,
    InvestmentProjectsSearchResult,
    MCPToolResultType,
    SectorCompanyGroup,
    SectorGroupedCompanySearchResult,
)
from sa_models import AccountManagementObjectives, Company, Interaction, InvestmentProjects
from sqlalchemy import create_engine, inspect, or_
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def serialize(obj):
    """
    Recursively convert object to JSON-serializable types.
    Works for dicts, lists, and objects with __dict__.
    """
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return serialize(obj.__dict__)
    else:
        return obj  # int, str, float, bool, None


def object_as_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}


def dataclass_to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    elif isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]
    else:
        return obj


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


def db_check():
    session = get_session()
    # We just want to check we can query at this point
    session.query(Company).count()
    return True


def resolve_companies_by_name(session: Session, company_name: str):
    # 1️⃣ Exact match
    exact = session.query(Company).filter(Company.name == company_name).all()
    if len(exact) == 1:
        return MCPToolResultType.SINGLE_EXACT_MATCH, exact
    if len(exact) > 1:
        return MCPToolResultType.MULTI_EXACT_MATCH, exact

    # 2️⃣ Fetch candidates (case-insensitive) and filter in Python
    candidates = session.query(Company).filter(Company.name.ilike(f"%{company_name}%")).all()

    # Keep only rows where company_name is actually a substring of company.name
    filtered = [c for c in candidates if company_name.lower() in c.name.lower()]

    if len(filtered) == 1:
        return MCPToolResultType.SINGLE_SIM_MATCH, filtered
    if len(filtered) > 1:
        return MCPToolResultType.MULTI_SIM_MATCH, filtered

    # 3️⃣ No matches
    return MCPToolResultType.NONE, []


def fetch_companies(session, company_name: str) -> tuple[MCPToolResultType, list]:
    """Resolve companies by name (exact or similar matches)."""
    return resolve_companies_by_name(session, company_name)


def fetch_related_data(
    session,
    company_ids: list[str],
    fetch_interactions: bool,
    fetch_objectives: bool,
    fetch_investments: bool,
) -> tuple[dict, dict, dict]:
    """Fetch interactions, objectives, and investment projects for given company_ids."""
    interactions_map = {}
    objectives_map = {}
    investments_map = {}

    if not company_ids:
        return interactions_map, objectives_map, investments_map

    # Interactions
    if fetch_interactions:
        interactions_all = session.query(Interaction).filter(Interaction.company_id.in_(company_ids)).all()
        for i in interactions_all:
            interactions_map.setdefault(i.company_id, []).append(
                CompanyInteraction.populate_from_record(object_as_dict(i))
            )

    # Objectives
    if fetch_objectives:
        objectives_all = (
            session.query(AccountManagementObjectives)
            .filter(AccountManagementObjectives.company_id.in_(company_ids))
            .all()
        )
        for o in objectives_all:
            objectives_map.setdefault(o.company_id, []).append(
                AccountManagementObjective.populate_from_record(object_as_dict(o))
            )

    # Investments
    if fetch_investments:
        investments_all = (
            session.query(InvestmentProjects)
            .filter(
                or_(
                    InvestmentProjects.uk_company_id.in_(company_ids),
                    InvestmentProjects.investor_company_id.in_(company_ids),
                )
            )
            .all()
        )
        for ip in investments_all:
            for cid in filter(None, [ip.uk_company_id, ip.investor_company_id]):
                investments_map.setdefault(cid, []).append(InvestmentProject.populate_from_record(object_as_dict(ip)))

    return interactions_map, objectives_map, investments_map


def build_enriched_company(
    company,
    interactions_map: dict,
    objectives_map: dict,
    investments_map: dict,
    fetch_interactions: bool,
    fetch_objectives: bool,
    fetch_investments: bool,
) -> CompanyEnrichmentResult:
    """Return CompanyEnrichmentResult dataclass for a single company."""
    cid = company.id

    interactions = interactions_map.get(cid, []) if fetch_interactions else []
    objectives = objectives_map.get(cid, []) if fetch_objectives else []
    projects = investments_map.get(cid, []) if fetch_investments else []

    return CompanyEnrichmentResult(
        company=CompanyDetails.populate_from_record(object_as_dict(company)),
        interactions=CompanyInteractionSearchResult.load(interactions=interactions),
        objectives=AccountManagementObjectivesSearchResult.load(objectives=objectives),
        investment_projects=InvestmentProjectsSearchResult.load(investment_projects=projects),
    )


def get_company_details(
    company_name: str,
    page_size: int = 10,
    page: int = 0,
    fetch_interactions: bool = True,
    fetch_objectives: bool = True,
    fetch_investments: bool = True,
) -> CompanyEnrichmentSearchResult:
    session = get_session()
    try:
        # 1️⃣ Fetch companies
        result_type, companies = fetch_companies(session, company_name)
        total = len(companies)

        # 2️⃣ Apply company-level pagination
        companies_page = companies[page * page_size : (page + 1) * page_size]
        company_ids = [c.id for c in companies_page]

        # 3️⃣ Fetch related data if requested
        interactions_map, objectives_map, investments_map = fetch_related_data(
            session,
            company_ids,
            fetch_interactions,
            fetch_objectives,
            fetch_investments,
        )

        # 4️⃣ Build enriched results
        companies_enriched = [
            build_enriched_company(
                company,
                interactions_map,
                objectives_map,
                investments_map,
                fetch_interactions,
                fetch_objectives,
                fetch_investments,
            )
            for company in companies_page
        ]

        # 5️⃣ Return top-level MCP result
        return CompanyEnrichmentSearchResult(
            result_type=result_type,
            companies=companies_enriched,
            total=total,
            page=page,
            page_size=page_size,
        )

    finally:
        session.close()


def resolve_sectors_by_name(
    session: Session,
    sector_name: str,
):
    """
    Resolve sectors using the same semantics as resolve_companies_by_name:
    - Exact match first
    - Fallback to ILIKE
    - Return MCPToolResultType + list[str]
    """

    # 1️⃣ Exact match
    exact = session.query(Company.sector).filter(Company.sector == sector_name).distinct().all()

    exact_sectors = [row[0] for row in exact if row[0]]

    if len(exact_sectors) == 1:
        return MCPToolResultType.SINGLE_EXACT_MATCH, exact_sectors

    if len(exact_sectors) > 1:
        return MCPToolResultType.MULTI_EXACT_MATCH, exact_sectors

    # 2️⃣ Similar (ILIKE) match
    similar = session.query(Company.sector).filter(Company.sector.ilike(f"%{sector_name}%")).distinct().all()

    similar_sectors = [row[0] for row in similar if row[0]]

    if len(similar_sectors) == 1:
        return MCPToolResultType.SINGLE_SIM_MATCH, similar_sectors

    if len(similar_sectors) > 1:
        return MCPToolResultType.MULTI_SIM_MATCH, similar_sectors

    # 3️⃣ Nothing found
    return MCPToolResultType.NONE, []


def get_related_company_details(
    sector_name: str | None = None,
    company_name: str | None = None,
    page: int = 0,
    page_size: int = 10,
    fetch_interactions: bool = True,
    fetch_objectives: bool = True,
    fetch_investments: bool = True,
) -> SectorGroupedCompanySearchResult:
    """
    Fetch companies related by sector or by a given company's sector.
    Results are grouped by sector.

    If querying by company_name, only the resolved companies are used (no additional sector query).
    """
    session = get_session()
    try:
        if company_name:
            # 1️⃣ Resolve companies by name
            result_type, companies = resolve_companies_by_name(session, company_name)
            total_companies = len(companies)

            if total_companies == 0:
                return SectorGroupedCompanySearchResult(
                    result_type=MCPToolResultType.NONE,
                    sectors=[],
                    total_companies=0,
                    page=page,
                    page_size=page_size,
                )

            # Apply pagination
            companies_page = companies[page * page_size : (page + 1) * page_size]

        else:
            # 1️⃣ Resolve sectors by name
            result_type, sectors = resolve_sectors_by_name(session, sector_name)
            if not sectors:
                return SectorGroupedCompanySearchResult(
                    result_type=MCPToolResultType.NONE,
                    sectors=[],
                    total_companies=0,
                    page=page,
                    page_size=page_size,
                )

            # 2️⃣ Query companies in all resolved sectors
            base_query = session.query(Company).filter(Company.sector.in_(sectors))
            total_companies = base_query.count()

            # Apply pagination
            companies_page = base_query.offset(page * page_size).limit(page_size).all()

        # 3️⃣ Fetch related data if requested
        company_ids = [c.id for c in companies_page]
        interactions_map, objectives_map, investments_map = fetch_related_data(
            session,
            company_ids,
            fetch_interactions,
            fetch_objectives,
            fetch_investments,
        )

        # 4️⃣ Build enriched companies and group by sector
        sector_map: dict[str, list[CompanyEnrichmentResult]] = {}
        for company in companies_page:
            enriched = build_enriched_company(
                company,
                interactions_map,
                objectives_map,
                investments_map,
                fetch_interactions,
                fetch_objectives,
                fetch_investments,
            )
            sector_map.setdefault(company.sector, []).append(enriched)

        # 5️⃣ Construct sector groups
        sector_groups = [
            SectorCompanyGroup(
                sector=sector,
                companies=companies,
                total=len(companies),
            )
            for sector, companies in sector_map.items()
        ]

        # 6️⃣ Return grouped result
        return SectorGroupedCompanySearchResult(
            result_type=result_type,
            sectors=sector_groups,
            total_sectors=len(sector_groups),
            total_companies=total_companies,
            page=page,
            page_size=page_size,
        )

    finally:
        session.close()

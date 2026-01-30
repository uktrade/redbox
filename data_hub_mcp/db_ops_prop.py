import logging
import os

from data_classes_prop import (
    AccountManagementObjective,
    AccountManagementObjectivesSearchResult,
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
    SectorGroupedInvestmentSummary,
    SectorGroupedOverview,
    SectorInvestmentGroup,
    SectorInvestmentProject,
    SectorOverviewGroup,
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


def resolve_sector(
    session: Session,
    sector_name: str | None = None,
    company_name: str | None = None,
) -> str | None:
    """
    Determine a single sector to query.
    1️⃣ If company_name is provided, use resolve_companies_by_name and pick the first sector.
    2️⃣ If sector_name is provided, use resolve_sectors_by_name and pick the first match.
    Returns None if no sector could be determined.
    """
    # 1️⃣ Resolve by company_name first
    if company_name:
        _, companies = resolve_companies_by_name(session, company_name)
        sectors = [c.sector for c in companies if c.sector]
        if sectors:
            # Return the first sector found
            return sectors[0]

    # 2️⃣ Resolve by sector_name
    if sector_name:
        _, sectors = resolve_sectors_by_name(session, sector_name)
        if sectors:
            return sectors[0]

    # 3️⃣ Nothing found
    return None


def get_sector_overview(
    sector_name: str | None = None,
    company_name: str | None = None,
) -> SectorGroupedOverview:
    """
    Fetch aggregated sector metrics (companies, turnover, employees, investment, GVA) grouped by sector.
    Resolves sector via sector_name or company_name. Returns multiple sectors if company_name maps to multiple.
    """

    def aggregate(field_name: str, item_list: list):
        values = [getattr(item, field_name) for item in item_list if getattr(item, field_name) is not None]
        return (sum(values), len(values)) if values else (None, 0)

    session = get_session()
    try:
        # 1️⃣ Resolve sectors
        if company_name:
            _, companies = resolve_companies_by_name(session, company_name)
            sectors = sorted({c.sector for c in companies if c.sector})
        elif sector_name:
            _, sectors = resolve_sectors_by_name(session, sector_name)
        else:
            sectors = []

        if not sectors:
            return SectorGroupedOverview(sectors=[], total_sectors=0)

        sector_groups: list[SectorOverviewGroup] = []

        for sector in sectors:
            companies = session.query(Company).filter(Company.sector == sector).all()
            total_companies = len(companies)
            total_turnover, turnover_count = aggregate("turnover_gbp", companies)
            total_employees, employees_count = aggregate("number_of_employees", companies)

            # Investment projects
            company_ids = [c.id for c in companies]
            investments = (
                session.query(InvestmentProjects)
                .filter(
                    (InvestmentProjects.uk_company_id.in_(company_ids))
                    | (InvestmentProjects.investor_company_id.in_(company_ids))
                )
                .all()
            )

            total_investment, invest_count = aggregate("total_investment", investments)
            total_gva, gva_count = aggregate("gross_value_added", investments)

            # Compute averages
            avg_turnover = total_turnover / turnover_count if total_turnover is not None else None
            avg_employees = total_employees / employees_count if total_employees is not None else None
            avg_investment = total_investment / invest_count if total_investment is not None else None
            avg_gva = total_gva / gva_count if total_gva is not None else None

            sector_groups.append(
                SectorOverviewGroup(
                    sector=sector,
                    total_companies=total_companies,
                    total_turnover_gbp=total_turnover,
                    total_employees=total_employees,
                    total_investment_gbp=total_investment,
                    total_gva=total_gva,
                    average_turnover_gbp=avg_turnover,
                    average_employees=avg_employees,
                    average_investment_gbp=avg_investment,
                    average_gva=avg_gva,
                )
            )

        return SectorGroupedOverview(sectors=sector_groups, total_sectors=len(sector_groups))

    finally:
        session.close()


def get_sector_investment_projects(
    sector_name: str | None = None,
    company_name: str | None = None,
    page: int = 0,
    page_size: int = 10,
) -> SectorGroupedInvestmentSummary:
    """
    Fetch investment projects grouped by sector. Can resolve sector via sector_name or company_name.
    Supports multiple sectors if company_name resolves to multiple companies.
    Pagination applies to projects within each sector.
    """
    session = get_session()
    try:
        # 1️⃣ Resolve sectors
        if company_name:
            _, companies = resolve_companies_by_name(session, company_name)
            sectors = sorted({c.sector for c in companies if c.sector})
        elif sector_name:
            _, sectors = resolve_sectors_by_name(session, sector_name)
        else:
            sectors = []

        if not sectors:
            return SectorGroupedInvestmentSummary(
                sectors=[],
                total_sectors=0,
                total_projects=0,
                page=page,
                page_size=page_size,
            )

        sector_groups: list[SectorInvestmentGroup] = []
        total_projects_count = 0

        # 2️⃣ Loop over sectors
        for sector in sectors:
            # Companies in this sector
            companies = session.query(Company).filter(Company.sector == sector).all()
            company_ids = [c.id for c in companies]

            # Projects for these companies
            projects_query = session.query(InvestmentProjects).filter(
                (InvestmentProjects.uk_company_id.in_(company_ids))
                | (InvestmentProjects.investor_company_id.in_(company_ids))
            )

            sector_total_projects = projects_query.count()
            total_projects_count += sector_total_projects

            projects_page = projects_query.offset(page * page_size).limit(page_size).all()

            # Aggregate metrics for this sector
            total_investment = 0
            total_gva = 0
            total_new_jobs = 0
            total_safeguarded_jobs = 0
            project_list = []

            for p in projects_page:
                project_list.append(
                    SectorInvestmentProject(
                        project_name=p.name,
                        company_name=p.uk_company_id or p.investor_company_id or "",
                        uk_company_id=p.uk_company_id,
                        investor_company_id=p.investor_company_id,
                        investment_type=p.investment_type or "",
                        stage=p.stage or "",
                        status=p.status or "",
                        likelihood_to_land=p.likelihood_to_land or "",
                        total_investment=p.total_investment,
                        foreign_equity_investment=p.foreign_equity_investment,
                        gross_value_added=p.gross_value_added,
                        number_new_jobs=p.number_new_jobs,
                        number_safeguarded_jobs=p.number_safeguarded_jobs,
                    )
                )
                total_investment += p.total_investment or 0
                total_gva += p.gross_value_added or 0
                total_new_jobs += p.number_new_jobs or 0
                total_safeguarded_jobs += p.number_safeguarded_jobs or 0

            sector_groups.append(
                SectorInvestmentGroup(
                    sector=sector,
                    total_projects=sector_total_projects,
                    total_investment=total_investment or None,
                    total_gva=total_gva or None,
                    total_new_jobs=total_new_jobs or None,
                    total_safeguarded_jobs=total_safeguarded_jobs or None,
                    projects=project_list,
                )
            )

        return SectorGroupedInvestmentSummary(
            sectors=sector_groups,
            total_sectors=len(sector_groups),
            total_projects=total_projects_count,
            page=page,
            page_size=page_size,
        )

    finally:
        session.close()

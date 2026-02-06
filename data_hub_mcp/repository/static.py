from collections.abc import Callable, Iterable

from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)
from static_data import (
    STATIC_COMPANY_DETAILS,
    STATIC_COMPANY_DETAILS_2,
    STATIC_COMPANY_SHORT,
    STATIC_COMPANY_SHORT_2,
    STATIC_INTERACTION,
    STATIC_INTERACTION_2,
    STATIC_INVESTMENT_PROJECT,
    STATIC_INVESTMENT_PROJECT_2,
    STATIC_OBJECTIVE,
    STATIC_OBJECTIVE_2,
)

from repository.base import DataHubRepository

# Static Data
STATIC_COMPANIES = [STATIC_COMPANY_SHORT, STATIC_COMPANY_SHORT_2]
STATIC_COMPANY_DETAILS_LIST = [STATIC_COMPANY_DETAILS, STATIC_COMPANY_DETAILS_2]
STATIC_INTERACTIONS = [STATIC_INTERACTION, STATIC_INTERACTION_2]
STATIC_OBJECTIVES = [STATIC_OBJECTIVE, STATIC_OBJECTIVE_2]
STATIC_INVESTMENT_PROJECTS = [STATIC_INVESTMENT_PROJECT, STATIC_INVESTMENT_PROJECT_2]


def filter_exact(items: Iterable, key_fn: Callable[[object], str], query: str, case_sensitive: bool = True) -> list:
    """Return items that match exactly first, otherwise items that contain query."""
    query_res = query.lower() if not case_sensitive else query
    return [i for i in items if (key_fn(i).lower() if not case_sensitive else key_fn(i)) == query_res]


def filter_exact_or_contains(items: Iterable, key_fn: Callable[[object], str], query: str) -> list:
    """Return items that match exactly first, otherwise items that contain query."""
    exact_matches = filter_exact(items, key_fn, query, False)
    if exact_matches:
        return exact_matches
    return [i for i in items if query.lower() in key_fn(i).lower()]


class StaticRepository(DataHubRepository):
    """Static repository using structured lists and generic filtering."""

    async def companies(self, company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
        matches = filter_exact_or_contains(STATIC_COMPANIES, key_fn=lambda c: c.name, query=company_name)
        return CompanySearchResult(companies=matches, total=len(matches), page=page, page_size=page_size)

    async def company_details(self, company_id: str) -> CompanyDetails | None:
        matches = filter_exact(STATIC_COMPANY_DETAILS_LIST, key_fn=lambda c: c.id, query=company_id)
        return matches[0] if matches else None

    async def companies_or_interactions(
        self, company_name: str, page_size: int = 10, page: int = 0
    ) -> CompaniesOrInteractionSearchResult:
        companies = filter_exact_or_contains(STATIC_COMPANIES, key_fn=lambda c: c.name, query=company_name)
        interactions = []
        # Match interactions by position if companies are found
        for c in companies:
            idx = STATIC_COMPANIES.index(c)
            interactions.append(STATIC_INTERACTIONS[idx])
        return CompaniesOrInteractionSearchResult(
            companies_search_result=CompanySearchResult(
                companies=companies, total=len(companies), page=page, page_size=page_size
            ),
            interactions_search_result=CompanyInteractionSearchResult(
                interactions=interactions,
                total=len(interactions),
                page=page,
                page_size=page_size,
            ),
        )

    async def company_interactions(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> CompanyInteractionSearchResult:
        companies = filter_exact(STATIC_COMPANIES, key_fn=lambda c: c.id, query=company_id)
        interactions = [STATIC_INTERACTIONS[STATIC_COMPANIES.index(c)] for c in companies]
        return CompanyInteractionSearchResult(
            interactions=interactions, total=len(interactions), page=page, page_size=page_size
        )

    async def account_management_objectives(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> AccountManagementObjectivesSearchResult:
        companies = filter_exact(STATIC_COMPANIES, key_fn=lambda c: c.id, query=company_id)
        objectives = [STATIC_OBJECTIVES[STATIC_COMPANIES.index(c)] for c in companies]
        return AccountManagementObjectivesSearchResult(
            account_management_objectives=objectives, total=len(objectives), page=page, page_size=page_size
        )

    async def investment_projects(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> InvestmentProjectsSearchResult:
        companies = filter_exact(STATIC_COMPANIES, key_fn=lambda c: c.id, query=company_id)
        projects = [STATIC_INVESTMENT_PROJECTS[STATIC_COMPANIES.index(c)] for c in companies]
        return InvestmentProjectsSearchResult(
            investment_projects=projects, total=len(projects), page=page, page_size=page_size
        )

import db_ops
from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)

from repository.base import DataHubRepository


class PostgresRepository(DataHubRepository):
    """Repository that uses actual DB queries via db_ops"""

    async def companies(self, company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
        return db_ops.get_companies(company_name, page_size, page)

    async def company_details(self, company_id: str) -> CompanyDetails | None:
        return db_ops.get_company(company_id)

    async def companies_or_interactions(
        self, company_name: str, page_size: int = 10, page: int = 0
    ) -> CompaniesOrInteractionSearchResult:
        return db_ops.get_companies_or_interactions(company_name, page_size, page)

    async def company_interactions(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> CompanyInteractionSearchResult:
        return db_ops.get_company_interactions(company_id, page_size, page)

    async def account_management_objectives(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> AccountManagementObjectivesSearchResult:
        return db_ops.get_account_management_objectives(company_id, page_size, page)

    async def investment_projects(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> InvestmentProjectsSearchResult:
        return db_ops.get_investment_projects(company_id, page_size, page)

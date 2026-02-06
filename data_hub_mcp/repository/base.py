import asyncio
import logging
from abc import ABC, abstractmethod
from functools import wraps

from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)

logging.basicConfig(level=logging.INFO)


class DataHubRepository(ABC):
    """Abstract repository for all DataHub-related operations."""

    def __init__(self):
        self.logger = logging.getLogger(f"repository.{self.__class__.__name__}")

    def __init_subclass__(cls):
        super().__init_subclass__()
        for name, attr in cls.__dict__.items():
            if callable(attr) and not name.startswith("_") and asyncio.iscoroutinefunction(attr):
                setattr(cls, name, cls._wrap_logging(attr, name))

    @staticmethod
    def _wrap_logging(method, method_name):
        @wraps(method)
        async def wrapper(self, *args, **kwargs):
            self.logger.info("Calling %s with args=%s, kwargs=%s", method_name, args, kwargs)
            result = await method(self, *args, **kwargs)
            self.logger.info("%s returned %s", method_name, result)
            return result

        return wrapper

    @abstractmethod
    async def companies(self, company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult: ...

    @abstractmethod
    async def company_details(self, company_id: str) -> CompanyDetails | None: ...

    @abstractmethod
    async def companies_or_interactions(
        self, company_name: str, page_size: int = 10, page: int = 0
    ) -> CompaniesOrInteractionSearchResult: ...

    @abstractmethod
    async def company_interactions(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> CompanyInteractionSearchResult: ...

    @abstractmethod
    async def account_management_objectives(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> AccountManagementObjectivesSearchResult: ...

    @abstractmethod
    async def investment_projects(
        self, company_id: str, page_size: int = 10, page: int = 0
    ) -> InvestmentProjectsSearchResult: ...

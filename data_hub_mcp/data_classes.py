from dataclasses import dataclass


@dataclass
class CompanyShort:
    """
    Class for representing company, the short version
    """

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
    address_postcode: str | None
    address_country: str | None
    company_number: str | None
    description: str | None
    name: str
    turnover_gbp: int | None
    company_number: str | None

    @staticmethod
    def populate_from_record(record):
        return CompanyDetails(
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
class CompanySearchResult:
    """Company search result"""

    companies: list[CompanyShort]
    total: int
    page: int
    page_size: int

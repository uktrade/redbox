import asyncio
from dataclasses import asdict

import log


async def call_tool(client, tool_name: str, name: str):
    async with client:
        result = await client.call_tool(tool_name, {"name": name})
        log.logger.info(result)
        return asdict(result)


async def call_companies(client, company_name: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "companies", {"company_name": company_name, "page": page, "page_size": page_size}
        )
        log.logger.info(result)

        return asdict(result)


async def call_company_details(client, company_id: str):
    async with client:
        result = await client.call_tool("company_details", {"company_id": company_id})
        log.logger.info(result)

        return asdict(result)


async def call_company_interactions(client, company_id: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "company_interactions", {"company_id": company_id, "page": page, "page_size": page_size}
        )
        log.logger.info(result)

        return asdict(result)


async def call_account_management_objectives(client, company_id: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "account_management_objectives", {"company_id": company_id, "page": page, "page_size": page_size}
        )
        log.logger.info(result)

        return asdict(result)


async def call_investment_projects(client, company_id: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "investment_projects", {"company_id": company_id, "page": page, "page_size": page_size}
        )
        log.logger.info(result)

        return asdict(result)


async def call_companies_or_interactions(client, company_name: str, page: int = 0, page_size: int = 10):
    async with client:
        result = await client.call_tool(
            "companies_or_interactions", {"company_name": company_name, "page": page, "page_size": page_size}
        )
        log.logger.info(result)

        return asdict(result)


async def call_company_details_extended(
    client, company_id, include_interactions=True, include_objectives=True, include_investment_projects=True
):
    async with client:
        result = await client.call_tool(
            "company_details_extended",
            {
                "company_id": company_id,
                "include_interactions": include_interactions,
                "include_objectives": include_objectives,
                "include_investment_projects": include_investment_projects,
            },
        )
        log.logger.info(result)

        return asdict(result)


def run_examples(client):
    asyncio.run(call_tool(client, "greet", "Ford"))
    asyncio.run(call_tool(client, "greet", "Ford"))
    asyncio.run(call_tool(client, "greet", "Ford"))

    # asyncio.run(call_tool('greet', "Doug"))
    asyncio.run(call_companies(client, "SOME"))
    asyncio.run(call_companies(client, "SOME OTHER", 0, 2))

    asyncio.run(call_company_details(client, "00000000-0000-0000-0000-000000000000"))
    asyncio.run(call_company_details(client, "00000000-0000-0000-0000-000000000001"))

    # invalid id
    asyncio.run(call_company_details(client, "00000000-0000-0000-0000-000000aaaaaa"))

    # Company interactions
    asyncio.run(call_company_interactions(client, "00000000-0000-0000-0000-000000aaaaaa"))
    asyncio.run(call_company_interactions(client, "00000000-0000-0000-0000-000000000001"))
    asyncio.run(call_company_interactions(client, "00000000-0000-0000-0000-000000000000"))

    # Account management objectives
    asyncio.run(call_account_management_objectives(client, "00000000-0000-0000-0000-000000aaaaaa"))
    asyncio.run(call_account_management_objectives(client, "00000000-0000-0000-0000-000000000001"))
    asyncio.run(call_account_management_objectives(client, "00000000-0000-0000-0000-000000000000"))

    # Investment projects
    asyncio.run(call_investment_projects(client, "00000000-0000-0000-0000-000000aaaaaa"))
    asyncio.run(call_investment_projects(client, "00000000-0000-0000-0000-000000000001"))
    asyncio.run(call_investment_projects(client, "00000000-0000-0000-0000-000000000000"))

    # # companies or interactions
    asyncio.run(call_companies_or_interactions(client, "CRESCENT MARIGOLD"))
    asyncio.run(call_companies_or_interactions(client, "NIMBUS"))
    asyncio.run(call_companies_or_interactions(client, "BANANAJOE"))

    # Company details extended
    asyncio.run(call_company_details_extended(client, "df27f8a4-6341-4571-82a4-94732d23eca5"))
    asyncio.run(call_company_details_extended(client, "7f535d50-fa98-475b-be46-1f763b4398e5"))
    asyncio.run(call_company_details_extended(client, "00000000-0000-0000-0000-000000000000"))

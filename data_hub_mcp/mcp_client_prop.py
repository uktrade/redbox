import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

import log
from fastmcp.exceptions import ToolError


async def call_tool(client, tool_name: str, name: str):
    async with client:
        result = await client.call_tool(tool_name, {"name": name})
        log.logger.info(result)
        return asdict(result)


def dataclass_to_json_safe(obj):
    """Recursively convert dataclass / objects to JSON-safe dicts/lists."""
    if is_dataclass(obj):
        return {k: dataclass_to_json_safe(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_json_safe(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Add custom handling for TextContent
    elif type(obj).__name__ == "TextContent":
        return str(obj)
    else:
        return obj


async def run_examples(client):
    """
    Run example queries against multiple tools and collect results.
    """

    results = []

    # Example queries
    examples = [
        {
            "tool": "company_details",
            "params": {
                "company_name": "VERTEX",
                "page": 0,
                "page_size": 2,
                "fetch_interactions": True,
                "fetch_objectives": True,
                "fetch_investments": True,
            },
        },
        {
            "tool": "company_details",
            "params": {
                "company_name": "VERTEX NIMBUS INDUSTRIES",
                "page": 0,
                "page_size": 2,
                "fetch_interactions": False,
                "fetch_objectives": True,
                "fetch_investments": False,
            },
        },
        {
            "tool": "related_company_details",
            "params": {
                "company_name": "VERTEX",
                "page": 0,
                "page_size": 3,
                "fetch_interactions": True,
                "fetch_objectives": True,
                "fetch_investments": True,
            },
        },
        {
            "tool": "related_company_details",
            "params": {
                "sector_name": "Manufacturing",
                "page": 0,
                "page_size": 3,
                "fetch_interactions": False,
                "fetch_objectives": False,
                "fetch_investments": True,
            },
        },
        # 3️⃣ Sector overview: aggregated investment, jobs, GVA
        {
            "tool": "sector_overview",
            "params": {
                "sector_name": "Manufacturing",
            },
        },
        # 4️⃣ Sector investment projects: FDI, status, economic impact
        {
            "tool": "sector_investment_projects",
            "params": {
                "sector_name": "Technology",
                "page": 0,
                "page_size": 5,
            },
        },
        # 5️⃣ Sector overview by company name
        {
            "tool": "sector_overview",
            "params": {
                "company_name": "VERTEX NIMBUS INDUSTRIES",
            },
        },
        {
            "tool": "sector_overview",
            "params": {},
        },
        # 6️⃣ Sector investment projects by company name
        {
            "tool": "sector_investment_projects",
            "params": {
                "company_name": "VERTEX",
            },
        },
    ]

    async with client:
        for example in examples:
            tool_name = example["tool"]
            params = example["params"]

            # Call tool
            try:
                response = await client.call_tool(tool_name, params)
                # Convert dataclass to dict safely
                response_dict = dataclass_to_json_safe(response)
                result = response_dict.get("structured_content")
            except ToolError as e:
                result = str(e)

            results.append(
                {
                    "tool": tool_name,
                    "query": params,
                    "response": result,  # no need for serialize or structured_content
                }
            )

    # Return nicely formatted JSON string
    return json.dumps(results, indent=4)

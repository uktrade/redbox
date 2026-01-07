from data_classes import CompanyDetails, CompanySearchResult
from db_ops import get_companies, get_company
from fastmcp import FastMCP
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from starlette.responses import JSONResponse

# Allow 10 requests per second with bursts up to 20
rate_limiter = RateLimitingMiddleware(max_requests_per_second=10, burst_capacity=20)

mcp = FastMCP(
    name="Data Hub companies MCP server",
    stateless_http=True,
    json_response=True,
)

mcp.add_middleware(rate_limiter)


@mcp.custom_route("/health", methods=["GET"])
async def health_check():
    return JSONResponse({"status": "healthy", "service": "mcp-server"})


@mcp.tool(
    name="greet",  # Custom tool name for the LLM
    description="Basic example tool useful for testing.",  # Custom description
    tags={"testing"},  # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "Doug Mills"},  # Custom metadata
)
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool(
    name="companies",
    description="Query companies based on company name, returns a short overview of a list of companies",
    tags={"data_hub", "companies", "search"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies(company_name: str, page_size: int = 10, page: int = 0) -> CompanySearchResult:
    return get_companies(company_name, page_size, page)


@mcp.tool(
    name="company_details",
    description="Full details of a company",
    tags={"data_hub", "companies"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_details(company_id: str) -> CompanyDetails | None:
    return get_company(company_id)

import logging
import os

import httpx
from data_classes import (
    AccountManagementObjectivesSearchResult,
    CompaniesOrInteractionSearchResult,
    CompanyDetails,
    CompanyInteractionSearchResult,
    CompanySearchResult,
    InvestmentProjectsSearchResult,
)
from db_ops import (
    db_check,
    get_account_management_objectives,
    get_companies,
    get_companies_or_interactions,
    get_company,
    get_company_interactions,
    get_investment_projects,
)
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from starlette.responses import JSONResponse

# load_dotenv()

# auth_provider = Auth0Provider(
#     config_url=os.getenv("AUTHBROKER_CONFIG_URL"),  # Your Auth0 configuration URL
#     client_id=os.getenv("AUTHBROKER_CLIENT_ID"),  # Your Auth0 application Client ID
#     client_secret=os.getenv("AUTHBROKER_CLIENT_SECRET"),  # Your Auth0 application Client Secret
#     audience=os.getenv("AUTHBROKER_AUDIENCE"),  # Your Auth0 API audience
#     base_url=os.getenv("AUTHBROKER_BASE_URL"),  # Must match your application configuration
# )


logger = logging.getLogger(__name__)


SSO_INTROSPECTION_URL = os.getenv("SSO_INTROSPECTION_URL")
INTROSPECTION_TOKEN = os.getenv("INTROSPECTION_TOKEN")
AUTH_ENABLED = os.getenv("MCP_AUTH_ENABLED")
http_client = httpx.AsyncClient(timeout=5.0)


class SSOIntrospectionVerifier(TokenVerifier):
    def __init__(self, introspection_url: str, introspection_token: str):
        super().__init__()
        self._introspection_url = introspection_url
        self._introspection_token = introspection_token
        self._client = httpx.AsyncClient(timeout=5.0)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not self._introspection_url or not self._introspection_token:
            return None
        try:
            response = await self._client.post(
                self._introspection_url,
                data={"token": token},
                headers={"Authorization": f"Bearer {self._introspection_token}"},
            )
        except httpx.HTTPError as exc:
            logger.warning("SSO communication failure: %s", exc)
            return None

        if response.status_code not in (200, 401):
            logger.warning("SSO introspection failed: %s", response.status_code)
            return None

        token_data = response.json()
        if not token_data or not token_data.get("active"):
            return None

        scopes = str(token_data.get("scope", "")).split()
        client_id = token_data.get("client_id") or "sso"
        expires_at = token_data.get("exp")
        resource = token_data.get("aud") if isinstance(token_data.get("aud"), str) else None

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=resource,
            claims=token_data,
        )


auth_provider = SSOIntrospectionVerifier(SSO_INTROSPECTION_URL, INTROSPECTION_TOKEN) if AUTH_ENABLED else None


mcp = FastMCP(name="Data Hub MCP Server", stateless_http=True, json_response=True, auth=auth_provider)


@mcp.custom_route("/health", methods=["GET"])
async def health_check():
    db_status = "failed"
    if db_check():
        db_status = "success"

    return JSONResponse({"status": "healthy", "service": "mcp-server", "db_access_status": db_status})


@mcp.custom_route("/config", methods=["GET"])
async def config():
    return JSONResponse({"env": str(os.environ)})


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


@mcp.tool(
    name="companies_or_interactions",
    description="""
Query companies, and will return interactions on a single result,
or a list of companies of there are multiple matches
""",
    tags={"data_hub", "companies_or_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def companies_or_interactions(
    company_name: str, page_size: int = 10, page: int = 0
) -> CompaniesOrInteractionSearchResult | None:
    return get_companies_or_interactions(company_name, page_size, page)


@mcp.tool(
    name="company_interactions",
    description="Query company interactions based on company id",
    tags={"data_hub", "company_interactions"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def company_interactions(
    company_id: str, page_size: int = 10, page: int = 0
) -> CompanyInteractionSearchResult | None:
    return get_company_interactions(company_id, page_size, page)


@mcp.tool(
    name="account_management_objectives",
    description="Query account management objectives based on company id",
    tags={"data_hub", "company", "account_management_objectives"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def account_management_objectives(
    company_id: str, page_size: int = 10, page: int = 0
) -> AccountManagementObjectivesSearchResult | None:
    return get_account_management_objectives(company_id, page_size, page)


@mcp.tool(
    name="investment_projects",
    description="Query investment projects based on company id",
    tags={"data_hub", "company", "investment_projects"},
    meta={"version": "1.0", "author": "Doug Mills"},
)
async def investment_projects(
    company_id: str, page_size: int = 10, page: int = 0
) -> InvestmentProjectsSearchResult | None:
    return get_investment_projects(company_id, page_size, page)

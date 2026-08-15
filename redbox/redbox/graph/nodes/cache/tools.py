import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from redbox.api.wrapper import SensitiveValue
from redbox.graph.nodes.tools import get_datahub_mcp_tools

log = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    value: Any
    expires_at: float = Field(default_factory=lambda: time.monotonic() + 300)

    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    model_config = {"arbitrary_types_allowed": True}


_datahub_mcp_tool_cache: dict[SensitiveValue, CacheEntry] = {}
_datahub_mcp_tool_cache_lock = asyncio.Lock()


async def get_cached_datahub_mcp_tools(sso_token_getter: Callable[[], Awaitable[str]]) -> list:
    raw_token: str = await sso_token_getter()
    token_key = SensitiveValue(value=raw_token)

    if (entry := _datahub_mcp_tool_cache.get(token_key)) and not entry.is_expired():
        log.warning("[get_cached_datahub_mcp_tools] Fetching Datahub_Agent MCP tools from cache...")
        return entry.value

    async with _datahub_mcp_tool_cache_lock:
        if not (entry := _datahub_mcp_tool_cache.get(token_key)) or entry.is_expired():
            log.warning(
                "[get_cached_datahub_mcp_tools] Fetching Datahub_Agent MCP tools from server and updating cache..."
            )
            _datahub_mcp_tool_cache[token_key] = CacheEntry(
                value=await get_datahub_mcp_tools(sso_token_getter=sso_token_getter)
            )
    return _datahub_mcp_tool_cache[token_key].value

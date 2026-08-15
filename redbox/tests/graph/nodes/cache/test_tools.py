import asyncio
import time
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from redbox.graph.nodes.cache.tools import CacheEntry, SensitiveValue, get_cached_datahub_mcp_tools


class TestCacheEntry:
    @pytest.mark.parametrize(
        "offset,expected",
        [
            (-1, True),  # expired 1s ago
            (0, True),  # exactly at expiry
            (1, False),  # 1s in the future
            (300, False),  # full TTL remaining
        ],
    )
    def test_is_expired(self, offset: float, expected: bool):
        entry = CacheEntry(value="x", expires_at=time.monotonic() + offset)
        assert entry.is_expired() is expected

    @pytest.mark.parametrize(
        "value",
        [
            [],
            [{"name": "tool_a"}],
            [{"name": "tool_a"}, {"name": "tool_b"}],
        ],
    )
    def test_stores_any_value(self, value: Any):
        entry = CacheEntry(value=value)
        assert entry.value == value

    def test_default_expires_at_is_five_minutes(self):
        before = time.monotonic()
        entry = CacheEntry(value=[])
        after = time.monotonic()
        assert before + 299 < entry.expires_at <= after + 301


class TestSensitiveValue:
    @pytest.mark.parametrize("token", ["token_a", "token_b", "token_a"])
    def test_repr_is_redacted(self, token: str):
        assert token not in repr(SensitiveValue(token))

    @pytest.mark.parametrize(
        "token_a,token_b,equal",
        [
            ("abc", "abc", True),
            ("abc", "xyz", False),
        ],
    )
    def test_equality(self, token_a: str, token_b: str, equal: bool):
        assert (SensitiveValue(token_a) == SensitiveValue(token_b)) is equal

    @pytest.mark.parametrize(
        "token_a,token_b,same_bucket",
        [
            ("abc", "abc", True),
            ("abc", "xyz", False),
        ],
    )
    def test_hash(self, token_a: str, token_b: str, same_bucket: bool):
        assert (hash(SensitiveValue(token_a)) == hash(SensitiveValue(token_b))) is same_bucket

    def test_usable_as_dict_key(self):
        key1 = SensitiveValue("token1")
        key2 = SensitiveValue("token2")
        d = {key1: "value1", key2: "value2"}
        assert d[SensitiveValue("token1")] == "value1"
        assert d[SensitiveValue("token2")] == "value2"


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset module-level cache between tests."""
    from redbox.graph.nodes.cache.tools import _datahub_mcp_tool_cache

    _datahub_mcp_tool_cache.clear()
    yield
    _datahub_mcp_tool_cache.clear()


def make_token_getter(token: str) -> Callable[[], Awaitable[str]]:
    async def _get_token() -> str:
        return token

    return _get_token


@pytest.mark.asyncio
class TestGetCachedTools:
    @pytest.mark.parametrize(
        "tools",
        [
            [],
            [{"name": "tool_a"}],
            [{"name": "tool_a"}, {"name": "tool_b"}],
        ],
    )
    async def test_returns_tools_on_cold_cache(self, tools: list):
        getter = make_token_getter("token_x")
        with patch("redbox.graph.nodes.cache.tools.get_datahub_mcp_tools", AsyncMock(return_value=tools)):
            result = await get_cached_datahub_mcp_tools(getter)
        assert result == tools

    async def test_cache_hit_skips_fetch(self):
        getter = make_token_getter("token_x")
        mock = AsyncMock(return_value=[{"name": "tool_a"}])
        with patch("redbox.graph.nodes.cache.tools.get_datahub_mcp_tools", mock):
            await get_cached_datahub_mcp_tools(getter)
            await get_cached_datahub_mcp_tools(getter)
        mock.assert_awaited_once()

    async def test_cache_miss_on_expiry(self):
        getter = make_token_getter("token_x")
        mock = AsyncMock(return_value=[{"name": "tool_a"}])
        with patch("redbox.graph.nodes.cache.tools.get_datahub_mcp_tools", mock):
            await get_cached_datahub_mcp_tools(getter)
            # Force expiry
            from redbox.graph.nodes.cache.tools import _datahub_mcp_tool_cache

            for entry in _datahub_mcp_tool_cache.values():
                entry.expires_at = time.monotonic() - 1
            await get_cached_datahub_mcp_tools(getter)
        assert mock.await_count == 2

    @pytest.mark.parametrize(
        "token_a,token_b,expected_calls",
        [
            ("token_x", "token_x", 1),  # same token -> shared cache
            ("token_x", "token_y", 2),  # different tokens -> separate fetches
        ],
    )
    async def test_cache_isolation_by_token(self, token_a: str, token_b: str, expected_calls: int):
        mock = AsyncMock(return_value=[])
        with patch("redbox.graph.nodes.cache.tools.get_datahub_mcp_tools", mock):
            await get_cached_datahub_mcp_tools(make_token_getter(token_a))
            await get_cached_datahub_mcp_tools(make_token_getter(token_b))
        assert mock.await_count == expected_calls

    async def test_concurrent_requests_fetch_once(self):
        getter = make_token_getter("token_x")
        mock = AsyncMock(return_value=[{"name": "tool_a"}])
        with patch("redbox.graph.nodes.cache.tools.get_datahub_mcp_tools", mock):
            await asyncio.gather(*[get_cached_datahub_mcp_tools(getter) for _ in range(10)])
        mock.assert_awaited_once()

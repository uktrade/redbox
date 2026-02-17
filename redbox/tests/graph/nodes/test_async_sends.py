import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch
from httpx import ConnectError

from redbox.graph.nodes.sends import wrap_async_tool


@dataclass
class FakeTool:
    name: str
    args_schema: dict
    ainvoke: AsyncMock


@dataclass
class FakeToolDefinition:
    metadata: dict


@pytest.fixture
def fake_tool_definition() -> FakeToolDefinition:
    return FakeToolDefinition(metadata={"url": "http://fake-mcp-server"})


@pytest.fixture
def fake_selected_tool() -> FakeTool:
    return FakeTool(name="my_tool", args_schema={"required": []}, ainvoke=AsyncMock(return_value={"status": "ok"}))


class TestWrapAsyncTool:
    def test_connection_failure(self, fake_tool_definition: FakeToolDefinition, fake_selected_tool: FakeTool):
        wrapped = wrap_async_tool(fake_tool_definition, fake_selected_tool.name)
        args = {"foo": "bar"}

        with pytest.raises(ExceptionGroup) as exc_info:
            wrapped(args)

        exceptions = exc_info.value.exceptions
        assert all(isinstance(e, ConnectError) for e in exceptions)

    @pytest.mark.parametrize(
        "args,expected_args",
        [
            ({"foo": "bar", "is_intermediate_step": True}, {"foo": "bar"}),  # intermediate step present
            ({"foo": "bar"}, {"foo": "bar"}),  # no intermediate step
        ],
    )
    def test_success(
        self, fake_tool_definition: "FakeToolDefinition", fake_selected_tool: "FakeTool", args, expected_args
    ):
        wrapped = wrap_async_tool(fake_tool_definition, fake_selected_tool.name)

        with (
            patch("redbox.graph.nodes.sends.streamablehttp_client") as mock_stream_client,
            patch("redbox.graph.nodes.sends.ClientSession") as mock_client_session,
            patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock) as mock_load_tools,
        ):
            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__.return_value = ("read", "write", None)
            mock_stream_cm.__aexit__.return_value = None
            mock_stream_client.return_value = mock_stream_cm

            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            mock_session_cm.__aexit__.return_value = None
            mock_client_session.return_value = mock_session_cm

            mock_load_tools.return_value = [fake_selected_tool]

            result = wrapped(args)

            assert args == expected_args
            fake_selected_tool.ainvoke.assert_awaited_once_with(expected_args)
            assert result == {"status": "ok"}
            mock_session_instance.initialize.assert_awaited_once()

    def test_tool_not_found(self, fake_tool_definition):
        missing_tool_name = "missing_tool"
        wrapped = wrap_async_tool(fake_tool_definition, missing_tool_name)

        with (
            patch("redbox.graph.nodes.sends.streamablehttp_client") as mock_stream_client,
            patch("redbox.graph.nodes.sends.ClientSession") as mock_client_session,
            patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock) as mock_load_tools,
        ):
            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__.return_value = ("read", "write", None)
            mock_stream_cm.__aexit__.return_value = None
            mock_stream_client.return_value = mock_stream_cm

            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session_instance
            mock_session_cm.__aexit__.return_value = None
            mock_client_session.return_value = mock_session_cm

            mock_load_tools.return_value = []

            with pytest.raises(ValueError, match=f"tool with name '{missing_tool_name}' not found"):
                wrapped({"foo": "bar"})

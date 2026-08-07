from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ConnectError
from tests.conftest import MCPTool
from tests.retriever.data import MCP_TOOL_RESULTS

import redbox.graph.nodes.runner.models as tr_models
from redbox.api.format import MCPResponseMetadata
from redbox.api.wrapper import SensitiveValue
from redbox.graph.nodes.runner.wrap_async import _get_mcp_headers, execute_mcp_tools
from redbox.models.file import ChunkCreatorType


class TestExecuteMCPTools:
    def _assert_streamable_http_client_call(self, mock_http_client, expected_url: str, expected_token: str):
        mock_http_client.assert_called_once()

        call_args, call_kwargs = mock_http_client.call_args
        assert call_args == (expected_url,)
        assert "http_client" in call_kwargs
        assert call_kwargs["http_client"].headers["Authorization"] == f"Bearer {expected_token}"

    def _patch_mcp_env(self, mock_load_tools, mock_http_client, mock_session_class, tools):
        """Patch MCP networking to allow execute_mcp_tools to succeed."""
        # streamable_http_client mock
        mock_read, mock_write = AsyncMock(), AsyncMock()
        mock_http_cm = AsyncMock()
        mock_http_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_http_cm.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.return_value = mock_http_cm

        # ClientSession mock
        mock_session = AsyncMock()

        # initialize() must return something with real strings at serverInfo.name/version
        mock_server_info = MagicMock()
        mock_server_info.name = "test-server"
        mock_server_info.version = "1.0"
        mock_init_result = MagicMock()
        mock_init_result.serverInfo = mock_server_info
        mock_session.initialize = AsyncMock(return_value=mock_init_result)

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # load_mcp_tools returns the tool
        mock_load_tools.return_value = tools

        return mock_session

    @pytest.mark.parametrize(
        "url",
        [
            "http://fake-mcp-url",  # non-existent hostname
            "http://127.0.0.1:59999",  # unused localhost port
        ],
    )
    def test_connection_failure(
        self,
        url: Literal["http://fake-mcp-url"] | Literal["http://127.0.0.1:59999"],
    ):
        """Test execute_mcp_tools fails when MCP server cannot be reached."""
        wrapped = execute_mcp_tools(
            mcp_input=tr_models.ToolCallRequest.MCPAsync(
                mcp_server=url,
                access_token=SensitiveValue(value="fake"),
                creator_type=ChunkCreatorType.datahub,
                tool_calls=[{"id": "dummy_tool_1", "name": "dummy_tool", "args": {}}],
            )
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            wrapped()

        # All inner exceptions should match the expected types
        exceptions = exc_info.value.exceptions
        assert all(isinstance(e, ConnectError) or type(e).__name__ == "ConnectError" for e in exceptions)

    @pytest.mark.parametrize("expected_tool_result, expected_documents", MCP_TOOL_RESULTS)
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamable_http_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
    def test_returns_expected_results(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_mcp_tool: type[MCPTool.Passing],
        expected_tool_result: tuple[tuple[str, MCPResponseMetadata], str],
        expected_documents: tuple[tuple[str, MCPResponseMetadata], str],
    ):
        """Test that execute_mcp_tools correctly returns results from async tool invocation"""
        expected_tool_content, expected_tool_metadata = expected_tool_result

        # Mock tool with metadata
        tool_name = "company_tool"
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name"]}
        tool = fake_mcp_tool(tool_name, return_value=expected_tool_content, args_schema=args_schema)

        test_args = {"company_name": "BMW"}
        tool_call_id = "company_tool_1"
        tool_calls = [{"id": tool_call_id, "name": tool_name, "args": test_args}]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue(value="fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        # mock session with patched mcp setup
        mock_session = self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        # create the wrapped function
        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)

        # rest invocation with sample args
        result, failures = wrapped_func()

        # verify correct interactions
        self._assert_streamable_http_client_call(mock_http_client, tool.metadata["url"], "fake")
        mock_session.initialize.assert_called_once()
        mock_load_tools.assert_called_once_with(mock_session)
        tool.ainvoke.assert_called_once_with(test_args)

        # assert the result matches our expected output
        assert failures == []

        tool_result = result.get(tool_call_id)
        assert tool_result is not None
        assert isinstance(tool_result, tuple)
        assert tool_result[0] == expected_documents
        assert tool_result[1] == expected_tool_metadata

    @pytest.mark.parametrize("expected_tool_result, expected_documents", MCP_TOOL_RESULTS)
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamable_http_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools")
    def test_returns_expected_results_no_args(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_mcp_tool: type[MCPTool.Passing],
        expected_tool_result: tuple[tuple[str, MCPResponseMetadata], str],
        expected_documents: tuple[tuple[str, MCPResponseMetadata], str],
    ):
        """Test that execute_mcp_tools correctly returns results from async tool invocation"""
        expected_tool_content, expected_tool_metadata = expected_tool_result

        # Mock tool with metadata
        tool_name = "company_tool"
        args_schema = {}
        tool = fake_mcp_tool(tool_name, return_value=expected_tool_content, args_schema=args_schema)

        tool_args = {"company_name": "fake company"}
        tool_call_id = "company_tool_1"
        tool_calls = [{"id": tool_call_id, "name": tool_name, "args": tool_args}]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue(value="fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        # mock session with patched mcp setup
        mock_session = self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        # create the wrapped function
        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)

        # rest invocation with sample args
        result, failures = wrapped_func()

        # verify correct interactions
        self._assert_streamable_http_client_call(mock_http_client, tool.metadata["url"], "fake")
        mock_session.initialize.assert_called_once()
        mock_load_tools.assert_called_once_with(mock_session)
        tool.ainvoke.assert_called_once_with(tool_args)

        assert failures == []

        tool_result = result.get(tool_call_id)
        assert tool_result is not None
        assert isinstance(tool_result, tuple)
        assert tool_result[0] == expected_documents
        assert tool_result[1] == expected_tool_metadata

    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamable_http_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
    def test_tool_not_found(
        self, mock_load_tools, mock_http_client, mock_session_class, fake_mcp_tool: type[MCPTool.Passing]
    ):
        """Test execute_mcp_tools raises ValueError when the requested tool is not in the MCP tool list."""

        tool = fake_mcp_tool("dummy_tool", return_value=None)
        tool_calls = [{"id": "missing_tool_1", "name": "missing_tool", "args": {}}]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue(value="fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        result, failures = wrapped_func()

        assert result == {}
        assert failures == [
            tr_models.ToolCallResult.Failure(
                tool_name="missing_tool",
                metadata={"tool_args": {}},
                error=f"MCP Async tool 'missing_tool' not found on server '{tool.metadata['url']}'",
            )
        ]

    def test_sso_token_retrieval_failure(self, fake_mcp_tool: type[MCPTool.Passing]):
        """Test that execute_mcp_tools raises when sso_access_token.get() fails."""
        tool = fake_mcp_tool("dummy_tool", return_value=None)
        mock_token = MagicMock(spec=SensitiveValue)
        mock_token.get.side_effect = RuntimeError("vault unavailable")

        tool_calls = [{"id": "dummy_tool_1", "name": "dummy_tool", "args": {}}]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=mock_token,
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)

        with pytest.raises(RuntimeError, match="vault unavailable"):
            wrapped_func()

    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamable_http_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools")
    def test_intermediate_step_stripped_when_not_in_schema(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_mcp_tool: type[MCPTool.Passing],
    ):
        """Test that is_intermediate_step is removed from args when not required by the tool schema."""
        return_value = "some content"
        tool_name = "company_tool"
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name"]}
        tool = fake_mcp_tool(tool_name, return_value=return_value, args_schema=args_schema)

        # make ainvoke return something format_mcp_tool_response can handle, or use a non-datahub type
        tool.metadata["creator_type"] = MagicMock()  # non-datahub, returns raw result
        tool.ainvoke = AsyncMock(return_value=return_value)

        tool_args = {"company_name": "BMW"}
        tool_calls = [
            {
                "id": "company_tool_1",
                "name": "company_tool",
                "args": {**tool_args, "is_intermediate_step": True},
            }
        ]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue("fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)
        wrapped_func()

        # is_intermediate_step should have been popped before ainvoke was called
        tool.ainvoke.assert_called_once_with(tool_args)

    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamable_http_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools")
    def test_intermediate_step_retained_when_in_schema(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_mcp_tool: type[MCPTool.Passing],
    ):
        """Test that is_intermediate_step is kept in args when the tool schema requires it."""
        return_value = "some content"
        tool_name = "company_tool"
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name", "is_intermediate_step"]}
        tool = fake_mcp_tool(tool_name, return_value=return_value, args_schema=args_schema)

        tool.metadata["creator_type"] = MagicMock()
        tool.ainvoke = AsyncMock(return_value=return_value)

        tool_args = {"company_name": "BMW", "is_intermediate_step": True}
        tool_calls = [
            {
                "id": "company_tool_1",
                "name": "company_tool",
                "args": tool_args,
            }
        ]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue("fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)
        wrapped_func()

        # is_intermediate_step should be preserved
        tool.ainvoke.assert_called_once_with(tool_args)

    @patch("redbox.graph.nodes.runner.wrap_async.asyncio")
    def test_asyncio_run_failure(self, mock_asyncio, fake_mcp_tool: type[MCPTool.Passing]):
        """Test that execute_mcp_tools re-raises when asyncio.run itself fails."""
        tool = fake_mcp_tool("dummy_tool", return_value=None)
        mock_asyncio.run.side_effect = RuntimeError("event loop closed")

        tool_calls = [
            {"id": "dummy_tool_1", "name": "dummy_tool", "args": {"foo": "bar", "is_intermediate_step": True}}
        ]

        mcp_input = tr_models.ToolCallRequest.MCPAsync(
            mcp_server=tool.metadata["url"],
            access_token=SensitiveValue("fake"),
            creator_type=ChunkCreatorType.datahub,
            tool_calls=tool_calls,
        )

        wrapped_func = execute_mcp_tools(mcp_input=mcp_input)

        with pytest.raises(RuntimeError, match="event loop closed"):
            wrapped_func()


@pytest.mark.parametrize(
    "token_input, expected_output",
    [
        (None, {}),
        ("", {}),
        ("   ", {}),
        ("simple-token-123", {"Authorization": "Bearer simple-token-123"}),
        ("Bearer already-has-prefix", {"Authorization": "Bearer already-has-prefix"}),
        ("bearer lowercase-prefix", {"Authorization": "bearer lowercase-prefix"}),
        ("  token-with-spaces  ", {"Authorization": "Bearer token-with-spaces"}),
    ],
)
def test_get_mcp_headers_logic(
    token_input: None
    | Literal[""]
    | Literal["   "]
    | Literal["simple-token-123"]
    | Literal["Bearer already-has-prefix"]
    | Literal["bearer lowercase-prefix"]
    | Literal["  token-with-spaces  "],
    expected_output: dict[str, str],
):
    """Verify that headers are correctly formatted or returned empty based on input."""
    assert _get_mcp_headers(token_input) == expected_output


def test_get_mcp_headers_no_args():
    """Verify the default parameter behavior (None)."""
    assert _get_mcp_headers() == {}

import pytest
import logging
from unittest.mock import Mock, patch
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool

from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.runner import ToolRunner


class TestToolRunner:
    """Comprehensive test suite for ToolRunner class."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock RedboxState."""
        return Mock(spec=RedboxState)

    @pytest.fixture
    def mock_tool(self):
        """Create a mock StructuredTool."""
        tool = Mock(spec=StructuredTool)
        tool.name = "test_tool"
        tool.func = Mock(return_value="test result")
        tool.coroutine = None
        tool.invoke = Mock(return_value="test result")
        return tool

    @pytest.fixture
    def tool_runner(self, mock_tool, mock_state):
        """Create a ToolRunner instance with default settings."""
        return ToolRunner(tools=[mock_tool], state=mock_state, max_workers=2, is_loop=False, parallel_timeout=30.0)

    # Test initialization
    def test_init_creates_executor(self, mock_tool, mock_state):
        """Test that initialization creates ThreadPoolExecutor."""
        runner = ToolRunner(tools=[mock_tool], state=mock_state, max_workers=4, is_loop=True, parallel_timeout=60.0)
        assert runner.tools == [mock_tool]
        assert runner.state == mock_state
        assert runner.executor is not None
        assert runner.is_loop is True
        assert runner.parallel_timeout == 60.0
        assert runner.log_stub.startswith("[run_tools_parallel run_id=")

    # Test submit method
    @pytest.mark.parametrize(
        "tool_call,expected_args_key",
        [
            ({"name": "test_tool", "args": {"param1": "value1"}}, "param1"),
            ({"name": "test_tool", "args": {}}, None),
            ({"name": "test_tool", "args": {"key": "val"}}, "key"),
        ],
    )
    def test_submit_valid_tool_call(self, tool_runner, mock_tool, tool_call, expected_args_key):
        """Test submitting valid tool calls with various arguments."""
        result = tool_runner.submit(tool_call)
        assert result is not None
        future, metadata = result
        assert isinstance(future, Future)
        assert metadata["name"] == "test_tool"
        assert metadata["intermediate_step"] == "False"

    def test_submit_tool_not_found(self, tool_runner):
        """Test that ToolNotFoundError is raised when tool doesn't exist."""
        tool_call = {"name": "nonexistent_tool", "args": {}}
        with pytest.raises(tool_exceptions.ToolNotFoundError, match="Tool 'nonexistent_tool' not found"):
            tool_runner.submit(tool_call)

    @pytest.mark.parametrize(
        "invalid_args,expected_error",
        [
            ("not_a_dict", tool_exceptions.ToolValidationError),
            (["list", "args"], tool_exceptions.ToolValidationError),
            (123, tool_exceptions.ToolValidationError),
        ],
    )
    def test_submit_invalid_args(self, tool_runner, invalid_args, expected_error):
        """Test that ToolValidationError is raised for non-dict arguments."""
        tool_call = {"name": "test_tool", "args": invalid_args}
        with pytest.raises(expected_error, match="expected dict"):
            tool_runner.submit(tool_call)

    def test_submit_async_tool_with_loop(self, mock_state):
        """Test submitting async tool in loop mode."""
        async_tool = Mock(spec=StructuredTool)
        async_tool.name = "async_tool"
        async_tool.func = None
        async_tool.coroutine = Mock

        runner = ToolRunner(tools=[async_tool], state=mock_state, max_workers=2, is_loop=True, parallel_timeout=30.0)

        tool_call = {
            "name": "async_tool",
            "args": {"is_intermediate_step": "True"},
        }

        with patch("redbox.graph.nodes.runner.runner.wrap_async_tool"):
            result = runner.submit(tool_call)
            assert result is not None
            future, metadata = result
            assert metadata["intermediate_step"] == "True"

    def test_submit_execution_error(self, tool_runner, mock_tool):
        """Test that ToolExecutionError is raised when submission fails."""
        tool_runner.executor.submit = Mock(side_effect=Exception("Submission failed"))
        tool_call = {"name": "test_tool", "args": {}}
        with pytest.raises(tool_exceptions.ToolExecutionError, match="Failed to submit tool"):
            tool_runner.submit(tool_call)

    # Test parse method
    @pytest.mark.parametrize(
        "response,is_loop,expected_result",
        [
            ("simple response", False, "simple response"),
            (("tuple response", Mock(spec=MCPResponseMetadata)), False, "tuple response"),
            ("loop response", True, "loop response"),
        ],
    )
    def test_parse_success(self, mock_state, response, is_loop, expected_result):
        """Test parsing successful tool responses."""
        runner = ToolRunner(tools=[], state=mock_state, max_workers=2, is_loop=is_loop, parallel_timeout=30.0)

        future = Mock(spec=Future)
        future.result.return_value = response
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        result = runner.parse(future, metadata)
        assert isinstance(result, AIMessage)
        if isinstance(expected_result, str):
            assert expected_result in str(result.content)

    def test_parse_timeout_error(self, tool_runner):
        """Test that ToolTimeoutError is raised on timeout."""
        future = Mock(spec=Future)
        future.result.side_effect = FuturesTimeoutError()
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        with pytest.raises(tool_exceptions.ToolTimeoutError, match="timed out"):
            tool_runner.parse(future, metadata)

    def test_parse_execution_error(self, tool_runner):
        """Test that ToolExecutionError is raised on execution failure."""
        future = Mock(spec=Future)
        future.result.side_effect = Exception("Tool failed")
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        with pytest.raises(tool_exceptions.ToolExecutionError, match="failed"):
            tool_runner.parse(future, metadata)

    def test_parse_none_response(self, tool_runner):
        """Test that ToolExecutionError is raised when response is None."""
        future = Mock(spec=Future)
        future.result.return_value = None
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        with pytest.raises(tool_exceptions.ToolExecutionError, match="returned None"):
            tool_runner.parse(future, metadata)

    @pytest.mark.parametrize(
        "invalid_response",
        [
            "",
            "   ",
            ("   ", Mock(spec=MCPResponseMetadata)),
            ("", Mock(spec=MCPResponseMetadata)),
        ],
    )
    def test_parse_empty_response(self, tool_runner, invalid_response):
        """Test that ToolValidationError is raised for empty responses."""
        future = Mock(spec=Future)
        future.result.return_value = invalid_response
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        with pytest.raises(tool_exceptions.ToolValidationError, match="empty or whitespace-only"):
            tool_runner.parse(future, metadata)

    def test_parse_with_user_feedback_required(self, mock_state):
        """Test parsing response with user feedback requirement in loop mode."""
        runner = ToolRunner(tools=[], state=mock_state, max_workers=2, is_loop=True, parallel_timeout=30.0)

        metadata_obj = Mock(spec=MCPResponseMetadata)
        metadata_obj.user_feedback = Mock(spec=MCPResponseMetadata.UserFeedback)
        metadata_obj.user_feedback.required = True
        metadata_obj.user_feedback.reason = "Need clarification"

        response = ("result", metadata_obj)
        future = Mock(spec=Future)
        future.result.return_value = response
        metadata = {"name": "test_tool", "intermediate_step": "True"}

        result = runner.parse(future, metadata)
        assert isinstance(result, AIMessage)

    # Test _submit_all method
    def test_submit_all_success(self, tool_runner):
        """Test submitting multiple tool calls successfully."""
        tool_calls = [
            {"name": "test_tool", "args": {"param": "value1"}},
            {"name": "test_tool", "args": {"param": "value2"}},
        ]

        futures = tool_runner._submit_all(tool_calls)
        assert len(futures) == 2
        assert all(isinstance(f, Future) for f in futures.keys())

    def test_submit_all_with_failures(self, tool_runner, caplog):
        """Test that _submit_all continues on individual failures."""
        tool_calls = [
            {"name": "test_tool", "args": {"param": "value1"}},
            {"name": "nonexistent_tool", "args": {}},
            {"name": "test_tool", "args": "invalid"},
        ]

        with caplog.at_level(logging.WARNING):
            futures = tool_runner._submit_all(tool_calls)

        assert len(futures) == 1  # Only one valid submission
        assert "not found" in caplog.text
        assert "validation error" in caplog.text

    # Test _collect method
    def test_collect_all_success(self, tool_runner):
        """Test collecting all successful futures."""
        future1 = Mock(spec=Future)
        future1.result.return_value = "result1"

        future2 = Mock(spec=Future)
        future2.result.return_value = "result2"

        futures = {
            future1: {"name": "tool1", "intermediate_step": "False"},
            future2: {"name": "tool2", "intermediate_step": "False"},
        }

        with patch("redbox.graph.nodes.runner.runner.as_completed", return_value=[future1, future2]):
            responses = tool_runner._collect(futures)

        assert responses is not None
        assert len(responses) == 2
        assert all(isinstance(r, AIMessage) for r in responses)

    def test_collect_all_failures(self, tool_runner, caplog):
        """Test that _collect returns None when all tools fail."""
        future1 = Mock(spec=Future)
        future1.result.side_effect = Exception("Failed")

        futures = {future1: {"name": "tool1", "intermediate_step": "False"}}

        with patch("redbox.graph.nodes.runner.runner.as_completed", return_value=[future1]):
            with caplog.at_level(logging.WARNING):
                responses = tool_runner._collect(futures)

        assert responses is None
        assert "Every tool execution has failed" in caplog.text

    def test_collect_partial_success(self, tool_runner, caplog):
        """Test collecting with some successful and some failed futures."""
        future1 = Mock(spec=Future)
        future1.result.return_value = "success"

        future2 = Mock(spec=Future)
        future2.result.side_effect = FuturesTimeoutError()

        futures = {
            future1: {"name": "tool1", "intermediate_step": "False"},
            future2: {"name": "tool2", "intermediate_step": "False"},
        }

        with patch("redbox.graph.nodes.runner.runner.as_completed", return_value=[future1, future2]):
            with caplog.at_level(logging.WARNING):
                responses = tool_runner._collect(futures)

        assert responses is not None
        assert len(responses) == 1
        assert "timed out" in caplog.text
        assert "1 tool(s) failed" in caplog.text

    # Test run method (integration)
    def test_run_end_to_end(self, tool_runner):
        """Test complete run flow from submission to collection."""
        tool_calls = [{"name": "test_tool", "args": {"param": "value"}}]

        with patch.object(tool_runner, "_submit_all") as mock_submit:
            with patch.object(tool_runner, "_collect") as mock_collect:
                mock_future = Mock(spec=Future)
                mock_submit.return_value = {mock_future: {"name": "test_tool", "intermediate_step": "False"}}
                mock_collect.return_value = [AIMessage("result")]

                result = tool_runner.run(tool_calls)

                mock_submit.assert_called_once_with(tool_calls=tool_calls)
                mock_collect.assert_called_once
                assert result is not None
                assert len(result) == 1

    def test_run_with_no_tool_calls(self, tool_runner):
        """Test run with empty tool calls list."""
        result = tool_runner.run(tool_calls=[])
        # Should return None or empty list depending on implementation
        assert result is None or result == []

    @pytest.mark.parametrize(
        "max_workers,parallel_timeout",
        [
            (1, 10.0),
            (5, 60.0),
            (10, 120.0),
        ],
    )
    def test_run_with_different_configurations(self, mock_tool, mock_state, max_workers, parallel_timeout):
        """Test ToolRunner with various configuration parameters."""
        runner = ToolRunner(
            tools=[mock_tool],
            state=mock_state,
            max_workers=max_workers,
            is_loop=False,
            parallel_timeout=parallel_timeout,
        )

        assert runner.executor._max_workers == max_workers
        assert runner.parallel_timeout == parallel_timeout

    # Test logging behaviour
    def test_logging_on_tool_not_found(self, tool_runner, caplog):
        """Test that appropriate warnings are logged for tool not found."""
        tool_calls = [{"name": "missing_tool", "args": {}}]

        with caplog.at_level(logging.WARNING):
            tool_runner._submit_all(tool_calls)

        assert "not found" in caplog.text

    def test_logging_on_successful_completion(self, tool_runner, caplog):
        """Test logging on successful tool completion."""
        future = Mock(spec=Future)
        future.result.return_value = "test result"
        metadata = {"name": "test_tool", "intermediate_step": "False"}

        with caplog.at_level(logging.WARNING):
            tool_runner.parse(future, metadata)

        assert "This is what I got from tool" in caplog.text
        assert "response not None" in caplog.text

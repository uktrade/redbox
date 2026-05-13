import pytest
import logging
import time
from asyncio import CancelledError
from unittest.mock import Mock, patch
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool

from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.runner import ToolRunner, ToolExecutionResult


@pytest.fixture
def mock_state():
    return Mock(spec=RedboxState)


@pytest.fixture
def mock_tool():
    """Synchronous StructuredTool that always succeeds."""
    tool = Mock(spec=StructuredTool)
    tool.name = "test_tool"
    tool.func = Mock(return_value="test result")
    tool.coroutine = None
    tool.invoke = Mock(return_value="test result")
    return tool


@pytest.fixture
def mock_failing_tool():
    """Synchronous StructuredTool whose func and invoke always raise."""
    tool = Mock(spec=StructuredTool)
    tool.name = "test_failing_tool"
    tool.func = Mock(side_effect=Exception("Tool execution failed"))
    tool.coroutine = None
    tool.invoke = Mock(side_effect=Exception("Tool execution failed"))
    return tool


@pytest.fixture
def mock_async_tool():
    """Async StructuredTool (func=None, coroutine set)."""
    tool = Mock(spec=StructuredTool)
    tool.name = "async_tool"
    tool.func = None
    tool.coroutine = Mock  # truthy non-None sentinel
    return tool


@pytest.fixture
def tool_runner(mock_tool, mock_state):
    return ToolRunner(tools=[mock_tool], state=mock_state, max_workers=2, is_loop=False, parallel_timeout=30.0)


@pytest.fixture
def loop_runner(mock_tool, mock_state):
    return ToolRunner(tools=[mock_tool], state=mock_state, max_workers=2, is_loop=True, parallel_timeout=30.0)


@pytest.fixture
def mock_tool_b():
    """A second distinct synchronous tool with a different return value."""
    tool = Mock(spec=StructuredTool)
    tool.name = "other_tool"
    tool.func = Mock(return_value="other result")
    tool.coroutine = None
    tool.invoke = Mock(return_value="other result")
    return tool


@pytest.fixture
def multi_tool_runner(mock_tool, mock_tool_b, mock_state):
    """ToolRunner with two distinct tools registered."""
    return ToolRunner(
        tools=[mock_tool, mock_tool_b], state=mock_state, max_workers=4, is_loop=False, parallel_timeout=30.0
    )


def _future_returning(value) -> Mock:
    f = Mock(spec=Future)
    f.result.return_value = value
    return f


def _future_raising(exc) -> Mock:
    f = Mock(spec=Future)
    f.result.side_effect = exc
    return f


def _plain_metadata(name="test_tool") -> dict:
    return {"name": name, "intermediate_step": "False"}


def _mcp_meta(feedback_required=False, reason=None):
    meta = Mock(spec=MCPResponseMetadata)
    meta.user_feedback = Mock(spec=MCPResponseMetadata.UserFeedback)
    meta.user_feedback.required = feedback_required
    meta.user_feedback.reason = reason
    return meta


class TestToolRunner:
    @pytest.mark.parametrize(
        "max_workers,is_loop,parallel_timeout",
        [
            (1, False, 10.0),
            (4, True, 60.0),
            (10, False, 120.0),
        ],
    )
    def test_init_stores_attributes(self, mock_tool, mock_state, max_workers, is_loop, parallel_timeout):
        runner = ToolRunner(
            tools=[mock_tool],
            state=mock_state,
            max_workers=max_workers,
            is_loop=is_loop,
            parallel_timeout=parallel_timeout,
        )
        assert runner.tools == [mock_tool]
        assert runner.state is mock_state
        assert runner.executor._max_workers == max_workers
        assert runner.is_loop is is_loop
        assert runner.parallel_timeout == parallel_timeout
        assert runner.log_stub.startswith("[run_tools_parallel run_id=")

    @pytest.mark.parametrize(
        "raises,expected_exc,match",
        [
            (None, None, None),  # clean exit — executor still shuts down
            (RuntimeError("boom"), RuntimeError, "boom"),  # exception propagates out
        ],
        ids=["clean-exit", "exception-propagates"],
    )
    def test_context_manager(self, mock_tool, mock_state, raises, expected_exc, match):
        runner = ToolRunner(tools=[mock_tool], state=mock_state, max_workers=2, is_loop=False, parallel_timeout=5.0)
        with patch.object(runner.executor, "shutdown") as mock_shutdown:
            if expected_exc:
                with pytest.raises(expected_exc, match=match):
                    with runner:
                        raise raises
            else:
                with runner:
                    pass
            mock_shutdown.assert_called_once_with(wait=True)


class TestToolRunner_Submit:
    @pytest.mark.parametrize(
        "args,expected_intermediate_step",
        [
            ({"p": "v"}, "False"),
            ({}, "False"),
        ],
        ids=["with-args", "empty-args"],
    )
    def test_sync_tool_returns_future_and_metadata(self, tool_runner, args, expected_intermediate_step):
        future, metadata = tool_runner.submit({"name": "test_tool", "args": args})
        assert isinstance(future, Future)
        assert metadata["name"] == "test_tool"
        assert metadata["intermediate_step"] == expected_intermediate_step

    @pytest.mark.parametrize(
        "args,expected_intermediate_step",
        [
            ({"is_intermediate_step": "True"}, "True"),
            ({"is_intermediate_step": "False"}, "False"),
            ({}, "False"),
        ],
        ids=["intermediate-true", "intermediate-false", "intermediate-default"],
    )
    def test_async_tool_in_loop_mode_intermediate_step(
        self, mock_async_tool, mock_state, args, expected_intermediate_step
    ):
        runner = ToolRunner(
            tools=[mock_async_tool], state=mock_state, max_workers=2, is_loop=True, parallel_timeout=30.0
        )
        with patch("redbox.graph.nodes.runner.runner.wrap_async_tool"):
            _, metadata = runner.submit({"name": "async_tool", "args": args})
        assert metadata["intermediate_step"] == expected_intermediate_step

    @pytest.mark.parametrize(
        "tool_call,exc_type,match",
        [
            (
                {"name": "nonexistent_tool", "args": {}},
                tool_exceptions.ToolNotFoundError,
                r"Tool 'nonexistent_tool' not found",
            ),
            (
                {"name": "test_tool", "args": "not_a_dict"},
                tool_exceptions.ToolValidationError,
                r"Invalid input for tool 'test_tool': expected dict, got 'str'",
            ),
            (
                {"name": "test_tool", "args": ["list", "args"]},
                tool_exceptions.ToolValidationError,
                r"Invalid input for tool 'test_tool': expected dict, got 'list'",
            ),
            (
                {"name": "test_tool", "args": 123},
                tool_exceptions.ToolValidationError,
                r"Invalid input for tool 'test_tool': expected dict, got 'int'",
            ),
            (
                {"name": "test_tool", "args": None},
                tool_exceptions.ToolValidationError,
                r"Invalid input for tool 'test_tool': expected dict, got 'NoneType'",
            ),
        ],
        ids=["unknown-tool", "str-args", "list-args", "int-args", "none-args"],
    )
    def test_raises_on_invalid_tool_call(self, tool_runner, tool_call, exc_type, match):
        with pytest.raises(exc_type, match=match):
            tool_runner.submit(tool_call)

    def test_sync_tool_injects_state_into_invoke_args(self, tool_runner, mock_tool, mock_state):
        """submit must pass state= to invoke so tools can access RedboxState."""
        future, _ = tool_runner.submit({"name": "test_tool", "args": {"p": "v"}})
        future.result(timeout=5)  # let the thread run
        mock_tool.invoke.assert_called_once_with({"p": "v", "state": mock_state})

    def test_async_tool_in_non_loop_mode_does_not_read_intermediate_step(self, mock_async_tool, mock_state):
        runner = ToolRunner(
            tools=[mock_async_tool], state=mock_state, max_workers=2, is_loop=False, parallel_timeout=30.0
        )
        with patch("redbox.graph.nodes.runner.runner.wrap_async_tool"):
            _, metadata = runner.submit({"name": "async_tool", "args": {"is_intermediate_step": "True"}})
        # is_loop=False - intermediate_step flag is ignored, always "False"
        assert metadata["intermediate_step"] == "False"

    def test_raises_tool_execution_error_when_executor_submit_fails(self, tool_runner):
        tool_runner.executor.submit = Mock(side_effect=Exception("Submission failed"))
        with pytest.raises(
            tool_exceptions.ToolExecutionError,
            match=r"Failed to submit tool 'test_tool' for execution: Submission failed",
        ):
            tool_runner.submit({"name": "test_tool", "args": {}})


class TestToolRunner_SubmitAll:
    @pytest.mark.parametrize(
        "tool_calls,expected_count",
        [
            ([], 0),
            ([{"name": "test_tool", "args": {"p": "v1"}}], 1),
            # two calls to the same tool
            (
                [{"name": "test_tool", "args": {"p": "v1"}}, {"name": "test_tool", "args": {"p": "v2"}}],
                2,
            ),
            # two calls to distinct tools — both must be submitted
            (
                [{"name": "test_tool", "args": {}}, {"name": "other_tool", "args": {}}],
                2,
            ),
        ],
        ids=["empty", "single", "same-tool-twice", "two-distinct-tools"],
    )
    def test_returns_correct_future_count(self, multi_tool_runner, tool_calls, expected_count):
        futures = multi_tool_runner._submit_all(tool_calls)
        assert len(futures) == expected_count
        assert all(isinstance(f, Future) for f in futures)

    @pytest.mark.parametrize(
        "tool_calls,expected_count,log_level,expected_log_fragments",
        [
            (
                [{"name": "ghost_tool", "args": {}}],
                0,
                logging.WARNING,
                ["not found"],
            ),
            (
                [{"name": "test_tool", "args": "bad"}],
                0,
                logging.WARNING,
                ["validation error"],
            ),
            (
                [
                    {"name": "test_tool", "args": {"p": "v"}},  # valid
                    {"name": "ghost_tool", "args": {}},  # not found
                    {"name": "test_tool", "args": "bad"},  # invalid args
                ],
                1,
                logging.WARNING,
                ["not found", "validation error"],
            ),
        ],
        ids=["unknown-tool", "invalid-args", "mixed"],
    )
    def test_skips_bad_calls_and_logs(
        self, tool_runner, caplog, tool_calls, expected_count, log_level, expected_log_fragments
    ):
        with caplog.at_level(log_level):
            futures = tool_runner._submit_all(tool_calls)
        assert len(futures) == expected_count
        for fragment in expected_log_fragments:
            assert fragment in caplog.text

    def test_logs_error_for_unexpected_submission_exception(self, tool_runner, caplog):
        tool_runner.executor.submit = Mock(side_effect=RuntimeError("kaboom"))
        with caplog.at_level(logging.ERROR):
            futures = tool_runner._submit_all([{"name": "test_tool", "args": {}}])
        assert futures == {}


class TestToolRunner_Parse:
    @pytest.mark.parametrize(
        "response,is_loop,metadata_override,expected_content",
        [
            # non-loop: plain string - content is the string itself
            (
                "hello",
                False,
                None,
                "hello",
            ),
            # non-loop: MCP tuple - content is the unwrapped first element (string)
            (
                ("payload", _mcp_meta()),
                False,
                None,
                "payload",
            ),
            # loop: plain string - content is the string itself
            (
                "loop result",
                True,
                None,
                "loop result",
            ),
            # loop: MCP tuple, no feedback - content is a list (result, status, intermediate_step)
            (
                ("ok", _mcp_meta(False)),
                True,
                {"name": "test_tool", "intermediate_step": "True"},
                ["ok", "pass", "True"],
            ),
            # loop: MCP tuple, feedback required - content is a list (result, status, intermediate_step, reason)
            (
                ("ok", _mcp_meta(True, "why")),
                True,
                {"name": "test_tool", "intermediate_step": "True"},
                ["ok", "pass", "True", "why"],
            ),
        ],
        ids=[
            "non-loop-plain-string",
            "non-loop-mcp-tuple",
            "loop-plain-string",
            "loop-mcp-no-feedback",
            "loop-mcp-with-feedback",
        ],
    )
    def test_successful_parse_returns_ai_message(
        self, tool_runner, loop_runner, response, is_loop, metadata_override, expected_content
    ):
        runner = loop_runner if is_loop else tool_runner
        metadata = metadata_override or _plain_metadata()
        result = runner.parse(_future_returning(response), metadata)
        assert isinstance(result, AIMessage)
        assert result.content == expected_content

    def test_logs_receipt_and_non_none_on_success(self, tool_runner, caplog):
        with caplog.at_level(logging.WARNING):
            tool_runner.parse(_future_returning("hello"), _plain_metadata())
        assert "This is what I got from tool" in caplog.text
        assert "response not None" in caplog.text

    @pytest.mark.parametrize(
        "future,exc_type,match",
        [
            (
                _future_raising(FuturesTimeoutError()),
                tool_exceptions.ToolTimeoutError,
                r"Tool 'test_tool' timed out after 30\.0s",
            ),
            (
                _future_raising(CancelledError()),
                tool_exceptions.ToolExecutionError,
                r"Tool 'test_tool' failed:",
            ),
            (
                _future_raising(Exception("kaboom")),
                tool_exceptions.ToolExecutionError,
                r"Tool 'test_tool' failed: kaboom",
            ),
            (
                _future_returning(None),
                tool_exceptions.ToolExecutionError,
                r"Tool 'test_tool' returned None — may have failed or timed out",
            ),
            (
                _future_returning(""),
                tool_exceptions.ToolValidationError,
                r"Tool 'test_tool' returned empty or whitespace-only response",
            ),
            (
                _future_returning("   "),
                tool_exceptions.ToolValidationError,
                r"Tool 'test_tool' returned empty or whitespace-only response",
            ),
            (
                _future_returning(("   ", Mock(spec=MCPResponseMetadata))),
                tool_exceptions.ToolValidationError,
                r"Tool 'test_tool' returned empty or whitespace-only response",
            ),
            (
                _future_returning(("", Mock(spec=MCPResponseMetadata))),
                tool_exceptions.ToolValidationError,
                r"Tool 'test_tool' returned empty or whitespace-only response",
            ),
        ],
        ids=[
            "futures-timeout",
            "cancelled-future",
            "generic-exception",
            "none-response",
            "empty-string",
            "whitespace-string",
            "whitespace-mcp-tuple",
            "empty-mcp-tuple",
        ],
    )
    def test_raises_on_bad_future_or_response(self, tool_runner, future, exc_type, match):
        with pytest.raises(exc_type, match=match):
            tool_runner.parse(future, _plain_metadata())


class TestToolRunner_Collect:
    @pytest.mark.parametrize(
        "futures_spec,expected_responses,expected_failed,expected_log_fragments",
        [
            # all succeed
            (
                [("r1", "tool1"), ("r2", "tool2")],
                [AIMessage("r1"), AIMessage("r2")],
                [],
                ["Completed"],
            ),
            # empty input treated as total failure
            (
                [],
                [],
                [],
                ["Every tool execution has failed"],
            ),
            # single exception failure
            (
                [(Exception("boom"), "tool1")],
                [],
                ["tool1"],
                ["Every tool execution has failed"],
            ),
            # timeout failure
            (
                [(FuturesTimeoutError(), "slow_tool")],
                [],
                ["slow_tool"],
                ["timed out", "Every tool execution has failed"],
            ),
            # partial: one success, one timeout
            (
                [("ok", "good_tool"), (FuturesTimeoutError(), "slow_tool")],
                [AIMessage("ok")],
                ["slow_tool"],
                ["1 tool(s) failed", "Completed"],
            ),
            # ToolValidationError (e.g. empty response) lands in failed_tools too
            (
                [("ok", "good_tool"), ("", "empty_tool")],
                [AIMessage("ok")],
                ["empty_tool"],
                ["1 tool(s) failed", "Completed"],
            ),
            # multiple distinct tools all succeed — responses preserve insertion order
            (
                [("alpha", "tool_a"), ("beta", "tool_b"), ("gamma", "tool_c")],
                [AIMessage("alpha"), AIMessage("beta"), AIMessage("gamma")],
                [],
                ["Completed"],
            ),
        ],
        ids=[
            "all-success",
            "empty",
            "single-failure",
            "timeout-failure",
            "partial-success",
            "validation-error-failure",
            "three-distinct-tools",
        ],
    )
    def test_collect(
        self,
        tool_runner,
        caplog,
        futures_spec,
        expected_responses,
        expected_failed,
        expected_log_fragments,
    ):
        futures = {
            (_future_raising(v) if isinstance(v, Exception) else _future_returning(v)): _plain_metadata(name)
            for v, name in futures_spec
        }
        with caplog.at_level(logging.WARNING):
            result = tool_runner._collect(futures)

        assert isinstance(result, ToolExecutionResult)
        assert result.responses == expected_responses
        assert result.failed_tools == expected_failed
        for fragment in expected_log_fragments:
            assert fragment in caplog.text


class TestToolRunner_Run:
    @pytest.mark.parametrize(
        "tool_calls,expected_responses,expected_failed",
        [
            # empty - no responses
            (
                [],
                [],
                [],
            ),
            # single call to test_tool - its specific return value
            (
                [{"name": "test_tool", "args": {}}],
                [AIMessage("test result")],
                [],
            ),
            # single call to other_tool - its specific return value
            (
                [{"name": "other_tool", "args": {}}],
                [AIMessage("other result")],
                [],
            ),
            # two calls to the same tool with different args - same return value twice
            (
                [{"name": "test_tool", "args": {"p": "v1"}}, {"name": "test_tool", "args": {"p": "v2"}}],
                [AIMessage("test result"), AIMessage("test result")],
                [],
            ),
            # one call to each distinct tool - each returns its own value
            (
                [{"name": "test_tool", "args": {}}, {"name": "other_tool", "args": {}}],
                [AIMessage("test result"), AIMessage("other result")],
                [],
            ),
            # unknown tool alongside valid call - one response, ghost skipped at submit
            (
                [{"name": "test_tool", "args": {}}, {"name": "ghost_tool", "args": {}}],
                [AIMessage("test result")],
                [],  # ghost_tool is dropped at _submit_all, never reaches _collect
            ),
            # one tool succeeds, other_tool's future raises — lands in failed_tools
            (
                [{"name": "test_tool", "args": {}}, {"name": "other_tool", "args": {}}],
                [AIMessage("test result")],
                ["other_tool"],
            ),
        ],
        ids=[
            "empty",
            "single-test-tool",
            "single-other-tool",
            "same-tool-twice",
            "two-distinct-tools",
            "valid-plus-unknown",
            "one-succeeds-one-fails",
        ],
    )
    def test_run_returns_correct_result(
        self, multi_tool_runner, mock_tool_b, tool_calls, expected_responses, expected_failed
    ):
        # For the "one-succeeds-one-fails" case mock_tool_b.invoke raises.
        if expected_failed == ["other_tool"]:
            mock_tool_b.invoke.side_effect = Exception("other_tool blew up")
        result = multi_tool_runner.run(tool_calls)
        assert isinstance(result, ToolExecutionResult)
        assert result.responses == expected_responses
        assert result.failed_tools == expected_failed

    @pytest.mark.parametrize(
        "submit_all_side_effect,expected_exc",
        [
            (None, None),
            (RuntimeError("crash"), RuntimeError),
        ],
        ids=["success", "exception"],
    )
    def test_executor_always_shuts_down(self, tool_runner, submit_all_side_effect, expected_exc):
        with patch.object(tool_runner.executor, "shutdown") as mock_shutdown:
            if submit_all_side_effect:
                with patch.object(tool_runner, "_submit_all", side_effect=submit_all_side_effect):
                    with pytest.raises(expected_exc):
                        tool_runner.run([{"name": "test_tool", "args": {}}])
            else:
                tool_runner.run([{"name": "test_tool", "args": {}}])
        mock_shutdown.assert_called_once_with(wait=True)

    def test_delegates_to_submit_all_and_collect(self, tool_runner):
        mock_future = Mock(spec=Future)
        stub_futures = {mock_future: _plain_metadata()}
        stub_result = ToolExecutionResult(responses=[AIMessage("r")])

        with patch.object(tool_runner, "_submit_all", return_value=stub_futures) as ms:
            with patch.object(tool_runner, "_collect", return_value=stub_result) as mc:
                result = tool_runner.run([{"name": "test_tool", "args": {}}])

        ms.assert_called_once_with(tool_calls=[{"name": "test_tool", "args": {}}])
        mc.assert_called_once_with(futures=stub_futures)
        assert result is stub_result

    def test_parallel_execution_is_faster_than_sequential(self, mock_state):
        """Three 100 ms tasks with 3 workers should finish in ~100 ms, not 300 ms."""

        def slow_invoke(args):
            time.sleep(0.1)
            return "done"

        tool = Mock(spec=StructuredTool)
        tool.name = "slow_tool"
        tool.func = Mock(side_effect=slow_invoke)
        tool.coroutine = None
        tool.invoke = Mock(side_effect=slow_invoke)

        runner = ToolRunner(tools=[tool], state=mock_state, max_workers=3, is_loop=False, parallel_timeout=5.0)
        start = time.time()
        result = runner.run([{"name": "slow_tool", "args": {}} for _ in range(3)])
        elapsed = time.time() - start

        assert elapsed < 0.25, f"Expected ~0.1s with parallelism, got {elapsed:.2f}s"
        assert result.responses == [AIMessage("done"), AIMessage("done"), AIMessage("done")]

    @pytest.mark.parametrize(
        "max_workers,parallel_timeout",
        [(1, 10.0), (5, 60.0), (10, 120.0)],
    )
    def test_run_honours_configuration(self, mock_tool, mock_state, max_workers, parallel_timeout):
        runner = ToolRunner(
            tools=[mock_tool],
            state=mock_state,
            max_workers=max_workers,
            is_loop=False,
            parallel_timeout=parallel_timeout,
        )
        assert runner.executor._max_workers == max_workers
        assert runner.parallel_timeout == parallel_timeout

import pytest
import logging
import time
from asyncio import CancelledError
from unittest.mock import Mock, patch
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from langchain_core.messages import AIMessage
from langchain.tools import StructuredTool

from redbox.api.wrapper import SensitiveValue
from redbox.models.chain import RedboxState
from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.runner import exceptions as tool_exceptions
from redbox.graph.nodes.runner.runner import ToolRunner
import redbox.graph.nodes.runner.models as tr_models
from redbox.models.file import ChunkCreatorType


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
    def test_sync_tool_returns_future_and_metadata(self, tool_runner: ToolRunner, args, expected_intermediate_step):
        submitted_request = tool_runner.submit_sync(
            sync_call=tr_models.ToolCallRequest.Sync(tool_call={"id": "test", "name": "test_tool", "args": args})
        )
        assert isinstance(submitted_request.future, Future)
        assert submitted_request.name == "test_tool"
        assert submitted_request.metadata["intermediate_step"] == expected_intermediate_step

    # @pytest.mark.parametrize(
    #     "args,expected_intermediate_step",
    #     [
    #         ({"is_intermediate_step": "True"}, "True"),
    #         ({"is_intermediate_step": "False"}, "False"),
    #         ({}, "False"),
    #     ],
    #     ids=["intermediate-true", "intermediate-false", "intermediate-default"],
    # )
    # def test_async_tool_in_loop_mode_intermediate_step(
    #     self, mock_async_tool, mock_state, args, expected_intermediate_step
    # ):
    #     runner = ToolRunner(
    #         tools=[mock_async_tool], state=mock_state, max_workers=2, is_loop=True, parallel_timeout=30.0
    #     )
    #     with patch("redbox.graph.nodes.runner.runner.execute_mcp_tools"):
    #         _, metadata = runner.submit_mcp_async(mcp_server="", tool_call=tr_models.ToolCallWrapper.MCPAsync(mcp_server="", access_token=SensitiveValue(value="fake"), creator_type=ChunkCreatorType.datahub, tool_calls=[{"id": "test", "name": "async_tool", "args": args}]))
    #     assert metadata["intermediate_step"] == expected_intermediate_step

    def test_sync_tool_injects_state_into_invoke_args(self, tool_runner: ToolRunner, mock_tool, mock_state):
        """submit must pass state= to invoke so tools can access RedboxState."""
        submitted_request = tool_runner.submit_sync(
            sync_call=tr_models.ToolCallRequest.Sync(tool_call={"id": "test", "name": "test_tool", "args": {"p": "v"}})
        )
        submitted_request.future.result(timeout=5)  # let the thread run
        mock_tool.invoke.assert_called_once_with({"p": "v", "state": mock_state})

    def test_async_tool_in_non_loop_mode_does_not_read_intermediate_step(self, mock_async_tool, mock_state):
        runner = ToolRunner(
            tools=[mock_async_tool], state=mock_state, max_workers=2, is_loop=False, parallel_timeout=30.0
        )
        with patch("redbox.graph.nodes.runner.runner.execute_mcp_tools"):
            submitted_request = runner.submit_mcp_async(
                mcp_async_call=tr_models.ToolCallRequest.MCPAsync(
                    mcp_server="http://fakemcpurl:8080",
                    creator_type=ChunkCreatorType.datahub,
                    access_token=SensitiveValue(value="fake"),
                    tool_calls=[{"id": "test", "name": "async_tool", "args": {"is_intermediate_step": "True"}}],
                )
            )
        # is_loop=False - intermediate_step flag is ignored, always "False"
        assert submitted_request.metadata["intermediate_step"] == "False"

    def test_raises_tool_execution_error_when_executor_submit_fails(self, tool_runner: ToolRunner):
        tool_runner.executor.submit = Mock(side_effect=Exception("Submission failed"))
        with pytest.raises(
            tool_exceptions.ToolExecutionError,
            match=r"Failed to submit tool 'test_tool' for execution: Submission failed",
        ):
            tool_runner.submit_sync(
                sync_call=tr_models.ToolCallRequest.Sync(tool_call={"id": "test", "name": "test_tool", "args": {}})
            )


class TestToolRunner_SubmitAll:
    @pytest.mark.parametrize(
        "tool_calls,expected_future_count,expected_failures",
        [
            ([], 0, []),
            ([{"id": "test_tool_1", "name": "test_tool", "args": {"p": "v1"}}], 1, []),
            # two calls to the same tool
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {"p": "v1"}},
                    {"id": "test_tool_2", "name": "test_tool", "args": {"p": "v2"}},
                ],
                2,
                [],
            ),
            # two calls to distinct tools — both must be submitted
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {}},
                    {"id": "other_tool_1", "name": "other_tool", "args": {}},
                ],
                2,
                [],
            ),
        ],
        ids=["empty", "single", "same-tool-twice", "two-distinct-tools"],
    )
    def test_submits_futures(self, multi_tool_runner: ToolRunner, tool_calls, expected_future_count, expected_failures):
        requests = multi_tool_runner._submit_all(tool_calls)
        assert len(requests.futures) == expected_future_count
        assert all(isinstance(f, tr_models.SubmittedToolCallRequest) for f in requests.futures)
        for tc, request in zip(tool_calls, requests.futures):
            assert request.result_type == tr_models.FutureResultType.SYNC
            assert request.name == tc.get("name")
            assert request.future_args == tc.get("args")
        assert requests.failures == expected_failures

    def test_unexpected_submission_exception(self, tool_runner: ToolRunner, caplog):
        tool_runner.executor.submit = Mock(side_effect=RuntimeError("kaboom"))
        with caplog.at_level(logging.ERROR):
            requests = tool_runner._submit_all([{"id": "test_tool_1", "name": "test_tool", "args": {"a": "B"}}])
        assert requests.futures == []
        assert requests.failures == [
            tr_models.ToolCallResult.Failure(
                tool_name="test_tool",
                error="Failed to submit tool 'test_tool' for execution: kaboom",
                metadata={"tool_args": {"a": "B"}},
            )
        ]


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
        self, tool_runner: ToolRunner, loop_runner: ToolRunner, response, is_loop, metadata_override, expected_content
    ):
        runner = loop_runner if is_loop else tool_runner
        metadata = metadata_override or _plain_metadata()

        submitted_request = tr_models.SubmittedToolCallRequest(
            name="test_tool",
            result_type=tr_models.FutureResultType.SYNC,
            future=_future_returning(response),
            future_args=metadata,
            metadata={"intermediate_step": metadata.get("intermediate_step", "False")},
        )

        result, failures = runner.execute_request(submitted_request)
        assert isinstance(result, AIMessage)
        assert result.content == expected_content
        assert failures == []

    @pytest.mark.parametrize(
        "future_result_type,tool_name,args,metadata,response",
        [
            (
                tr_models.FutureResultType.SYNC,
                "test_tool",
                {
                    "p": 1,
                },
                {"intermediate_step": "False"},
                "hello from sync tool",
            ),
            (
                tr_models.FutureResultType.MCP_ASYNC,
                "test_mcp_async_tool",
                {
                    "creator_type": ChunkCreatorType.datahub,
                    "mcp_url": "http://localhost:59999/mcp",
                    "tool_calls": [
                        {"id": "test_tool_1", "name": "test_tool", "args": {}},
                        {"id": "other_tool_1", "name": "other_tool", "args": {}},
                    ],
                },
                {"intermediate_step": "False"},
                ({"test_tool_1": "hello from mcp async tool", "other_tool_1": "hello from other mcp async tool"}, []),
            ),
        ],
    )
    def test_logs_receipt_and_non_none_on_success(
        self, tool_runner: ToolRunner, caplog, future_result_type, tool_name, args, metadata, response
    ):
        submitted_request = tr_models.SubmittedToolCallRequest(
            name=tool_name,
            result_type=future_result_type,
            future=_future_returning(response),
            future_args=args,
            metadata=metadata,
        )

        with caplog.at_level(logging.WARNING):
            tool_runner.execute_request(submitted_request)
        assert f"This is what I got from tool '{tool_name}': {response}" in caplog.text
        assert f"{tool_name} response not None" in caplog.text

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
    def test_raises_on_bad_future_or_response(self, tool_runner: ToolRunner, future, exc_type, match):
        metadata = _plain_metadata()
        submitted_request = tr_models.SubmittedToolCallRequest(
            name="test_tool",
            result_type=tr_models.FutureResultType.SYNC,
            future=future,
            future_args=metadata,
            metadata={"intermediate_step": metadata.get("intermediate_step", "False")},
        )

        with pytest.raises(exc_type, match=match):
            tool_runner.execute_request(submitted_request)


class TestToolRunner_Collect:
    @pytest.mark.parametrize(
        "futures_spec,expected,expected_log_fragments",
        [
            # all succeed
            (
                [("r1", "tool1"), ("r2", "tool2")],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="tool1",
                            response=AIMessage("r1"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                        tr_models.ToolCallResult.Success(
                            tool_name="tool2",
                            response=AIMessage("r2"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                    ]
                ),
                ["Completed. Successful: 2, Failed: 0."],
            ),
            # empty input treated as total failure
            (
                [],
                tr_models.Result(),
                ["Every tool execution has failed"],
            ),
            # single exception failure
            (
                [(Exception("boom"), "tool1")],
                tr_models.Result(
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="tool1",
                            error="Tool 'tool1' failed: boom",
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ]
                ),
                ["Every tool execution has failed"],
            ),
            # timeout failure
            (
                [(FuturesTimeoutError(), "slow_tool")],
                tr_models.Result(
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="slow_tool",
                            error="Tool 'slow_tool' timed out after 30.0s",
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ]
                ),
                ["Every tool execution has failed"],
            ),
            # partial: one success, one timeout
            (
                [("ok", "good_tool"), (FuturesTimeoutError(), "slow_tool")],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="good_tool",
                            response=AIMessage("ok"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="slow_tool",
                            error="Tool 'slow_tool' timed out after 30.0s",
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                ),
                ["Completed. Successful: 1, Failed: 1."],
            ),
            # ToolValidationError (e.g. empty response) lands in failed_tools too
            (
                [("ok", "good_tool"), ("", "empty_tool")],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="good_tool",
                            response=AIMessage("ok"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="empty_tool",
                            error="Tool 'empty_tool' returned empty or whitespace-only response",
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                ),
                ["Completed. Successful: 1, Failed: 1."],
            ),
            # multiple distinct tools all succeed — responses preserve insertion order
            (
                [("alpha", "tool_a"), ("beta", "tool_b"), ("gamma", "tool_c")],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="tool_a",
                            response=AIMessage("alpha"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                        tr_models.ToolCallResult.Success(
                            tool_name="tool_b",
                            response=AIMessage("beta"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                        tr_models.ToolCallResult.Success(
                            tool_name="tool_c",
                            response=AIMessage("gamma"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                    ]
                ),
                ["Completed. Successful: 3, Failed: 0."],
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
        tool_runner: ToolRunner,
        caplog,
        futures_spec,
        expected,
        expected_log_fragments,
    ):
        submitted_requests = []

        for v, name in futures_spec:
            metadata = _plain_metadata(name)
            submitted_requests.append(
                tr_models.SubmittedToolCallRequest(
                    name=name,
                    result_type=tr_models.FutureResultType.SYNC,
                    future=_future_raising(v) if isinstance(v, Exception) else _future_returning(v),
                    future_args={},
                    metadata={"intermediate_step": metadata.get("intermediate_step")},
                )
            )
        submitted = tr_models.SubmittedRunRequest(futures=submitted_requests, failures=[])

        with caplog.at_level(logging.WARNING):
            result = tool_runner._collect(submitted)

        assert isinstance(result, tr_models.Result)
        assert result.failures == expected.failures
        assert result.results == expected.results
        assert result == expected
        for fragment in expected_log_fragments:
            assert fragment in caplog.text


class TestToolRunner_Run:
    @pytest.mark.parametrize(
        "tool_calls,expected",
        [
            # empty - no responses
            ([], tr_models.Result()),
            # single call to test_tool - its specific return value
            (
                [{"id": "test_tool_1", "name": "test_tool", "args": {}}],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ]
                ),
            ),
            # single call to other_tool - its specific return value
            (
                [{"id": "other_tool_1", "name": "other_tool", "args": {}}],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="other_tool",
                            response=AIMessage("other result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ]
                ),
            ),
            # two calls to the same tool with different args - same return value twice
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {"p": "v1"}},
                    {"id": "test_tool_2", "name": "test_tool", "args": {"p": "v2"}},
                ],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {"p": "v1"}},
                        ),
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {"p": "v2"}},
                        ),
                    ]
                ),
            ),
            # one call to each distinct tool - each returns its own value
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {}},
                    {"id": "other_tool_1", "name": "other_tool", "args": {}},
                ],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                        tr_models.ToolCallResult.Success(
                            tool_name="other_tool",
                            response=AIMessage("other result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        ),
                    ]
                ),
            ),
            # unknown tool alongside valid call - one response, ghost skipped at submit
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {}},
                    {"id": "ghost_tool_1", "name": "ghost_tool", "args": {"fake": "fakeval"}},
                ],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="ghost_tool",
                            error="Tool 'ghost_tool' not found. Available tools: test_tool, other_tool",
                            metadata={"tool_args": {"fake": "fakeval"}},
                        )
                    ],
                ),
            ),
            # one tool succeeds, other_tool's future raises — lands in failed_tools
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {}},
                    {"id": "other_tool_1", "name": "other_tool", "args": {}},
                ],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="other_tool",
                            error="Tool 'other_tool' failed: other_tool blew up",
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                ),
            ),
            # one tool succeeds, other_tool's future raises — lands in failed_tools with args
            (
                [
                    {"id": "test_tool_1", "name": "test_tool", "args": {}},
                    {"id": "other_tool_1", "name": "other_tool", "args": {"veg": "carrot"}},
                ],
                tr_models.Result(
                    results=[
                        tr_models.ToolCallResult.Success(
                            tool_name="test_tool",
                            response=AIMessage("test result"),
                            metadata={"intermediate_step": "False", "tool_args": {}},
                        )
                    ],
                    failures=[
                        tr_models.ToolCallResult.Failure(
                            tool_name="other_tool",
                            error="Tool 'other_tool' failed: other_tool blew up",
                            metadata={"intermediate_step": "False", "tool_args": {"veg": "carrot"}},
                        )
                    ],
                ),
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
            "one-succeeds-one-fails-with-args",
        ],
    )
    def test_run_returns_correct_result(self, multi_tool_runner: ToolRunner, mock_tool_b, tool_calls, expected):
        # For the "one-succeeds-one-fails" case mock_tool_b.invoke raises.
        if expected.failures:
            mock_tool_b.invoke.side_effect = Exception("other_tool blew up")
        result = multi_tool_runner.run(tool_calls)
        assert isinstance(result, tr_models.Result)
        assert result.results == expected.results
        assert result.failures == expected.failures
        assert result == expected

    @pytest.mark.parametrize(
        "submit_all_side_effect,expected_exc",
        [
            (None, None),
            (RuntimeError("crash"), RuntimeError),
        ],
        ids=["success", "exception"],
    )
    def test_executor_always_shuts_down(self, tool_runner: ToolRunner, submit_all_side_effect, expected_exc):
        with patch.object(tool_runner.executor, "shutdown") as mock_shutdown:
            if submit_all_side_effect:
                with patch.object(tool_runner, "_submit_all", side_effect=submit_all_side_effect):
                    with pytest.raises(expected_exc):
                        tool_runner.run([{"id": "test_tool_1", "name": "test_tool", "args": {}}])
            else:
                tool_runner.run([{"id": "test_tool_1", "name": "test_tool", "args": {}}])
        mock_shutdown.assert_called_once_with(wait=True)

    def test_delegates_to_submit_all_and_collect(self, tool_runner: ToolRunner):
        mock_future = Mock(spec=Future)
        stub_request = tr_models.SubmittedRunRequest(
            futures=[
                tr_models.SubmittedToolCallRequest(
                    name="test_tool",
                    result_type=tr_models.FutureResultType.SYNC,
                    future=mock_future,
                    future_args={},
                    metadata={"intermediate_step": "False"},
                )
            ],
            failures=[],
        )
        stub_result = tr_models.Result(
            results=[
                tr_models.ToolCallResult.Success(
                    tool_name="test_tool",
                    response=AIMessage("r"),
                    metadata={"intermediate_step": "False", "tool_args": {}},
                )
            ]
        )

        with patch.object(tool_runner, "_submit_all", return_value=stub_request) as ms:
            with patch.object(tool_runner, "_collect", return_value=stub_result) as mc:
                result = tool_runner.run([{"name": "test_tool", "args": {}}])

        ms.assert_called_once_with(tool_calls=[{"name": "test_tool", "args": {}}])
        mc.assert_called_once_with(submitted_request=stub_request)
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
        result = runner.run([{"id": "slow_tool_1", "name": "slow_tool", "args": {}} for _ in range(3)])
        elapsed = time.time() - start

        assert elapsed < 0.25, f"Expected ~0.1s with parallelism, got {elapsed:.2f}s"
        assert result == tr_models.Result(
            results=[
                tr_models.ToolCallResult.Success(
                    tool_name="slow_tool",
                    response=AIMessage("done"),
                    metadata={"intermediate_step": "False", "tool_args": {}},
                )
            ]
            * 3
        )

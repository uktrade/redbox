from concurrent.futures import TimeoutError
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ConnectError
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolCall
from langgraph.constants import Send
from pytest_mock import MockerFixture

from redbox.graph.nodes.sends import (
    build_document_chunk_send,
    build_document_group_send,
    build_tool_send,
    no_dependencies,
    run_tools_parallel,
    wrap_async_tool,
)
from redbox.graph.nodes.tools import build_govuk_search_tool, build_search_wikipedia_tool
from redbox.models.chain import DocumentState, RedboxQuery, RedboxState, TaskStatus, configure_agent_task_plan
from tests.conftest import fake_state


def test_build_document_group_send():
    target = "my-target"
    request = RedboxQuery(question="what colour is the sky?", user_uuid=uuid4(), chat_history=[])
    documents = DocumentState(
        groups={
            uuid4(): {
                uuid4(): Document(page_content="Hello, world!"),
                uuid4(): Document(page_content="Goodbye, world!"),
            }
        }
    )

    document_group_send = build_document_group_send("my-target")
    state = RedboxState(
        request=request,
        documents=documents,
        text=None,
        route_name=None,
    )
    actual = document_group_send(state)
    expected = [Send(node=target, arg=state)]
    assert expected == actual


def test_build_document_chunk_send():
    target = "my-target"
    request = RedboxQuery(question="what colour is the sky?", user_uuid=uuid4(), chat_history=[])

    uuid_1 = uuid4()
    doc_1 = Document(page_content="Hello, world!")
    uuid_2 = uuid4()
    doc_2 = Document(page_content="Goodbye, world!")

    document_chunk_send = build_document_chunk_send("my-target")
    state = RedboxState(
        request=request,
        documents=DocumentState(groups={uuid_1: {uuid_1: doc_1}, uuid_2: {uuid_2: doc_2}}),
        text=None,
        route_name=None,
    )
    actual = document_chunk_send(state)
    expected = [
        Send(
            node=target,
            arg=RedboxState(
                request=request,
                documents=DocumentState(groups={uuid_1: {uuid_1: doc_1}}),
                text=None,
                route_name=None,
            ),
        ),
        Send(
            node=target,
            arg=RedboxState(
                request=request,
                documents=DocumentState(groups={uuid_2: {uuid_2: doc_2}}),
                text=None,
                route_name=None,
            ),
        ),
    ]
    assert expected == actual


def test_build_tool_send():
    target = "my-target"
    request = RedboxQuery(question="what colour is the sky?", user_uuid=uuid4(), chat_history=[])

    tool_call_1 = [ToolCall(name="foo", args={"a": 1, "b": 2}, id="123")]
    tool_call_2 = [ToolCall(name="bar", args={"x": 10, "y": 20}, id="456")]

    tool_send = build_tool_send("my-target")
    actual = tool_send(
        RedboxState(
            request=request,
            messages=[AIMessage(content="", tool_calls=tool_call_1 + tool_call_2)],
            route_name=None,
        ),
    )
    expected = [
        Send(
            node=target,
            arg=RedboxState(
                request=request,
                messages=[AIMessage(content="", tool_calls=tool_call_1)],
                route_name=None,
            ),
        ),
        Send(
            node=target,
            arg=RedboxState(
                request=request,
                messages=[AIMessage(content="", tool_calls=tool_call_2)],
                route_name=None,
            ),
        ),
    ]
    assert expected == actual


class TestRunToolsParallel:
    def test_no_tool_call(self, fake_state):
        ai_msg = AIMessage(content="no tool call", tool_calls=[])

        response = run_tools_parallel(ai_msg=ai_msg, tools=[], state=fake_state)

        assert isinstance(response, str)
        assert response == "no tool call"

    def test_no_tool_found(self, fake_state):
        search_wikipedia = build_search_wikipedia_tool()
        ai_msg = AIMessage(
            content="I am calling a tool",
            tool_calls=[{"name": "_test_tool", "args": {"query": "fake query"}, "id": "1"}],
        )

        response = run_tools_parallel(ai_msg=ai_msg, tools=[search_wikipedia], state=fake_state)

        assert response is None

    def test_tool_found(self, mocker: MockerFixture):
        # mock tool
        mock_tool = mocker.patch("langchain_community.utilities.WikipediaAPIWrapper.load")
        mock_tool.return_value = [
            Document(
                metadata={"title": "fake title", "summary": "fake summary", "source": "https://fake.com"},
                page_content="Fake content",
            )
        ]

        search_wikipedia = build_search_wikipedia_tool()
        ai_msg = AIMessage(
            content="I am calling a tool",
            tool_calls=[{"name": "_search_wikipedia", "args": {"query": "fake query"}, "id": "1"}],
        )

        response = run_tools_parallel(ai_msg=ai_msg, tools=[search_wikipedia], state=fake_state)
        assert isinstance(response, list)
        assert len(response[0].content) > 0

    def test_tool_time_out(self, mocker: MockerFixture):
        mock_tool = mocker.patch("langchain_community.utilities.WikipediaAPIWrapper.load")
        mock_tool.side_effect = TimeoutError("Tool time out")

        search_wikipedia = build_search_wikipedia_tool()
        ai_msg = AIMessage(
            content="I am calling a tool",
            tool_calls=[{"name": "_search_wikipedia", "args": {"query": "fake query"}, "id": "1"}],
        )

        response = run_tools_parallel(ai_msg=ai_msg, tools=[search_wikipedia], state=fake_state)

        assert response is None

    @pytest.mark.parametrize("side_effect", [(TimeoutError("Thread time out")), (Exception("Thread error"))])
    def test_threadpool_time_out(self, side_effect, mocker: MockerFixture):
        mock_tool = mocker.patch("concurrent.futures.ThreadPoolExecutor.submit")
        mock_tool.side_effect = side_effect

        search_wikipedia = build_search_wikipedia_tool()
        ai_msg = AIMessage(
            content="I am calling a tool",
            tool_calls=[
                {"name": "_search_wikipedia", "args": {"query": "fake query"}, "id": "1"},
                {"name": "_search_gov", "args": {"query": "another fake query"}, "id": "2"},
            ],
        )

        search_gov = build_govuk_search_tool()

        response = run_tools_parallel(ai_msg=ai_msg, tools=[search_wikipedia, search_gov], state=fake_state)

        assert response is None

    @pytest.mark.parametrize("side_effect", [(TimeoutError("Thread time out"))])
    def test_not_all_time_out(self, side_effect, mocker: MockerFixture):
        """
        Test where one tool timeout and the other doesn't.
        Redbox should not return an error. It should return the response from the tool that did not timeout.
        """

        mock_tool = mocker.patch("langchain_community.utilities.WikipediaAPIWrapper.load")
        mock_tool.return_value = [
            Document(
                metadata={"title": "fake title", "summary": "fake summary", "source": "https://fake.com"},
                page_content="Fake content",
            )
        ]

        search_wikipedia_complete = build_search_wikipedia_tool()

        mock_tool = mocker.patch("requests.get")
        mock_tool.side_effect = TimeoutError("Tool time out")

        search_govuk_timeout = build_govuk_search_tool()

        ai_msg = AIMessage(
            content="I am calling a tool",
            tool_calls=[
                {"name": "_search_wikipedia", "args": {"query": "fake query"}, "id": "1"},
                {"name": "_search_gov", "args": {"query": "another fake query"}, "id": "1"},
            ],
        )

        responses = run_tools_parallel(
            ai_msg=ai_msg, tools=[search_wikipedia_complete, search_govuk_timeout], state=fake_state
        )
        assert isinstance(responses, list)
        assert len(responses[0].content) > 0
        assert len(responses) == 1


@pytest.mark.parametrize("dependencies, expected", [([], True), (["task0"], True), (["task1", "task2"], False)])
def test_no_dependencies(dependencies, expected):
    agent = "Internal_Retrieval_Agent"
    agent_task, multi_agent_plan = configure_agent_task_plan({agent: agent})
    tasks = [
        agent_task(id="task0", task="Fake Task", expected_output="Fake output", status=TaskStatus.COMPLETED),
        agent_task(
            id="task1",
            task="Fake Task",
            expected_output="Fake output",
            dependencies=["task0"],
            status=TaskStatus.PENDING,
        ),
        agent_task(
            id="task2",
            task="Fake Task",
            expected_output="Fake output",
            dependencies=["task0"],
            status=TaskStatus.PENDING,
        ),
    ]

    plan = multi_agent_plan(tasks=tasks)

    # test case
    actual = no_dependencies(dependencies, plan=plan)

    assert actual == expected


class TestRunToolsParallelAsync:
    def _patch_mcp_env(self, mock_load_tools, mock_http_client, mock_session_class, tools):
        """Patch MCP networking to allow wrap_async_tool to succeed."""
        # streamablehttp_client mock
        mock_read, mock_write = AsyncMock(), AsyncMock()
        mock_http_cm = AsyncMock()
        mock_http_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_http_cm.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.return_value = mock_http_cm

        # ClientSession mock
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # load_mcp_tools returns the tool
        mock_load_tools.return_value = tools

    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_returns_expected_response(
        self, mock_load_tools, mock_http_client, mock_session_class, fake_state, fake_mcp_tool
    ):
        expected_result = '{"status": "success"}'
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name"]}
        tool = fake_mcp_tool("company_tool", expected_result, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="call async tool",
            tool_calls=[{"name": "company_tool", "args": {"company_name": "BMW"}, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state)

        assert isinstance(responses, list)
        assert len(responses) == 1
        assert responses[0].content == expected_result
        tool.ainvoke.assert_awaited_once_with({"company_name": "BMW"})

    @pytest.mark.parametrize(
        "required_keys,expected_ainvoke_args",
        [
            (["is_intermediate_step"], {"is_intermediate_step": "False"}),
            ([], {}),
        ],
    )
    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_with_loop_agent(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_state,
        fake_mcp_tool,
        required_keys,
        expected_ainvoke_args,
    ):
        expected_result = '{"total": 1}'
        args_schema = {"required": required_keys}
        tool = fake_mcp_tool("loop_tool", expected_result, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="loop call",
            tool_calls=[{"name": "loop_tool", "args": {"is_intermediate_step": "False"}, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state, is_loop=True)
        assert isinstance(responses, list)

        transformed = responses[0].content
        assert isinstance(transformed, list)
        assert transformed[0] == expected_result
        assert transformed[1] == "pass"
        assert transformed[2] == "False"

        # Ensure ainvoke got the correct args based on whether 'is_intermediate_step' is required
        tool.ainvoke.assert_awaited_once_with(expected_ainvoke_args)

    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_with_non_loop_agent(
        self, mock_load_tools, mock_http_client, mock_session_class, fake_state, fake_mcp_tool
    ):
        expected_result = "plain async response"
        args_schema = {"required": []}
        tool = fake_mcp_tool("non_loop_tool", expected_result, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="non-loop call",
            tool_calls=[{"name": "non_loop_tool", "args": {"foo": "bar"}, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state, is_loop=False)
        assert isinstance(responses, list)
        assert responses[0].content == expected_result
        tool.ainvoke.assert_awaited_once_with({"foo": "bar"})

    @pytest.mark.parametrize(
        "exception",
        [TimeoutError("tool timed out"), ValueError("invalid value"), Exception("unknown error")],
    )
    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_failures_return_none(
        self, mock_load_tools, mock_http_client, mock_session_class, exception, fake_state, fake_mcp_tool_failing
    ):
        tool = fake_mcp_tool_failing("failing_tool", exception)
        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="call failing tool",
            tool_calls=[{"name": "failing_tool", "args": {"foo": "bar"}, "id": "1"}],
        )

        response = run_tools_parallel(ai_msg, tools=[tool], state=fake_state)
        assert response is None
        tool.ainvoke.assert_awaited_once_with({"foo": "bar"})

    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_not_found_returns_none(
        self, mock_load_tools, mock_http_client, mock_session_class, fake_state, fake_mcp_tool
    ):
        args_schema = {"required": []}
        tool = fake_mcp_tool("real_tool", "some response", args_schema=args_schema)
        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="call missing tool",
            tool_calls=[{"name": "missing_tool", "args": {"foo": "bar"}, "id": "1"}],
        )

        response = run_tools_parallel(ai_msg, tools=[tool], state=fake_state)
        assert response is None


class TestWrapAsyncTool:
    def _patch_mcp_env(self, mock_load_tools, mock_http_client, mock_session_class, tools):
        """Patch MCP networking to allow wrap_async_tool to succeed."""
        # streamablehttp_client mock
        mock_read, mock_write = AsyncMock(), AsyncMock()
        mock_http_cm = AsyncMock()
        mock_http_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
        mock_http_cm.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.return_value = mock_http_cm

        # ClientSession mock
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # load_mcp_tools returns the tool
        mock_load_tools.return_value = tools

        return mock_session

    @pytest.mark.parametrize(
        "url,expected_exceptions",
        [
            ("http://fake-mcp-url", (ConnectError)),  # non-existent hostname
            ("http://127.0.0.1:59999", (ConnectError)),  # unused localhost port
        ],
    )
    def test_connection_failure(self, fake_mcp_tool, url, expected_exceptions):
        """Test wrap_async_tool fails when MCP server cannot be reached."""
        tool = fake_mcp_tool("dummy_tool", return_value=None)
        tool.metadata["url"] = url

        wrapped = wrap_async_tool(tool, tool.name)
        args = {"foo": "bar"}

        with pytest.raises(ExceptionGroup) as exc_info:
            wrapped(args)

        # All inner exceptions should match the expected types
        exceptions = exc_info.value.exceptions
        assert all(isinstance(e, expected_exceptions) for e in exceptions)

    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools")
    def test_returns_expected_results(self, mock_load_tools, mock_http_client, mock_session_class, fake_mcp_tool):
        """Test that wrap_async_tool correctly returns results from async tool invocation"""
        # create expected result that we want to test for
        expected_result = {
            "status": "success",
            "data": {"company_name": "BMW", "country": "Germany", "sector": "Automotive"},
        }

        # Mock tool with metadata
        tool_name = "company_tool"
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name"]}
        tool = fake_mcp_tool(tool_name, return_value=expected_result, args_schema=args_schema)

        # mock session with patched mcp setup
        mock_session = self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        # create the wrapped function
        wrapped_func = wrap_async_tool(tool, tool_name)

        # rest invocation with sample args
        test_args = {"company_name": "BMW"}
        result = wrapped_func(test_args)

        # verify correct interactions
        mock_http_client.assert_called_once_with(tool.metadata["url"])
        mock_session.initialize.assert_called_once()
        mock_load_tools.assert_called_once_with(mock_session)
        tool.ainvoke.assert_called_once_with(test_args)

        # assert the result matches our expected output
        assert result == expected_result

    @patch("redbox.graph.nodes.sends.ClientSession")
    @patch("redbox.graph.nodes.sends.streamablehttp_client")
    @patch("redbox.graph.nodes.sends.load_mcp_tools", new_callable=AsyncMock)
    def test_tool_not_found(self, mock_load_tools, mock_http_client, mock_session_class, fake_mcp_tool):
        """Test wrap_async_tool raises ValueError when the requested tool is not in the MCP tool list."""

        tool = fake_mcp_tool("dummy_tool", return_value=None)
        wrapped_func = wrap_async_tool(tool, "missing_tool")

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        # assert ValueError is raised
        with pytest.raises(ValueError, match="tool with name 'missing_tool' not found"):
            wrapped_func({"foo": "bar"})

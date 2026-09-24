from concurrent.futures import TimeoutError
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolCall
from langgraph.types import Send
from pytest_mock import MockerFixture
from tests.conftest import fake_state
from tests.retriever.data import MCP_TOOL_RESULTS

from redbox.api.format import MCPResponseMetadata
from redbox.graph.nodes.sends import (
    build_document_chunk_send,
    build_document_group_send,
    build_tool_send,
    no_dependencies,
    run_tools_parallel,
)
from redbox.graph.nodes.tools import build_govuk_search_tool, build_search_wikipedia_tool
from redbox.models.chain import DocumentState, RedboxQuery, RedboxState, TaskStatus, configure_agent_task_plan


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
        """Patch MCP networking to allow execute_mcp_tools to succeed."""
        # streamablehttp_client mock
        mock_read, mock_write = AsyncMock(), AsyncMock()
        mock_http_cm = AsyncMock()
        mock_http_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, None))
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

    @pytest.mark.parametrize("expected_tool_result, expected_parsed_result", MCP_TOOL_RESULTS)
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamablehttp_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_returns_expected_response(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_state,
        fake_mcp_tool,
        expected_tool_result,
        expected_parsed_result,
    ):
        expected_tool_content, _ = expected_tool_result

        tool_name = "company_tool"
        args_schema = {"company_name": {"type": "string"}, "required": ["company_name"]}
        args = {"company_name": "BMW"}

        tool = fake_mcp_tool(tool_name, expected_tool_content, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="call async tool",
            tool_calls=[{"name": tool_name, "args": args, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state)

        assert isinstance(responses, list)
        assert len(responses) == 1
        assert responses[0].content == expected_parsed_result
        tool.ainvoke.assert_awaited_once_with({"company_name": "BMW"})

    @pytest.mark.parametrize(
        "required_keys,expected_ainvoke_args",
        [
            (
                ["company_name"],
                {"company_name": "BMW", "is_intermediate_step": "False"},
            ),
            (["company_name"], {"company_name": "BMW"}),
        ],
    )
    @pytest.mark.parametrize("expected_tool_result, expected_parsed_result", MCP_TOOL_RESULTS)
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamablehttp_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_with_loop_agent(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_state,
        fake_mcp_tool,
        required_keys,
        expected_ainvoke_args,
        expected_tool_result: tuple[str, MCPResponseMetadata],
        expected_parsed_result: str,
    ):
        expected_tool_content, expected_tool_metadata = expected_tool_result

        tool_name = "company_tool"
        args_schema = {"required": required_keys}

        tool = fake_mcp_tool(tool_name, expected_tool_content, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="loop call",
            tool_calls=[{"name": tool_name, "args": expected_ainvoke_args, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state, is_loop=True)
        assert isinstance(responses, list)

        transformed = responses[0].content
        assert isinstance(transformed, list)

        assert transformed[0] == expected_parsed_result
        expected_status = "pass" if expected_parsed_result != "" else "fail"

        assert transformed[1] == expected_status
        assert transformed[2] == "False"

        if expected_tool_metadata.user_feedback.required:
            assert len(transformed) == 4
            assert transformed[3] == expected_tool_metadata.user_feedback.reason
        else:
            assert len(transformed) == 3

        # Ensure ainvoke got the correct args based on whether 'is_intermediate_step' is required
        if "is_intermediate_step" not in required_keys and "is_intermediate_step" in expected_ainvoke_args.keys():
            expected_ainvoke_args.pop("is_intermediate_step")
        tool.ainvoke.assert_awaited_once_with(expected_ainvoke_args)

    @pytest.mark.parametrize("expected_tool_result, expected_parsed_result", MCP_TOOL_RESULTS)
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamablehttp_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
    def test_async_tool_with_non_loop_agent(
        self,
        mock_load_tools,
        mock_http_client,
        mock_session_class,
        fake_state,
        fake_mcp_tool,
        expected_tool_result,
        expected_parsed_result,
    ):
        expected_tool_content, _ = expected_tool_result

        tool_name = "company_tool"
        args_schema = {"required": []}
        args = {"company_name": "BMW"}
        tool = fake_mcp_tool(tool_name, expected_tool_content, args_schema=args_schema)

        self._patch_mcp_env(mock_load_tools, mock_http_client, mock_session_class, [tool])

        ai_msg = AIMessage(
            content="non-loop call",
            tool_calls=[{"name": tool_name, "args": args, "id": "1"}],
        )

        responses = run_tools_parallel(ai_msg, tools=[tool], state=fake_state, is_loop=False)
        assert isinstance(responses, list)
        assert responses[0].content == expected_parsed_result
        tool.ainvoke.assert_awaited_once_with(args)

    @pytest.mark.parametrize(
        "exception",
        [TimeoutError("tool timed out"), ValueError("invalid value"), Exception("unknown error")],
    )
    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamablehttp_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
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

    @patch("redbox.graph.nodes.runner.wrap_async.ClientSession")
    @patch("redbox.graph.nodes.runner.wrap_async.streamablehttp_client")
    @patch("redbox.graph.nodes.runner.wrap_async.load_mcp_tools", new_callable=AsyncMock)
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

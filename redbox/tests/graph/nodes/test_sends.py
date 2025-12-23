from concurrent.futures import TimeoutError
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolCall
from langgraph.constants import Send
from pytest_mock import MockerFixture

from redbox.graph.nodes.sends import (
    build_document_chunk_send,
    build_document_group_send,
    build_tool_send,
    run_tools_parallel,
)
from redbox.graph.nodes.tools import build_search_wikipedia_tool, build_govuk_search_tool
from redbox.models.chain import DocumentState, RedboxQuery, RedboxState
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
        print("my-responses")
        print(responses)
        assert isinstance(responses, list)
        assert len(responses[0].content) > 0
        assert len(responses) == 1

import logging
import os
from unittest.mock import AsyncMock, patch

import pytest
from channels.testing import WebsocketCommunicator
from langchain_core.documents import Document
from tests.consumers_helpers import CannedGraphLLM, Token

from redbox.models.graph import FINAL_RESPONSE_TAG, ROUTE_NAME_TAG
from redbox_app.redbox_core.consumers import ChatConsumer
from redbox_app.redbox_core.models import File

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

AGENTIC_ROUTE = "search/agentic"


def _text_event(text: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "tags": [FINAL_RESPONSE_TAG],
        "data": {"chunk": Token(content=text)},
    }


def _route_event(route: str) -> dict:
    return {
        "event": "on_chain_end",
        "tags": [ROUTE_NAME_TAG],
        "data": {"output": {"route_name": route}},
    }


def _source_event(file: File) -> dict:
    return {
        "event": "on_custom_event",
        "tags": ["on_source_report"],
        "data": [Document(metadata={"uri": file.unique_name}, page_content="Test document content.")],
    }


async def _send_and_collect(alice, agents_list, mocked_graph, message: str, n_responses: int) -> list[dict]:
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_graph):
            await communicator.send_json_to({"message": message})
            responses = [await communicator.receive_json_from(timeout=5) for _ in range(n_responses)]
            await communicator.disconnect()

    return responses


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_summarisation_agent_returns_non_empty_response(agents_list, alice, uploaded_file: File):
    mocked_graph = CannedGraphLLM(
        responses=[
            _text_event("Here is an summary of the document."),
            _route_event(AGENTIC_ROUTE),
            _source_event(uploaded_file),
        ]
    )
    responses = await _send_and_collect(
        alice, agents_list, mocked_graph, "Please summarise this document.", n_responses=4
    )

    # assertions
    assert responses[0]["type"] == "session-id"
    assert responses[1]["type"] == "text"
    assert responses[1]["data"], "Summarisation Agent returned blank response"
    assert responses[2]["type"] == "route"
    assert responses[2]["data"] == AGENTIC_ROUTE
    assert responses[3]["type"] == "source"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_internal_retrieval_agent_returns_non_empty_response(agents_list, alice, uploaded_file: File):
    mocked_graph = CannedGraphLLM(
        responses=[
            _text_event("Based on the documents you have, this is the relevant information for this topic."),
            _route_event(AGENTIC_ROUTE),
            _source_event(uploaded_file),
        ]
    )
    responses = await _send_and_collect(
        alice, agents_list, mocked_graph, "From my documents, what information is there on this topic", n_responses=4
    )

    # assertions
    assert responses[0]["type"] == "session-id"
    assert responses[1]["type"] == "text"
    assert responses[1]["data"], "Internal_Retrieval_Agent returned blank response"
    assert responses[2]["type"] == "route"
    assert responses[2]["data"] == AGENTIC_ROUTE
    assert responses[3]["type"] == "source"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_web_search_agent_returns_non_empty_response(agents_list, alice):
    mocked_graph = CannedGraphLLM(
        responses=[
            _text_event("Based on a web search, the weather in London is this."),
            _route_event(AGENTIC_ROUTE),
        ]
    )
    responses = await _send_and_collect(
        alice, agents_list, mocked_graph, "What is the weather in London today?", n_responses=3
    )

    # assertions
    assert responses[0]["type"] == "session-id"
    assert responses[1]["type"] == "text"
    assert responses[1]["data"], "Web_Search_Agent returned blank response"
    assert responses[2]["type"] == "route"
    assert responses[2]["data"] == AGENTIC_ROUTE


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_external_retrieval_agent_returns_non_empty_response(agents_list, alice):
    mocked_graph = CannedGraphLLM(
        responses=[
            _text_event("Based on external sources, this is some information on DBT."),
            _route_event(AGENTIC_ROUTE),
        ]
    )
    responses = await _send_and_collect(
        alice, agents_list, mocked_graph, "From external sources. find information about DBT", n_responses=3
    )

    # assertions
    assert responses[0]["type"] == "session-id"
    assert responses[1]["type"] == "text"
    assert responses[1]["data"], "External_Retrieval_Agent returned blank response"
    assert responses[2]["type"] == "route"
    assert responses[2]["data"] == AGENTIC_ROUTE

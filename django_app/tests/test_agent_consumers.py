import logging
import os
from unittest.mock import AsyncMock, patch

from channels.testing import WebsocketCommunicator
from langchain_core.documents import Document
from tests.consumers_helpers import Token

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
        "tags": "on_source_report",
        "data": [Document(metadata={"uri": file.unique_name}, page_content="Test document content.")],
    }


async def _send_and_collect(alice, agents_list, mocked_graph, message: str, n_reponses: int) -> list[dict]:
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_graph):
            await communicator.send_json_to({"message": message})
            responses = [await communicator.receive_json_from(timeout=5) for _ in range(n_reponses)]
            await communicator.disconnect()

    return responses

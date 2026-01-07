import json
import logging
import os
from asyncio import CancelledError
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.db.models import Model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from redbox import Redbox
from websockets import WebSocketClientProtocol
from websockets.legacy.client import Connect

from redbox.models.chain import LLMCallMetadata, RedboxQuery, RequestMetadata
from redbox.models.graph import FINAL_RESPONSE_TAG, ROUTE_NAME_TAG, RedboxActivityEvent
from redbox.models.prompts import CHAT_MAP_QUESTION_PROMPT

from redbox.models.chain import RedboxState


@pytest.mark.asyncio
async def test_run_stream_events_with_retry():
    # setup
    redbox = Redbox()

    x = redbox.run


    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == ["Hello Hal."]
        assert await get_chat_message_text(alice, ChatMessage.Role.ai) == ["Good afternoon, Mr. Amor."]

        await refresh_from_db(uploaded_file)

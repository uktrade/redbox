import asyncio
import json
import logging
import os
import uuid
from asyncio import CancelledError
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models import Model
from django.utils import timezone
from langchain_core.documents import Document
from tests.consumers_helpers import CannedGraphLLM, Token
from websockets.asyncio.client import ClientConnection

from redbox.models.chain import AISettings as PydanticAISettings
from redbox.models.chain import Citation as AICitation
from redbox.models.chain import LLMCallMetadata, RedboxQuery, RequestMetadata, Source
from redbox.models.graph import FINAL_RESPONSE_TAG, ROUTE_NAME_TAG, RedboxActivityEvent
from redbox.models.prompts import CHAT_MAP_QUESTION_PROMPT
from redbox.models.settings import ChatLLMBackend as PydanticChatLLMBackend
from redbox_app.redbox_core import consumers as consumers_module
from redbox_app.redbox_core import error_messages
from redbox_app.redbox_core.consumers import ChatConsumer, resolve_citable_file
from redbox_app.redbox_core.models import ActivityEvent, Chat, ChatMessage, ChatMessageTokenUse, Citation, File
from redbox_app.redbox_core.models import Chat as ChatModel
from redbox_app.redbox_core.models import ChatMessage as ChatMessageModel

User = get_user_model()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

GOV_UK_URL = "https://www.gov.uk/guidance/trade-tariff"
DATAHUB_URL = "https://datahub.trade.gov.uk/companies/0f5b4d1e/overview"


def make_file(user: User, name: str) -> File:
    """Create a complete uploaded file belonging to the specified User."""
    return File.objects.create(
        user=user,
        original_file=SimpleUploadedFile(name, b"Lorem Ipsum."),
        status=File.Status.complete,
    )


def make_citation_consumer(
    chat_message: ChatMessage,
    citable_files: list[File],
) -> ChatConsumer:
    """Create a chat consumer configured with a message and permitted citation files."""
    consumer = ChatConsumer()
    consumer._file_cache = {}  # noqa: SLF001
    consumer._citable_files = citable_files  # noqa: SLF001
    consumer.chat_message = chat_message
    consumer.send_to_client = AsyncMock()
    return consumer


@database_sync_to_async
def get_token_use_model(use_type: str) -> str:
    return ChatMessageTokenUse.objects.filter(use_type=use_type).latest("created_at").model_name


@database_sync_to_async
def get_activity_model() -> str:
    return ActivityEvent.objects.latest("created_at").message


@database_sync_to_async
def get_token_use_count(use_type: str) -> int:
    return ChatMessageTokenUse.objects.filter(use_type=use_type).latest("created_at").token_count


async def assert_mocked_connect_response(communicator: WebsocketCommunicator, message: str = "Hello Hal."):
    await communicator.send_json_to({"message": message})
    response1 = await communicator.receive_json_from(timeout=5)
    response2 = await communicator.receive_json_from(timeout=5)
    response3 = await communicator.receive_json_from(timeout=5)
    response4 = await communicator.receive_json_from(timeout=5)
    response5 = await communicator.receive_json_from(timeout=5)
    response6 = await communicator.receive_json_from(timeout=5)
    response7 = await communicator.receive_json_from(timeout=5)
    response8 = await communicator.receive_json_from(timeout=5)

    assert response1["type"] == "message_created"
    assert response1["data"]["chat_message_id"]
    assert response1["data"]["chat_message_role"] == "user"
    assert response1["data"]["html"]

    assert response2["type"] == "message_created"
    assert response2["data"]["chat_message_id"]
    assert response2["data"]["chat_message_role"] == "ai"
    assert response2["data"]["html"]

    assert response3["type"] == "session-id"

    assert response4["type"] == "message_update"
    assert response4["data"]["chat_message_id"]
    assert response4["data"]["sr_text"] == "Good afternoon, "
    assert response4["data"]["html"]

    assert response5["type"] == "message_update"
    assert response5["data"]["chat_message_id"]
    assert response5["data"]["sr_text"] == "Mr. "
    assert response5["data"]["html"]

    assert response6["type"] == "message_update"
    assert response6["data"]["chat_message_id"]
    assert response6["data"]["sr_text"] == "Amor."
    assert response6["data"]["html"]

    assert response7["type"] == "route"
    assert response7["data"] == "gratitude"

    assert response8["type"] == "message_complete"
    assert response8["data"]["chat_message_id"]
    assert response8["data"]["title"] == message
    assert response8["data"]["session_id"]
    assert response8["data"]["html"]

    # Close
    await communicator.disconnect()


def test_resolve_citable_file_only_matches_permitted_files():
    """Resolve full storage keys and filenames only against permitted files."""
    report = File(original_file="alice@example.com/report_20250101.pdf", original_file_name="report.pdf")
    notes = File(original_file="alice@example.com/notes_20250101.pdf", original_file_name="notes.pdf")
    files = [report, notes]

    assert resolve_citable_file("alice@example.com/report_20250101.pdf", files) is report
    assert resolve_citable_file("report_20250101.pdf", files) is report
    assert resolve_citable_file("report.pdf", files) is report

    # confirm unwanted instances return None
    assert resolve_citable_file("", files) is None
    assert resolve_citable_file("   ", files) is None
    assert resolve_citable_file(None, files) is None
    assert resolve_citable_file("pdf", files) is None
    assert resolve_citable_file("bob@example.com/secret_20250101.pdf", files) is None


def test_resolve_citable_file_rejects_ambiguous_names():
    """Reject filename references that match more than one permitted file."""
    first = File(original_file="alice@example.com/report_1.pdf", original_file_name="report.pdf")
    second = File(original_file="alice@example.com/report_2.pdf", original_file_name="report.pdf")

    assert resolve_citable_file("report.pdf", [first, second]) is None


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_new_session(
    agents_list: list, alice: User, uploaded_file: File, mocked_connect: ClientConnection
):
    # Given
    message = "Hello Hal."

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect(timeout=5)
        assert connected
        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect):
            await assert_mocked_connect_response(communicator=communicator, message=message)

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == [message]
        assert await get_chat_message_route(alice, ChatMessage.Role.ai) == ["gratitude"]

        await refresh_from_db(uploaded_file)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_staff_user(agents_list: list, staff_user: User, mocked_connect: ClientConnection):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list

        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = staff_user
        connected, _ = await communicator.connect()
        assert connected
        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect):
            await assert_mocked_connect_response(communicator=communicator)

        assert await get_chat_message_route(staff_user, ChatMessage.Role.ai) == ["gratitude"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_existing_session(
    agents_list: list, alice: User, chat: Chat, mocked_connect: ClientConnection
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected
        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect):
            await communicator.send_json_to({"message": "Hello Hal.", "sessionId": str(chat.id)})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response3["data"] == str(chat.id)

            # Close
            await communicator.disconnect()

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == ["Hello Hal."]
        assert await get_chat_message_text(alice, ChatMessage.Role.ai) == ["Good afternoon, Mr. Amor."]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_replaces_null_byte_in_question(
    agents_list: list, alice: User, uploaded_file: File, mocked_connect: ClientConnection
):
    """
    Before saving the user's question null bytes are replaced with a safe replacement character.
    """
    # Given
    message = "Hello Hal."
    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect):
            await assert_mocked_connect_response(communicator=communicator, message=f"{message} \x00")

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == [f"{message} \ufffd"]
        assert await get_chat_message_text(alice, ChatMessage.Role.ai) == ["Good afternoon, Mr. Amor."]
        assert await get_chat_message_route(alice, ChatMessage.Role.ai) == ["gratitude"]
    await refresh_from_db(uploaded_file)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_handles_response_without_citations(
    agents_list: list, alice: User, uploaded_file: File, mocked_connect_without_citations: ClientConnection
):
    """
    Chat consumer can handle LLM responses that don't include citations.
    """
    # Given
    message = "Hello Hal."

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect_without_citations):
            await assert_mocked_connect_response(communicator=communicator, message=message)

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == [message]
        assert await get_chat_message_text(alice, ChatMessage.Role.ai) == ["Good afternoon, Mr. Amor."]
        assert await get_chat_message_route(alice, ChatMessage.Role.ai) == ["gratitude"]
        await refresh_from_db(uploaded_file)


@pytest.mark.parametrize(
    "invalid_source",
    ["", "   "],
)
@pytest.mark.asyncio
async def test_handle_citations_skips_empty_sources(invalid_source):
    """handle_citations passes over citations from LLM that have an empty string as a source."""

    # Given: ChatConsumer receives a citation with an empty string source
    consumer = ChatConsumer()
    consumer.chat_message = MagicMock()
    consumer.chat_message.aget_citations = AsyncMock(return_value=[])
    consumer.get_file_cached = AsyncMock()
    consumer.create_citations = AsyncMock()
    consumer.send_to_client = AsyncMock()
    citations = [consumers_module.AICitation(sources=[consumers_module.Source(source=invalid_source, ref_id="ref_1")])]

    # When: The consumer processes the citations
    with patch(
        "redbox_app.redbox_core.consumers.message_service.render_resources",
        return_value="",
    ):
        await consumer.handle_citations(citations)

    # Then: That source is removed from the citations list
    # Assert: No file lookup occurred
    consumer.get_file_cached.assert_not_awaited()

    # Assert: No citation creation occurred
    consumer.create_citations.assert_not_awaited()

    # Assert: Remaining processing still occurred
    consumer.chat_message.aget_citations.assert_awaited_once()
    consumer.send_to_client.assert_awaited_once()
    assert consumer.send_to_client.await_args.args[0] == "source"


@pytest.mark.asyncio
async def test_handle_citations_keeps_valid_sources():
    """handle_citations ignores invalid sources and processes valid ones."""

    # Given: ChatConsumer receives citations with valid and invalid sources
    consumer = ChatConsumer()

    consumer.chat_message = MagicMock()
    consumer.chat_message.aget_citations = AsyncMock(return_value=[])
    consumer.get_file_cached = AsyncMock(return_value=None)
    consumer.create_citations = AsyncMock()
    consumer.send_to_client = AsyncMock()

    citations = [
        consumers_module.AICitation(
            sources=[
                consumers_module.Source(source="", ref_id="ref_1"),
                consumers_module.Source(source="report_2.pdf", ref_id="ref_2"),
                consumers_module.Source(source="report_3.odt", ref_id="ref_3"),
                consumers_module.Source(source="  ", ref_id="ref_4"),
            ]
        )
    ]

    # When: The consumer processes citations
    with patch(
        "redbox_app.redbox_core.consumers.message_service.render_resources",
        return_value="",
    ):
        await consumer.handle_citations(citations)

    # Then: Only valid sources are processed
    # Assert: Only valid sources are looked up
    assert consumer.get_file_cached.await_count == 2
    assert [call.args[0] for call in consumer.get_file_cached.await_args_list] == ["report_2.pdf", "report_3.odt"]

    # Assert: Only valid sources are processed for citation creation
    assert consumer.create_citations.await_count == 2
    processed_citations = [call.kwargs["ai_citation"] for call in consumer.create_citations.await_args_list]
    assert [citation.sources[0].source for citation in processed_citations] == ["report_2.pdf", "report_3.odt"]
    assert [citation.sources[0].ref_id for citation in processed_citations] == ["ref_2", "ref_3"]

    # Assert: Remaining processing completes
    consumer.chat_message.aget_citations.assert_awaited_once()
    consumer.send_to_client.assert_awaited_once()
    assert consumer.send_to_client.await_args.args[0] == "source"


@pytest.mark.parametrize(
    ("documents", "expected_uris"),
    [
        (
            [
                Document(page_content="Invalid document 1", metadata={"uri": ""}),
                Document(page_content="Invalid document 2", metadata={"uri": "   "}),
                Document(page_content="Invalid document 3", metadata={}),
            ],
            [],
        ),
        (
            [
                Document(page_content="Valid document 1", metadata={"uri": "report_1.pdf"}),
                Document(page_content="Valid document 2", metadata={"uri": "report_2.odt"}),
            ],
            ["report_1.pdf", "report_2.odt"],
        ),
        (
            [
                Document(page_content="Invalid document 1", metadata={"uri": ""}),
                Document(page_content="Valid document 1", metadata={"uri": "report_1.pdf"}),
                Document(page_content="Invalid document 2", metadata={"uri": "   "}),
                Document(page_content="Valid document 2", metadata={"uri": "report_2.odt"}),
                Document(page_content="Invalid document 3", metadata={}),
            ],
            ["report_1.pdf", "report_2.odt"],
        ),
    ],
    ids=["all_invalid", "all_valid", "mixed_valid_and)_invalid"],
)
@pytest.mark.asyncio
async def test_handle_documents_only_processes_valid_source_uris(documents, expected_uris):
    """handle_documents processes documents with valid source URIs and skips those without."""

    # Given: Chat consumer
    consumer = ChatConsumer()
    consumer.chat_message = MagicMock()
    consumer.chat_message.aget_citations = AsyncMock(return_value=[])
    consumer.get_file_cached = AsyncMock(return_value=None)
    consumer.create_citations = AsyncMock()
    consumer.send_to_client = AsyncMock()

    # When: The consumer processes documents
    await consumer.handle_documents(documents)

    # Then: Only valid URIs are searched
    searched_uris = [call.args[0] for call in consumer.get_file_cached.await_args_list]
    assert searched_uris == expected_uris

    # Then: Only valid URIs are passed into citation creation
    processed_uris = [
        source.source
        for call in consumer.create_citations.await_args_list
        for source in call.kwargs["ai_citation"].sources
    ]
    assert processed_uris == expected_uris


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_anonymous_user_error_message():
    # Given

    # When
    communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
    communicator.scope["user"] = AnonymousUser()
    connected, _ = await communicator.connect()
    assert connected

    # Then
    response = await communicator.receive_json_from(timeout=5)
    assert response["type"] == "error"
    assert response["data"] == error_messages.AUTH_REQUIRED

    # Close
    await communicator.disconnect()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_agentic(
    agents_list: list, alice: User, uploaded_file: File, mocked_connect_agentic_search: ClientConnection
):
    # Given
    message = "Hello Hal."

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect_agentic_search):
            # Then
            await communicator.send_json_to({"message": message})
            response1 = await communicator.receive_json_from(timeout=5)
            response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)
            response5 = await communicator.receive_json_from(timeout=5)
            response6 = await communicator.receive_json_from(timeout=5)
            response7 = await communicator.receive_json_from(timeout=5)
            response8 = await communicator.receive_json_from(timeout=5)
            response9 = await communicator.receive_json_from(timeout=5)

            assert response1["type"] == "message_created"
            assert response1["data"]["chat_message_id"]
            assert response1["data"]["chat_message_role"] == "user"
            assert response1["data"]["html"]

            assert response2["type"] == "message_created"
            assert response2["data"]["chat_message_id"]
            assert response2["data"]["chat_message_role"] == "ai"
            assert response2["data"]["html"]

            assert response3["type"] == "session-id"

            assert response4["type"] == "message_update"
            assert response4["data"]["chat_message_id"]
            assert response4["data"]["sr_text"] == "Good afternoon, "
            assert response4["data"]["html"]

            assert response5["type"] == "message_update"
            assert response5["data"]["chat_message_id"]
            assert response5["data"]["sr_text"] == "Mr. "
            assert response5["data"]["html"]

            assert response6["type"] == "message_update"
            assert response6["data"]["chat_message_id"]
            assert response6["data"]["sr_text"] == "Amor."
            assert response6["data"]["html"]

            assert response7["type"] == "route"
            assert response7["data"] == "search/agentic"

            assert response8["type"] == "source"

            assert response9["type"] == "message_complete"
            assert response9["data"]["chat_message_id"]
            assert response9["data"]["title"] == message
            assert response9["data"]["session_id"]
            assert response9["data"]["html"]

            # Close
            await communicator.disconnect()

        assert await get_chat_message_text(alice, ChatMessage.Role.user) == [message]
        assert await get_chat_message_text(alice, ChatMessage.Role.ai) == ["Good afternoon, Mr. Amor."]

        await refresh_from_db(uploaded_file)


@database_sync_to_async
def get_chat_message_text(user: User, role: ChatMessage.Role) -> Sequence[str]:
    return [m.text for m in ChatMessage.objects.filter(chat__user=user, role=role)]


@database_sync_to_async
def get_chat_message_citation_set(user: User, role: ChatMessage.Role) -> Sequence[tuple[str, tuple[int]]]:
    return {
        (citation.text, tuple(citation.page_numbers or []))
        for message in ChatMessage.objects.filter(chat__user=user, role=role)
        for source_file in message.source_files.all()
        for citation in source_file.citation_set.all()
    }


@database_sync_to_async
def get_chat_message_route(user: User, role: ChatMessage.Role) -> Sequence[str]:
    return [m.route for m in ChatMessage.objects.filter(chat__user=user, role=role)]


@pytest.mark.xfail
@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_selected_files(
    agents_list: list,
    alice: User,
    several_files: Sequence[File],
    chat_with_files: Chat,
    mocked_connect_with_several_files: ClientConnection,
):
    # Given
    selected_files: Sequence[File] = several_files[2:]

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect_with_several_files):
            selected_file_core_uuids: Sequence[str] = [f.unique_name for f in selected_files]
            await communicator.send_json_to(
                {
                    "message": "Third question, with selected files?",
                    "sessionId": str(chat_with_files.id),
                    "selectedFiles": selected_file_core_uuids,
                }
            )
            response1 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response1["type"] == "session-id"
            assert response1["data"] == str(chat_with_files.id)

            # Close
            await communicator.disconnect()

        # Then

        # TODO (@brunns): Assert selected files sent to core.
        # Requires fix for https://github.com/django/channels/issues/1091
        # fixed now merged in https://github.com/django/channels/pull/2101, but not released
        # Retry this when a version of Channels after 4.1.0 is released
        mocked_websocket = mocked_connect_with_several_files.return_value.__aenter__.return_value
        expected = json.dumps(
            {
                "message_history": [
                    {"role": "user", "text": "A question?"},
                    {"role": "ai", "text": "An answer."},
                    {"role": "user", "text": "A second question?"},
                    {"role": "ai", "text": "A second answer."},
                    {"role": "user", "text": "Third question, with selected files?"},
                ],
                "selected_files": selected_file_core_uuids,
                "ai_settings": await ChatConsumer.get_ai_settings(alice),
            }
        )
        mocked_websocket.send.assert_called_with(expected)

        # TODO (@brunns): Assert selected files saved to model.
        # Requires fix for https://github.com/django/channels/issues/1091
        all_messages = get_chat_messages(alice)
        last_user_message = [m for m in all_messages if m.rule == ChatMessage.Role.user][-1]
        assert last_user_message.selected_files == selected_files


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_connection_error(
    agents_list: list, alice: User, mocked_breaking_connect: ClientConnection
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_breaking_connect):
            await communicator.send_json_to({"message": "Hello Hal."})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            _response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response4["type"] == "error"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_explicit_unhandled_error(
    agents_list: list, alice: User, mocked_connect_with_explicit_unhandled_error: ClientConnection
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected
        with patch(
            "redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph",
            new=mocked_connect_with_explicit_unhandled_error,
        ):
            await communicator.send_json_to({"message": "Hello Hal."})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)
            response5 = await communicator.receive_json_from(timeout=5)
            _response6 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response4["type"] == "message_update"
            assert response4["data"]["sr_text"] == "Good afternoon, "
            assert response5["type"] == "message_update"
            assert response5["data"]["sr_text"] in error_messages.CORE_ERROR_MESSAGE

            # Close
            await communicator.disconnect()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_rate_limited_error(
    agents_list: list, alice: User, mocked_connect_with_rate_limited_error: ClientConnection
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch(
            "redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph", new=mocked_connect_with_rate_limited_error
        ):
            await communicator.send_json_to({"message": "Hello Hal."})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)
            response5 = await communicator.receive_json_from(timeout=5)
            _response6 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response4["type"] == "message_update"
            assert response4["data"]["sr_text"] == "Good afternoon, "
            assert response5["type"] == "message_update"
            assert response5["data"]["sr_text"] in error_messages.RATE_LIMITED

            # Close
            await communicator.disconnect()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_explicit_no_document_selected_error(
    agents_list: list, alice: User, mocked_connect_with_explicit_no_document_selected_error: ClientConnection
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        with patch(
            "redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph",
            new=mocked_connect_with_explicit_no_document_selected_error,
        ):
            await communicator.send_json_to({"message": "Hello Hal."})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)
            _response5 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response4["type"] == "message_update"
            assert response4["data"]["sr_text"] in error_messages.SELECT_DOCUMENT

            # Close
            await communicator.disconnect()


@pytest.mark.django_db
@pytest.mark.asyncio
async def test_chat_consumer_get_ai_settings(
    chat_with_alice: Chat,
):
    ai_settings = await ChatConsumer.get_ai_settings(chat_with_alice)

    assert ai_settings.chat_map_question_prompt == CHAT_MAP_QUESTION_PROMPT
    assert ai_settings.chat_backend.name == chat_with_alice.chat_backend.name
    assert ai_settings.chat_backend.provider == chat_with_alice.chat_backend.provider
    assert not hasattr(ai_settings, "label")


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_with_american_to_british_conversion(
    agents_list: list, alice: User, mocked_connect_with_american_to_british_conversion: ClientConnection
):
    # # Reset cache before test
    # ChatConsumer.cached_agents = None

    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected
        with patch(
            "redbox_app.redbox_core.consumers.ChatConsumer.redbox.graph",
            new=mocked_connect_with_american_to_british_conversion,
        ):
            mock_get.return_value = agents_list
            communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
            communicator.scope["user"] = alice
            connected, _ = await communicator.connect()
            assert connected
            await communicator.send_json_to({"message": "Hello Hal."})
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)
            response4 = await communicator.receive_json_from(timeout=5)
            _response5 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response4["type"] == "message_update"
            assert (
                response4["data"]["sr_text"] in "Recognise these coloured filters?"
            )  # Converted from "Recognize these colored filters?"

            # Close
            await communicator.disconnect()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_chat_consumer_redbox_state(
    agents_list: list,
    alice: User,
    several_files: Sequence[File],
    chat_with_files: Chat,
):
    # Given
    selected_files: Sequence[File] = several_files[2:]
    previous_selected_files: Sequence[File] = several_files[:2]
    communicator = None

    # When
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        ai_settings = await ChatConsumer.get_ai_settings(chat_with_files)
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        selected_file_uuids: Sequence[str] = [str(f.id) for f in selected_files]
        selected_file_keys: Sequence[str] = [f.unique_name for f in selected_files]
        permitted_file_keys: Sequence[str] = [
            f.unique_name async for f in File.objects.filter(user=alice, status=File.Status.complete)
        ]
        previous_file_keys: Sequence[str] = [f.unique_name for f in previous_selected_files]
        assert selected_file_keys != permitted_file_keys

        with patch("redbox_app.redbox_core.consumers.ChatConsumer.redbox.run") as mock_run:
            await communicator.send_json_to(
                {
                    "message": "Third question, with selected files?",
                    "sessionId": str(chat_with_files.id),
                    "selectedFiles": selected_file_uuids,
                }
            )
            _response1 = await communicator.receive_json_from(timeout=5)
            _response2 = await communicator.receive_json_from(timeout=5)
            response3 = await communicator.receive_json_from(timeout=5)

            # Then
            assert response3["type"] == "session-id"
            assert response3["data"] == str(chat_with_files.id)

            # Close
            await communicator.disconnect()

            # Then
            expected_request = RedboxQuery(
                question="Third question, with selected files?",
                s3_keys=selected_file_keys,
                user_uuid=alice.id,
                chat_history=[
                    {"role": "user", "text": "A question?"},
                    {"role": "ai", "text": "An answer."},
                    {"role": "user", "text": "A second question?"},
                    {"role": "ai", "text": "A second answer."},
                ],
                ai_settings=ai_settings,
                permitted_s3_keys=permitted_file_keys,
                previous_s3_keys=previous_file_keys,
                db_location=None,
            )

            mock_run.return_value = expected_request
            redbox_state = mock_run.call_args.args[0]  # pulls out the args that redbox.run was called with
            assert redbox_state.request.question == expected_request.question, "Question mismatch"
            assert redbox_state.request.user_uuid == expected_request.user_uuid, "UUID mismatch"
            assert redbox_state.request.chat_history == expected_request.chat_history, "Chat history mismatch"
            assert redbox_state.request.ai_settings == expected_request.ai_settings, "AI settings mismatch"

            assert set(redbox_state.request.s3_keys) == set(expected_request.s3_keys), "s3_keys content mismatch"
            assert set(redbox_state.request.permitted_s3_keys) == set(expected_request.permitted_s3_keys), (
                "permitted_s3_keys content mismatch"
            )

            assert set(redbox_state.request.previous_s3_keys) == set(expected_request.previous_s3_keys), (
                "previous_s3_keys mismatch"
            )
            assert redbox_state.request.db_location == expected_request.db_location, "db_location mismatch"


@database_sync_to_async
def get_chat_messages(user: User) -> Sequence[ChatMessage]:
    return list(
        ChatMessage.objects.filter(chat__user=user)
        .order_by("created_at")
        .prefetch_related("chat")
        .prefetch_related("source_files")
        .prefetch_related("selected_files")
    )


@database_sync_to_async
def saved_citations(chat_message: ChatMessage) -> list[tuple]:
    """Return the saved citations for a message in creation order."""
    return list(
        Citation.objects.filter(chat_message=chat_message)
        .order_by("created_at")
        .values_list("citation_name", "file_id", "url", "source")
    )


@pytest.fixture
def ai_message(chat: Chat) -> ChatMessage:
    # A clean AI message: unlike `chat_message`, nothing has created a
    # Citation row on it yet.
    return ChatMessage.objects.create(
        chat=chat,
        text="An answer.",
        role=ChatMessage.Role.ai,
    )


@pytest.fixture
def alice_file(alice: User) -> File:
    return make_file(alice, "alice_report.txt")


@pytest.fixture
def bob_file(bob: User) -> File:
    return make_file(bob, "bob_secret.txt")


@pytest.fixture
def mocked_connect() -> ClientConnection:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content="Good afternoon, ")},
        },
        {"event": "on_chat_model_stream", "tags": [FINAL_RESPONSE_TAG], "data": {"chunk": Token(content="Mr. Amor.")}},
        {"event": "on_chain_end", "tags": [ROUTE_NAME_TAG], "data": {"output": {"route_name": "gratitude"}}},
        {
            "event": "on_custom_event",
            "name": "on_metadata_generation",
            "data": RequestMetadata(
                llm_calls=[
                    LLMCallMetadata(
                        llm_model_name="anthropic.claude-3-7-sonnet-20250219-v1:0", input_tokens=123, output_tokens=1000
                    )
                ],
                selected_files_total_tokens=1000,
                number_of_selected_files=1,
            ),
        },
        {
            "event": "on_custom_event",
            "name": "activity",
            "data": RedboxActivityEvent(
                message="fish and chips",
            ),
        },
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_connect_without_citations() -> CannedGraphLLM:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content="Good afternoon, ")},
        },
        {"event": "on_chat_model_stream", "tags": [FINAL_RESPONSE_TAG], "data": {"chunk": Token(content="Mr. Amor.")}},
        {"event": "on_chain_end", "tags": [ROUTE_NAME_TAG], "data": {"output": {"route_name": "gratitude"}}},
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_breaking_connect() -> ClientConnection:
    mocked_graph = MagicMock(name="mocked_graph")
    mocked_graph.astream_events.side_effect = CancelledError()
    return mocked_graph


@pytest.fixture
def mocked_connect_with_explicit_unhandled_error() -> CannedGraphLLM:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content="Good afternoon, ")},
        },
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content=error_messages.CORE_ERROR_MESSAGE)},
        },
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_connect_with_rate_limited_error() -> CannedGraphLLM:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content="Good afternoon, ")},
        },
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content=error_messages.RATE_LIMITED)},
        },
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_connect_with_explicit_no_document_selected_error() -> CannedGraphLLM:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content=error_messages.SELECT_DOCUMENT)},
        },
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_connect_agentic_search(uploaded_file: File) -> ClientConnection:
    responses = [
        {
            "event": "on_custom_event",
            "name": "response_tokens",
            "data": "Good afternoon, ",
        },
        {
            "event": "on_custom_event",
            "name": "response_tokens",
            "data": "Mr. Amor.",
        },
        {"event": "on_chain_end", "tags": [ROUTE_NAME_TAG], "data": {"output": {"route_name": "search/agentic"}}},
        {
            "event": "on_custom_event",
            "name": "on_source_report",
            "data": [
                Document(metadata={"uri": uploaded_file.unique_name}, page_content="Good afternoon Mr Amor"),
                Document(metadata={"uri": uploaded_file.unique_name}, page_content="Good afternoon Mr Amor"),
                Document(
                    metadata={"uri": uploaded_file.unique_name, "page_number": [34, 35]},
                    page_content="Good afternoon Mr Amor",
                ),
            ],
        },
        {
            "event": "on_custom_event",
            "name": "on_metadata_generation",
            "data": RequestMetadata(
                llm_calls=[
                    LLMCallMetadata(
                        llm_model_name="anthropic.claude-3-7-sonnet-20250219-v1:0", input_tokens=123, output_tokens=1000
                    )
                ],
                selected_files_total_tokens=1000,
                number_of_selected_files=1,
            ),
        },
    ]

    return CannedGraphLLM(responses=responses)


@pytest.fixture
def mocked_connect_with_several_files(several_files: Sequence[File]) -> ClientConnection:
    mocked_websocket = AsyncMock(spec=ClientConnection, name="mocked_websocket")
    mocked_connect = MagicMock(spec=ClientConnection, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "text", "data": "Third "}),
        json.dumps({"resource_type": "text", "data": "answer."}),
        json.dumps(
            {
                "resource_type": "documents",
                "data": [{"s3_key": f.unique_name, "page_content": "a secret forth answer"} for f in several_files[2:]],
            }
        ),
        json.dumps({"resource_type": "end"}),
    ]
    return mocked_connect


@pytest.fixture
def mocked_connect_with_american_to_british_conversion() -> ClientConnection:
    responses = [
        {
            "event": "on_chat_model_stream",
            "tags": [FINAL_RESPONSE_TAG],
            "data": {"chunk": Token(content="Recognize these colored filters?")},
        }
    ]

    return CannedGraphLLM(responses=responses)


@database_sync_to_async
def refresh_from_db(obj: Model) -> None:
    obj.refresh_from_db()


@pytest.mark.asyncio
async def test_connect_with_agents_cache(
    agents_list: list,
    alice: User,
    staff_user: User,
):
    ChatConsumer.redbox = None
    with (
        patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get,
        patch("redbox_app.redbox_core.consumers.Redbox"),
    ):
        mock_get.return_value = agents_list

        # First connection - should call get_all_agents
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        await communicator.connect()
        assert mock_get.call_count == 1

        # Second connection - should use cache
        comm2 = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        comm2.scope["user"] = staff_user
        await comm2.connect()
        assert mock_get.call_count == 1  # Still 1, not 2!

        await communicator.disconnect()
        await comm2.disconnect()


@pytest.mark.asyncio
async def test_connect_with_agents_update_via_db(agents_list: list, alice: User):
    """
    Check when connects:
    1. only use a list of agents from agent_configs
    2. use max tokens from database
    3. use llm backend from database
    """
    ChatConsumer.redbox = None
    with patch("redbox_app.redbox_core.consumers.get_all_agents", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = agents_list
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        await communicator.connect(timeout=5)
        assert mock_get.call_count == 1

        assert "Fake_Agent" not in list(ChatConsumer.redbox.agent_configs.keys())
        assert ChatConsumer.redbox.agent_configs["Internal_Retrieval_Agent"].agents_max_tokens == 100
        assert ChatConsumer.redbox.agent_configs["Internal_Retrieval_Agent"].llm_backend.name == "gpt-4o"


@pytest.mark.parametrize(
    "mock_token, expired, expected_result",  # noqa: PT006
    [("mock_token1", None, "mock_token1"), ("mock_token2", True, None), ("mock_token3", False, "mock_token3")],
)
@pytest.mark.asyncio
async def test_extract_sso_token_success(mock_token, expired, expected_result):
    """Test successful token extraction when the session data is present."""
    consumer = ChatConsumer()
    consumer.scope = {"session": {"_authbroker_token": {"access_token": mock_token}}}
    if expired is not None:
        consumer.scope = {
            "session": {
                "_authbroker_token": {
                    "access_token": mock_token,
                    "expires_at": 0 if expired else timezone.now().timestamp() + 1000,
                }
            }
        }

    with (
        patch.object(ChatConsumer, "accept", return_value=None),
        patch.object(ChatConsumer, "send_to_client", return_value=None),
    ):
        token = await consumer._extract_sso_token()  # noqa: SLF001
    assert token == expected_result


@pytest.mark.asyncio
async def test_extract_sso_token_missing_session():
    """Test that it returns None if 'session' is missing from scope."""
    consumer = ChatConsumer()
    consumer.scope = {}  # Empty scope
    token = await consumer._extract_sso_token()  # noqa: SLF001
    assert token is None


@pytest.mark.asyncio
async def test_extract_sso_token_type_error():
    """Test that it returns None if session is None (triggers TypeError)."""
    consumer = ChatConsumer()
    consumer.scope = {"session": None}
    token = await consumer._extract_sso_token()  # noqa: SLF001
    assert token is None


@pytest.mark.asyncio
async def test_extract_sso_token_missing_key():
    """Test that it returns None if the expected SSO keys are missing."""
    consumer = ChatConsumer()
    consumer.scope = {"session": {"other_key": "no_token_here"}}
    token = await consumer._extract_sso_token()  # noqa: SLF001
    assert token is None


@pytest.mark.asyncio
async def test_connect_updates_sso_token_and_rebuilds_graph_if_redbox_exists(mocker):
    mock_user = MagicMock()
    mock_user.is_authenticated = True
    mock_user.uk_or_us_english = False

    new_token = "new-shiny-sso-token"  # noqa: S105

    expires_at_epoc = (datetime.now(UTC) + timedelta(hours=1)).timestamp()
    scope = {
        "user": mock_user,
        "session": {"_authbroker_token": {"access_token": new_token, "expires_at": int(expires_at_epoc)}},
    }

    mock_redbox_instance = MagicMock()
    ChatConsumer.redbox = mock_redbox_instance
    ChatConsumer.debug = True

    consumer = ChatConsumer(scope=scope)
    consumer.scope = scope

    mocker.patch.object(consumers_module, "database_sync_to_async", side_effect=lambda f: AsyncMock(side_effect=f))

    mocker.patch.object(consumers_module, "get_all_agents", new_callable=AsyncMock)
    consumer.accept = AsyncMock()

    await consumer.connect()

    consumer.accept.assert_called_once()
    ChatConsumer.redbox = None


@pytest.mark.asyncio
async def test_expired_sso_token_forces_logout():
    mock_user = MagicMock()
    mock_user.is_authenticated = True
    mock_user.uk_or_us_english = False

    expired_token = "expired-sso-token"  # noqa: S105
    expires_at_epoc = (datetime.now(UTC) - timedelta(hours=1)).timestamp()

    scope = {
        "user": mock_user,
        "session": {"_authbroker_token": {"access_token": expired_token, "expires_at": int(expires_at_epoc)}},
    }

    mock_redbox_instance = MagicMock()
    ChatConsumer.redbox = mock_redbox_instance
    ChatConsumer.debug = True

    consumer = ChatConsumer(scope=scope)
    consumer.scope = scope
    consumer.accept = AsyncMock()
    consumer.send_to_client = AsyncMock()

    await consumer.connect()

    consumer.accept.assert_called()
    consumer.send_to_client.assert_awaited_once_with("auth_expired", error_messages.AUTH_EXPIRED)
    ChatConsumer.redbox = None


@pytest.mark.asyncio
async def test_llm_conversation_updates_sso_token():
    test_user_uuid = uuid.uuid4()
    mock_user = MagicMock(spec=User)
    mock_user.id = test_user_uuid

    mock_session = MagicMock(spec=ChatModel)
    mock_session.id = uuid.uuid4()
    mock_session.user = mock_user

    mock_user_msg = MagicMock(spec=ChatMessageModel)
    mock_user_msg.text = "How do I test this?"
    mock_user_msg.role = "user"

    mock_ai_msg = MagicMock(spec=ChatMessageModel)
    mock_ai_msg.text = ""
    mock_ai_msg.role = "ai"

    valid_pydantic_settings = PydanticAISettings(
        chat_backend=PydanticChatLLMBackend(name="gpt-4o", provider="openai", description="test-backend"),
        worker_agents=[],
    )

    consumer = ChatConsumer()

    consumer.scope = {"user": mock_user, "session": {"_authbroker_token": {"access_token": "test-sso-token-456"}}}

    consumer.redbox = AsyncMock()
    consumer.send = AsyncMock()
    consumer.chat_message = MagicMock()

    with (
        patch("redbox_app.redbox_core.consumers.ChatMessage.objects.filter") as mock_filter,
        patch.object(ChatConsumer, "get_ai_settings", new_callable=AsyncMock) as mock_get_settings,
        patch.object(ChatConsumer, "update_ai_message", new_callable=AsyncMock),
        patch.object(ChatConsumer, "_files_to_s3_keys", return_value=[]),
        patch.object(ChatConsumer, "_load_agent_plan", new_callable=AsyncMock) as mock_load_plan,
    ):
        mock_filter.return_value.order_by.return_value.__aiter__.return_value = [mock_user_msg, mock_ai_msg]
        mock_get_settings.return_value = valid_pydantic_settings
        mock_load_plan.return_value = (None, "How do I test this?", "")

        await consumer.llm_conversation(
            selected_files=[],
            session=mock_session,
            user=mock_user,
            title="Test Conversation",
            permitted_files=[],
            previous_selected_files=[],
            knowledge_files=[],
            selected_agent_names=None,
        )

        assert consumer.redbox.run.called

        call_kwargs = consumer.redbox.run.call_args.kwargs
        getter = call_kwargs["sso_token_getter"]

        assert getter == consumer._extract_sso_token  # noqa: SLF001

        token = await getter()
        assert token == "test-sso-token-456"  # noqa: S105


@pytest.mark.asyncio
async def test_llm_conversation_multiple_users_get_correct_sso_token():
    """Verify that concurrent users each get their own SSO token and don't cross-contaminate."""
    original_redbox = getattr(ChatConsumer, "redbox", None)

    try:

        def make_consumer(token: str):
            user = MagicMock(spec=User)
            user.id = uuid.uuid4()

            session = MagicMock(spec=ChatModel)
            session.id = uuid.uuid4()
            session.user = user

            consumer = ChatConsumer()
            consumer.scope = {"user": user, "session": {"_authbroker_token": {"access_token": token}}}
            consumer.redbox = MagicMock()
            ChatConsumer.redbox = MagicMock()
            ChatConsumer.redbox.init_datahub_agent = AsyncMock()
            consumer.send = AsyncMock()
            consumer.chat_message = MagicMock()

            return consumer, user, session

        consumer_a, user_a, session_a = make_consumer("token-user-a")
        consumer_b, user_b, session_b = make_consumer("token-user-b")

        valid_pydantic_settings = PydanticAISettings(
            chat_backend=PydanticChatLLMBackend(name="gpt-4o", provider="openai", description="test-backend"),
            worker_agents=[],
        )

        mock_user_msg = MagicMock(spec=ChatMessageModel)
        mock_user_msg.text = "How do I test this?"
        mock_user_msg.role = "user"

        mock_ai_msg = MagicMock(spec=ChatMessageModel)
        mock_ai_msg.text = ""
        mock_ai_msg.role = "ai"

        async def run_conversation(consumer, user, session):
            with (
                patch("redbox_app.redbox_core.consumers.ChatMessage.objects.filter") as mock_filter,
                patch.object(ChatConsumer, "get_ai_settings", new_callable=AsyncMock) as mock_get_settings,
                patch.object(ChatConsumer, "update_ai_message", new_callable=AsyncMock),
                patch.object(ChatConsumer, "_files_to_s3_keys", return_value=[]),
                patch.object(ChatConsumer, "_load_agent_plan", new_callable=AsyncMock) as mock_load_plan,
            ):
                mock_filter.return_value.order_by.return_value.__aiter__.return_value = [mock_user_msg, mock_ai_msg]
                mock_get_settings.return_value = valid_pydantic_settings
                mock_load_plan.return_value = (None, "How do I test this?", "")

                await consumer.llm_conversation(
                    selected_files=[],
                    session=session,
                    user=user,
                    title="Test Conversation",
                    permitted_files=[],
                    previous_selected_files=[],
                    knowledge_files=[],
                    selected_agent_names=None,
                )

        # Run both consumers concurrently to simulate real multi-user scenario
        await asyncio.gather(
            run_conversation(consumer_a, user_a, session_a),
            run_conversation(consumer_b, user_b, session_b),
        )

        # Verify each consumer passed its own getter to run()
        getter_a = consumer_a.redbox.run.call_args.kwargs["sso_token_getter"]
        getter_b = consumer_b.redbox.run.call_args.kwargs["sso_token_getter"]

        assert getter_a == consumer_a._extract_sso_token  # noqa: SLF001
        assert getter_b == consumer_b._extract_sso_token  # noqa: SLF001

        # The critical assertion — getters must be different callables bound to different consumers
        assert getter_a != getter_b

        # Verify each getter returns the correct token for its user
        assert await getter_a() == "token-user-a"
        assert await getter_b() == "token-user-b"

        # Verify cross-contamination is impossible — a's getter cannot return b's token
        assert await getter_a() != "token-user-b"
        assert await getter_b() != "token-user-a"

    finally:
        ChatConsumer.redbox = original_redbox


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_handle_citations_never_links_another_users_file(
    ai_message: ChatMessage,
    alice_file: File,
    bob_file: File,
    caplog,
):
    """Drop invalid or unauthorised uploaded-file citations without saving them."""
    consumer = make_citation_consumer(ai_message, [alice_file])

    refs = [
        "",
        "txt",
        bob_file.original_file.name,
        alice_file.original_file.name,
    ]

    sources = [
        Source(
            source=ref,
            source_type=Citation.Origin.USER_UPLOADED_DOCUMENT,
            ref_id=f"ref_{i}",
        )
        for i, ref in enumerate(refs, start=1)
    ]

    with caplog.at_level(
        logging.WARNING,
        logger="redbox_app.redbox_core.consumers",
    ):
        await consumer.handle_citations([AICitation(sources=sources)])

    # Only Alice's own file is saved; empty, junk and Bob's key are
    # dropped entirely.
    assert await saved_citations(ai_message) == [
        (
            "ref_4",
            alice_file.id,
            None,
            Citation.Origin.USER_UPLOADED_DOCUMENT,
        )
    ]

    assert "dropping citation ref_1: source is not a populated string" in caplog.text

    for ref_id in ("ref_2", "ref_3"):
        assert (f"dropping citation {ref_id}: source does not match a permitted file") in caplog.text

    assert "dropping citation ref_4" not in caplog.text


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_handle_citations_keeps_external_citations(
    ai_message: ChatMessage,
    alice_file: File,
):
    """Save external citations as URLs rather than linking them to uploaded files."""
    consumer = make_citation_consumer(ai_message, [alice_file])

    sources = [
        # URL ends in the display name, but must
        # stay an external citation.
        Source(
            source=GOV_UK_URL,
            source_type=Citation.Origin.GOV_UK,
            ref_id="ref_1",
        ),
        Source(
            source="https://en.wikipedia.org/wiki/Trade",
            source_type=Citation.Origin.WIKIPEDIA,
            ref_id="ref_2",
        ),
        Source(
            source=DATAHUB_URL,
            source_type=Citation.Origin.DATAHUB,
            ref_id="ref_3",
        ),
    ]

    await consumer.handle_citations([AICitation(sources=sources)])

    assert await saved_citations(ai_message) == [
        (
            "ref_1",
            None,
            GOV_UK_URL,
            Citation.Origin.GOV_UK,
        ),
        (
            "ref_2",
            None,
            "https://en.wikipedia.org/wiki/Trade",
            Citation.Origin.WIKIPEDIA,
        ),
        (
            "ref_3",
            None,
            DATAHUB_URL,
            Citation.Origin.DATAHUB,
        ),
    ]

from datetime import UTC, datetime

from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import Chat, ChatMessage, Tool

User = get_user_model()


def test_clear_selected_files(client: Client, alice: User, chat_with_files: Chat):
    # Given
    client.force_login(alice)
    initial_selected_files_count = chat_with_files.last_user_message.selected_files.count()

    # When
    chat_with_files.clear_selected_files()

    # Then
    assert initial_selected_files_count > 0
    assert chat_with_files.last_user_message.selected_files.count() == 0


def test_last_user_message(client: Client, alice: User, chat_with_files: Chat):
    # Given
    client.force_login(alice)
    expected_message_text = "A second question?"
    # When
    message_text = chat_with_files.last_user_message.text

    # Then
    assert expected_message_text == message_text


def test_url(client: Client, alice: User, chat: Chat):
    # Given
    client.force_login(alice)
    expected_url = f"/chats/{chat.id}/"
    # When
    url = chat.url

    # Then
    assert expected_url == url


def test_tool_url(client: Client, alice: User, chat: Chat, default_tool: Tool):
    # Given
    client.force_login(alice)
    chat.tool = default_tool
    expected_url = f"/tools/{default_tool.slug}/chats/{chat.id}/"
    # When
    url = chat.url

    # Then
    assert expected_url == url


def test_filter_by_name_with_no_name_returns_all_user_chats(alice: User):
    # Given
    older_chat = Chat.objects.create(user=alice, name="Older chat")
    newer_chat = Chat.objects.create(user=alice, name="Newer chat")
    ChatMessage.objects.create(chat=older_chat, created_at=datetime(2024, 1, 1, tzinfo=UTC))
    ChatMessage.objects.create(chat=newer_chat, created_at=datetime(2024, 6, 1, tzinfo=UTC))

    # When
    results = Chat.filter_by_name_ordered_by_last_message_date(user=alice)

    # Then
    assert list(results) == [newer_chat, older_chat]


def test_filter_by_name_matches_case_insensitive_substring(alice: User):
    # Given
    matching_chat = Chat.objects.create(user=alice, name="Budget planning")
    other_chat = Chat.objects.create(user=alice, name="Holiday ideas")
    ChatMessage.objects.create(chat=matching_chat, created_at=datetime(2024, 1, 1, tzinfo=UTC))
    ChatMessage.objects.create(chat=other_chat, created_at=datetime(2024, 1, 1, tzinfo=UTC))

    # When
    results = Chat.filter_by_name_ordered_by_last_message_date(user=alice, chat_name_query="budget")

    # Then
    assert list(results) == [matching_chat]


def test_filter_excludes_other_users_chats(alice: User, bob: User):
    # Given
    alice_chat = Chat.objects.create(user=alice, name="Alice's chat")
    bob_chat = Chat.objects.create(user=bob, name="Bob's chat")
    ChatMessage.objects.create(chat=alice_chat, created_at=datetime(2024, 1, 1, tzinfo=UTC))
    ChatMessage.objects.create(chat=bob_chat, created_at=datetime(2024, 1, 1, tzinfo=UTC))

    # When
    results = Chat.filter_by_name_ordered_by_last_message_date(user=alice)

    # Then
    assert list(results) == [alice_chat]

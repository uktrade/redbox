from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import Chat, Tool

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

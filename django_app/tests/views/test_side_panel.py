from collections.abc import Callable, Sequence
from datetime import datetime
from http import HTTPStatus
from zoneinfo import ZoneInfo

import pytest
from bs4 import BeautifulSoup
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from waffle.testutils import override_flag

from redbox_app.redbox_core.models import (
    Chat,
)

User = get_user_model()


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_sidebar_renders_five_most_recent_chats(
    user_with_chats_with_messages_over_time: User, make_chat_with_message_at_specific_time: Callable, client: Client
):
    # Given
    user = user_with_chats_with_messages_over_time

    utc = ZoneInfo("UTC")
    # pad chats to test 5 chat cap
    for i in range(6):
        make_chat_with_message_at_specific_time(f"pad-chat-{i}", datetime(1900, 1, 1, tzinfo=utc), user)

    client.force_login(user)
    chats: Sequence[Chat] = Chat.get_ordered_by_last_message_date(user)
    expected_names = [chat.name for chat in chats[:5]]
    expected_names.append("All chats")
    url = reverse("chats")

    # When
    response = client.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Then
    assert response.status_code == HTTPStatus.OK
    chat_links = soup.select("ul.ids-card__list li.ids-card__item .item-link")
    rendered_names = [link.get_text(strip=True).removeprefix("Chat:").strip() for link in chat_links]
    assert rendered_names == expected_names
    assert len(chat_links) == 6


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_active_chat_is_marked_selected(user_with_chats_with_messages_over_time: User, client: Client):
    # Given
    user = user_with_chats_with_messages_over_time
    client.force_login(user)
    chat = Chat.get_ordered_by_last_message_date(user)[0]
    url = reverse("chats", kwargs={"chat_id": chat.id})  # adjust name to your detail route

    # When
    response = client.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Then
    selected = soup.select("li.ids-card__item--selected")
    assert len(selected) == 1
    assert selected[0]["data-chatid"] == str(chat.id)
    assert selected[0].select_one(".item-link")["aria-current"] == "page"

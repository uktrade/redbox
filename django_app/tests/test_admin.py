import csv
import io
import logging
from http import HTTPStatus

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import ChatMessage
from redbox_app.redbox_core.serializers import ChatMessageSerializer, ChatSerializer, UserSerializer

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_chat_export(superuser: User, chat_message_with_rating: ChatMessage, client: Client):
    # Given
    client.force_login(superuser)

    # When

    # See https://docs.djangoproject.com/en/dev/ref/contrib/admin/#reversing-admin-urls
    url = reverse("admin:redbox_core_chat_changelist")
    data = {"action": "export_as_csv", "_selected_action": [chat_message_with_rating.chat.pk]}
    response = client.post(url, data, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8"))))
    assert len(rows) == 1
    row = rows[0]
    assert row["history_name"] == "A chat"
    assert row["history_user"] == "alice@cabinetoffice.gov.uk"
    assert row["message_text"] == "A question?"
    assert row["message_rating"] == "3"
    assert row["message_rating_chips"] == "['speed', 'accuracy', 'blasphemy']"


@pytest.mark.django_db()
def test_chat_export_without_ratings(superuser: User, chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(superuser)

    # When
    url = reverse("admin:redbox_core_chat_changelist")
    data = {"action": "export_as_csv", "_selected_action": [chat_message.chat.pk]}
    response = client.post(url, data, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8"))))
    assert len(rows) == 1
    row = rows[0]
    assert row["history_name"] == "A chat"
    assert row["history_user"] == "alice@cabinetoffice.gov.uk"
    assert row["message_text"] == "A question?"
    assert row["message_rating"] == ""
    assert row["message_rating_chips"] == ""


@pytest.mark.django_db()
def test_message_serializer(chat_message_with_citation_and_tokens: ChatMessage):
    expected = {
        "rating": 3,
        "rating_chips": ["apple", "pear"],
        "rating_text": "not bad",
        "role": "ai",
        "route": "chat",
        "selected_files": [],
        "text": "An answer.",
    }

    expected_token_usage = [
        {"use_type": "input", "model_name": "anthropic.claude-3-sonnet-20240229-v1:0", "token_count": 20},
        {"use_type": "output", "model_name": "anthropic.claude-3-sonnet-20240229-v1:0", "token_count": 200},
    ]

    actual = ChatMessageSerializer().to_representation(chat_message_with_citation_and_tokens)
    for k, v in expected.items():
        assert actual[k] == v, k

    assert actual["source_files"][0]["file_name"].startswith("original_file")

    for k, v in expected_token_usage[0].items():
        assert actual["token_use"][0][k] == v, k

    for k, v in expected_token_usage[1].items():
        assert actual["token_use"][1][k] == v, k


@pytest.mark.django_db()
def test_chat_serializer(chat_message_with_citation: ChatMessage):
    actual = ChatSerializer().to_representation(chat_message_with_citation.chat)
    assert "id" in actual
    assert "created_at" in actual
    assert "modified_at" in actual
    assert "messages" in actual
    assert isinstance(actual["messages"], list)
    assert len(actual["messages"]) == 1
    message = actual["messages"][0]
    assert "id" in message
    assert "created_at" in message
    assert "modified_at" in message
    assert "text" in message
    assert message["text"] == "An answer."
    assert "role" in message
    assert "selected_files" in message
    assert "source_files" in message
    assert "rating" in message
    assert "rating_text" in message
    assert "rating_chips" in message
    assert "token_use" in message


@pytest.mark.django_db()
def test_user_serializer(chat_message_with_citation: ChatMessage):
    expected = {
        "ai_experience": "Experienced Navigator",
        "business_unit": "Digital, Data and Technology (DDaT)",
        "grade": "D",
        "profession": "IA",
    }
    actual = UserSerializer().to_representation(chat_message_with_citation.chat.user)
    for k, v in expected.items():
        assert actual[k] == v, k

    assert actual["chats"][0]["messages"][0]["text"] == "An answer."

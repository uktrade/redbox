import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    ChatMessage,
    Citation,
    Skill,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_internal_url(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    chat_id = chat_message_with_citation.chat.id
    message_id = chat_message_with_citation.id

    # When
    citation = Citation.objects.get(chat_message=chat_message_with_citation)

    # Then
    assert citation.internal_url == f"/chats/{chat_id}/citations/{message_id}/#{citation.id}"


@pytest.mark.django_db(transaction=True)
def test_internal_url_with_skill(
    client: Client, alice: User, chat_message_with_citation: ChatMessage, default_skill: Skill
):
    # Given
    client.force_login(alice)
    chat_id = chat_message_with_citation.chat.id
    message_id = chat_message_with_citation.id
    skill_slug = default_skill.slug

    # When
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    chat_message_with_citation.chat.skill = default_skill
    chat_message_with_citation.chat.save()

    # Then
    assert citation.internal_url == f"/skills/{skill_slug}/chats/{chat_id}/citations/{message_id}/#{citation.id}"

import uuid

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import Chat
from redbox_app.redbox_core.services import url as url_service

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_get_chat_url(client: Client, chat_with_alice: Chat, default_skill):
    # Given
    client.force_login(chat_with_alice.user)
    chat_id = chat_with_alice.id
    skill_slug = default_skill.slug

    # When
    new_chat_link = url_service.get_chat_url()
    new_skill_chat_link = url_service.get_chat_url(skill_slug=skill_slug)
    chat_link = url_service.get_chat_url(chat_id=chat_id)
    skill_chat_link = url_service.get_chat_url(chat_id=chat_id, skill_slug=skill_slug)

    # Then
    assert new_chat_link == "/chats/"
    assert new_skill_chat_link == f"/skills/{skill_slug}/chats/"
    assert chat_link == f"/chats/{chat_id}/"
    assert skill_chat_link == f"/skills/{skill_slug}/chats/{chat_id}/"


@pytest.mark.django_db(transaction=True)
def test_get_citation_url(client: Client, alice: User, default_skill):
    # Given
    client.force_login(alice)
    message_id = uuid.uuid4()
    citation_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    skill_slug = default_skill.slug

    # When
    citation_link = url_service.get_citation_url(
        message_id=message_id,
        citation_id=citation_id,
        chat_id=chat_id,
    )
    skill_citation_link = url_service.get_citation_url(
        message_id=message_id,
        citation_id=citation_id,
        chat_id=chat_id,
        skill_slug=skill_slug,
    )

    # Then
    assert citation_link == f"/chats/{chat_id}/citations/{message_id}/#{citation_id}"
    assert skill_citation_link == f"/skills/{skill_slug}/chats/{chat_id}/citations/{message_id}/#{citation_id}"

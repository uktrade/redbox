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
    slug = default_skill.slug
    chat_message_with_citation.chat.skill = default_skill
    chat_message_with_citation.chat.save()

    # When
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    chat_message_with_citation.chat.skill = default_skill
    chat_message_with_citation.chat.save()

    # Then
    assert citation.internal_url == f"/tools/{slug}/chats/{chat_id}/citations/{message_id}/#{citation.id}"


@pytest.mark.django_db(transaction=True)
def test_display_name(client: Client, alice: User, external_citation: Citation, internal_citation: Citation):
    # Given
    client.force_login(alice)

    # When
    expected_external_display_name = str(external_citation.uri)
    expected_internal_display_name = internal_citation.file.file_name

    # Then
    assert external_citation.display_name == expected_external_display_name
    assert internal_citation.display_name == expected_internal_display_name


@pytest.mark.django_db(transaction=True)
def test_ref_id(
    client: Client,
    alice: User,
    chat_message_with_citation: ChatMessage,
    external_citation: Citation,
    internal_citation: Citation,
):
    # Given
    client.force_login(alice)
    citation = Citation.objects.get(chat_message=chat_message_with_citation)

    # When
    citation.citation_name = "ref_2"
    citation.save()
    internal_citation.citation_name = "ref_x"

    # Then
    assert citation.ref_id == 2
    with pytest.raises(TypeError):
        assert external_citation.ref_id
    with pytest.raises(TypeError):
        assert internal_citation.ref_id

from django.contrib.auth import get_user_model
from django.test import Client
from yarl import URL

from redbox_app.redbox_core.models import (
    ChatMessage,
    Citation,
    Skill,
)

User = get_user_model()


def test_get_citations(
    client: Client, alice: User, chat_message: ChatMessage, external_citation: Citation, internal_citation: Citation
):
    # Given
    client.force_login(alice)

    # When
    chat_message.refresh_from_db()
    citations = chat_message.get_citations()

    # Then
    assert external_citation in citations
    assert internal_citation in citations
    assert citations[0].display_name == "http://example.com"
    assert citations[0].uri == URL("http://example.com")
    assert citations[1].display_name.startswith("original_file")
    assert citations[1].uri.parts[-1].startswith("original_file")


def test_citations_url(client: Client, alice: User, external_citation: Citation):
    # Given
    client.force_login(alice)
    chat_message = external_citation.chat_message
    expected_url = f"/chats/{chat_message.chat.id}/citations/{chat_message.id}/"
    # When
    citations_url = chat_message.citations_url

    # Then
    assert citations_url == expected_url


def test_citations_url_with_skill(client: Client, alice: User, external_citation: Citation, default_skill: Skill):
    # Given
    client.force_login(alice)
    chat_message = external_citation.chat_message
    chat_message.chat.skill = default_skill
    chat_message.save()
    expected_url = f"/tools/{default_skill.slug}/chats/{chat_message.chat.id}/citations/{chat_message.id}/"

    # When
    citations_url = chat_message.citations_url

    # Then
    assert citations_url == expected_url

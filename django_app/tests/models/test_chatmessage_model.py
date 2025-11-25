from django.contrib.auth import get_user_model
from django.test import Client
from yarl import URL

from redbox_app.redbox_core.models import (
    ChatMessage,
    Citation,
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

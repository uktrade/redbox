import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from yarl import URL

from redbox_app.redbox_core.models import (
    ChatMessage,
    Citation,
    File,
    Tool,
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
def test_internal_url_with_tool(
    client: Client, alice: User, chat_message_with_citation: ChatMessage, default_tool: Tool
):
    # Given
    client.force_login(alice)
    chat_id = chat_message_with_citation.chat.id
    message_id = chat_message_with_citation.id
    slug = default_tool.slug
    chat_message_with_citation.chat.tool = default_tool
    chat_message_with_citation.chat.save()

    # When
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    chat_message_with_citation.chat.tool = default_tool
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


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("source", "error_msg"),
    [
        (Citation.Origin.USER_UPLOADED_DOCUMENT, "file must be specified for a user-uploaded-document"),
        (Citation.Origin.WIKIPEDIA, "url must be specified for an external citation"),
    ],
)
def test_citation_save_fail_file_url_not_set(chat_message: ChatMessage, source, error_msg):
    citation = Citation(chat_message=chat_message, text="hello", source=source)

    with pytest.raises(ValueError, match=error_msg):
        citation.save()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("source", "error_msg"),
    [
        (Citation.Origin.USER_UPLOADED_DOCUMENT, "url should not be specified for a user-uploaded-document"),
        (Citation.Origin.WIKIPEDIA, "file should not be specified for an external citation"),
    ],
)
def test_citation_save_fail_file_and_url_set(chat_message: ChatMessage, uploaded_file: File, source, error_msg):
    citation = Citation(
        chat_message=chat_message,
        text="hello",
        source=source,
        url="http://example.com",
        file=uploaded_file,
    )

    with pytest.raises(ValueError, match=error_msg):
        citation.save()


def test_internal_citation_uri(chat_message: ChatMessage, uploaded_file: File):
    citation = Citation(
        chat_message=chat_message,
        text="hello",
        source=Citation.Origin.USER_UPLOADED_DOCUMENT,
        file=uploaded_file,
    )
    citation.save()
    assert citation.uri.parts[-1].startswith("original_file")


def test_external_citation_uri(
    chat_message: ChatMessage,
):
    citation = Citation(
        chat_message=chat_message,
        text="hello",
        source=Citation.Origin.WIKIPEDIA,
        url="http://example.com",
    )
    citation.save()
    assert citation.uri == URL("http://example.com")


@pytest.mark.parametrize(("value", "expected"), [("invalid origin", None), ("Wikipedia", "Wikipedia")])
def test_try_parse_origin(value, expected):
    assert Citation.Origin.try_parse(value) == expected


@pytest.mark.django_db(transaction=True)
def test_is_internal(client: Client, alice: User, internal_citation: Citation):
    # Given
    client.force_login(alice)

    # Then
    assert internal_citation.is_internal


@pytest.mark.django_db(transaction=True)
def test_is_external(client: Client, alice: User, external_citation: Citation):
    # Given
    client.force_login(alice)

    # Then
    assert external_citation.is_external

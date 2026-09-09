import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import ChatMessage, Citation
from redbox_app.redbox_core.services import message as message_service
from redbox_app.redbox_core.types import CitationMap

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_replace_ref(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    footnote_counter = 1
    citation.citation_name = "ref_1"
    citation.save()

    # When
    message_text = message_service.replace_ref(
        message_text=f"{citation.text} [ref_1]",
        citation=citation,
        footnote_counter=footnote_counter,
    )
    expected_result = f"{citation.text} {message_service.render_citation(citation, footnote_counter)}"

    # Then
    assert message_text == expected_result


@pytest.mark.django_db(transaction=True)
def test_replace_text_in_answer(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    footnote_counter = 1
    citation.text_in_answer = citation.text
    citation.save()

    # When
    message_text = message_service.replace_text_in_answer(
        message_text=citation.text,
        citation=citation,
        footnote_counter=footnote_counter,
    )
    expected_result = f"{citation.text}{message_service.render_citation(citation, footnote_counter)}"

    # Then
    assert message_text == expected_result


@pytest.mark.django_db(transaction=True)
def test_citation_not_inserted(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    footnote_counter = 1

    # When
    message_with_citation = f"{citation.text} {message_service.render_citation(citation, footnote_counter)}"

    # Then
    assert not message_service.citation_not_inserted(
        message_text=message_with_citation,
        citation=citation,
        footnote_counter=footnote_counter,
    )
    assert message_service.citation_not_inserted(
        message_text=citation.text,
        citation=citation,
        footnote_counter=footnote_counter,
    )


@pytest.mark.django_db(transaction=True)
def test_render_citation_placeholder(client: Client, alice: User):
    # Given
    client.force_login(alice)
    footnote_counter = 123

    # When
    citation_template = message_service.render_citation_placeholder(footnote_counter)

    # Then
    assert citation_template
    assert str(footnote_counter) in citation_template


@pytest.mark.django_db(transaction=True)
def test_render_resources(client: Client, alice: User, external_citation: Citation):
    # Given
    client.force_login(alice)
    message = external_citation.chat_message

    # When
    resources_template = message_service.render_resources(message)

    # Then
    assert resources_template
    assert str(message.id) in resources_template
    assert external_citation.internal_url in resources_template
    assert external_citation.display_name in resources_template


@pytest.mark.django_db(transaction=True)
def test_streaming_replace_refs(client: Client, alice: User, external_citation: Citation):
    # Given
    client.force_login(alice)
    text = f"{external_citation.text} ref_1"
    message_with_citation = f"{external_citation.text} {message_service.render_citation_placeholder(1)}"

    # When
    rendered_text = message_service.streaming_replace_refs(
        text=text,
        citation_map=CitationMap(),
    )

    # Then
    assert rendered_text
    assert rendered_text == message_with_citation


@pytest.mark.django_db(transaction=True)
def test_decorate_message(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    citation = Citation.objects.filter(chat_message=chat_message_with_citation).first()
    original_text = chat_message_with_citation.text
    chat_message_with_citation.text += " ref_1"

    # When
    decorated_message = message_service.decorate_message(message=chat_message_with_citation, as_html=False)
    rendered_citation = message_service.render_citation(citation=citation, footnote_counter=1)
    expected_text = f"{original_text} {rendered_citation}"

    # Then
    assert decorated_message
    assert decorated_message.text == expected_text

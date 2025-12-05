import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import ChatMessage, Citation
from redbox_app.redbox_core.services import message as message_service

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
    expceted_result = (
        f'{citation.text} <a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>'
    )

    # Then
    assert message_text == expceted_result


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
    expceted_result = (
        f'{citation.text}<a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>'
    )

    # Then
    assert message_text == expceted_result


@pytest.mark.django_db(transaction=True)
def test_citation_not_inserted(client: Client, alice: User, chat_message_with_citation: ChatMessage):
    # Given
    client.force_login(alice)
    citation = Citation.objects.get(chat_message=chat_message_with_citation)
    footnote_counter = 1

    # When
    message_with_citation = (
        f'{citation.text} <a class="rb-footnote-link" href="{citation.internal_url}">{footnote_counter}</a>'
    )

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

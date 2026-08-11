from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.test import Client
from django.urls import reverse
from waffle.testutils import override_flag

from redbox_app.redbox_core.models import ChatMessage, ChatMessageFeedback

FEEDBACK_FLAG = "enable_feedback_redesign"


# --- get_feedback_buttons ---


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=False)
def test_get_buttons_404_when_flag_inactive(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback-buttons", kwargs={"message_id": chat_message.id})

    response = client.get(url)

    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_get_buttons_renders_buttons_when_no_feedback(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback-buttons", kwargs={"message_id": chat_message.id})

    response = client.get(url)

    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    heading = soup.find("legend", class_="feedback__heading")
    assert heading is not None
    assert heading.get_text(strip=True) == "Did you get what you wanted from this response?"
    buttons = [b.get_text(strip=True) for b in soup.find_all("button")]
    assert "Yes" in buttons
    assert "Not quite" in buttons


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_get_buttons_renders_thanks_when_feedback_exists(alice, chat_message: ChatMessage, client: Client):
    ChatMessageFeedback.objects.create(message=chat_message, is_positive=True)
    client.force_login(alice)
    url = reverse("chat-message-feedback-buttons", kwargs={"message_id": chat_message.id})

    response = client.get(url)

    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    heading = soup.find("legend", class_="feedback__heading")
    assert heading is not None
    assert heading.get_text(strip=True) == "Thanks for your feedback"
    button = soup.find("button")
    assert button is not None
    assert button.get_text(strip=True) == "Change feedback"


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_get_buttons_404_for_other_users_message(bob, chat_message: ChatMessage, client: Client):
    # chat_message belongs to alice; bob must not see it
    client.force_login(bob)
    url = reverse("chat-message-feedback-buttons", kwargs={"message_id": chat_message.id})

    response = client.get(url)

    assert response.status_code == HTTPStatus.NOT_FOUND


# --- chat_message_feedback ---


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=False)
def test_feedback_404_when_flag_inactive(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.post(url, data={"is_positive": True})

    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_post_creates_and_redirects(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.post(url, data={"is_positive": True})

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == reverse("chat-message-feedback-buttons", kwargs={"message_id": chat_message.id})
    feedback = ChatMessageFeedback.objects.get(message=chat_message)
    assert feedback.is_positive is True


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_post_updates_existing(alice, chat_message: ChatMessage, client: Client):
    ChatMessageFeedback.objects.create(message=chat_message, is_positive=True)
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.post(url, data={"is_positive": False, "reason": "INACCURATE", "detail": "test 1"})

    assert response.status_code == HTTPStatus.FOUND
    assert ChatMessageFeedback.objects.filter(message=chat_message).count() == 1
    feedback = ChatMessageFeedback.objects.get(message=chat_message)
    assert feedback.is_positive is False
    assert feedback.reason == ["INACCURATE"]
    assert feedback.detail == "test 1"


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_post_show_form_returns_form(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.post(f"{url}?show_form=true", data={"is_positive": True})

    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    assert soup.find("div", class_="feedback-form") is not None
    submit = soup.find("button", attrs={"type": "submit"})
    assert submit is not None
    assert submit.get_text(strip=True) == "Send feedback"


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_post_invalid_returns_422(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.post(url, data={"reason": ["not-a-valid-choice"]})

    assert response.status_code == 422
    assert response["HX-Reswap"] == "innerHTML"
    soup = BeautifulSoup(response.content, "html.parser")
    assert soup.find("div", class_="feedback-form") is not None


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_delete_removes_and_renders_buttons(alice, chat_message: ChatMessage, client: Client):
    ChatMessageFeedback.objects.create(message=chat_message, is_positive=True)
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.delete(url)

    assert response.status_code == HTTPStatus.OK
    assert not ChatMessageFeedback.objects.filter(message=chat_message).exists()
    soup = BeautifulSoup(response.content, "html.parser")
    heading = soup.find("legend", class_="feedback__heading")
    assert heading is not None
    assert heading.get_text(strip=True) == "Did you get what you wanted from this response?"


@pytest.mark.django_db
@override_flag(FEEDBACK_FLAG, active=True)
def test_feedback_delete_no_instance_is_safe(alice, chat_message: ChatMessage, client: Client):
    client.force_login(alice)
    url = reverse("chat-message-feedback", kwargs={"message_id": chat_message.id})

    response = client.delete(url)

    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    heading = soup.find("legend", class_="feedback__heading")
    assert heading is not None
    assert heading.get_text(strip=True) == "Did you get what you wanted from this response?"

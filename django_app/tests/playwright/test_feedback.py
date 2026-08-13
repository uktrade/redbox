import pytest
from pages import FeedbackComponent, LandingPage
from playwright.sync_api import expect
from waffle.testutils import override_switch

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.models import ChatMessageFeedback


@pytest.fixture
def feedback_switch_active():
    with override_switch(flags.ENABLE_FEEDBACK_REDESIGN, active=True):
        yield


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("feedback_switch_active")
def test_positive_feedback_journey(page, live_server_url, vyvyan_ai_message):
    landing_page = LandingPage(page, live_server_url)

    # Sign in
    chats_page = landing_page.sign_in()

    existing_chat_page = chats_page.navigate_to_titled_chat(vyvyan_ai_message.chat.name)
    message = next(m for m in existing_chat_page.all_messages if m.element.locator("[id^='feedback-']").count())
    feedback_component = FeedbackComponent.for_message(message)
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    initial_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(initial_feedback) == 0

    # click yes
    feedback_component_with_positive_feedback = feedback_component.click_yes()
    feedback_component_with_positive_feedback.wait_for_feedback_ready()
    expect(feedback_component_with_positive_feedback.change_feedback_button).to_be_visible()

    saved_positive_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(saved_positive_feedback) == 1
    assert saved_positive_feedback[0].is_positive is True
    assert saved_positive_feedback[0].detail == ""
    assert saved_positive_feedback[0].reason == []

    # Change feedback
    feedback_component_with_changed_feedback = feedback_component_with_positive_feedback.click_change_feedback()
    feedback_component_with_changed_feedback.wait_for_feedback_ready()

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    deleted_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(deleted_feedback) == 0


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("feedback_switch_active")
def test_negative_feedback_journey_no_details(page, live_server_url, vyvyan_ai_message):
    landing_page = LandingPage(page, live_server_url)

    # Sign in
    chats_page = landing_page.sign_in()

    existing_chat_page = chats_page.navigate_to_titled_chat(vyvyan_ai_message.chat.name)
    message = next(m for m in existing_chat_page.all_messages if m.element.locator("[id^='feedback-']").count())
    feedback_component = FeedbackComponent.for_message(message)
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    initial_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(initial_feedback) == 0

    # click not quite
    feedback_component.click_not_quite()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.form).to_be_visible()
    expect(feedback_component.id_prefer_to_not_say_button).to_be_visible()
    expect(feedback_component.send_feedback_button).to_be_visible()

    saved_negative_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(saved_negative_feedback) == 1
    assert saved_negative_feedback[0].is_positive is False
    assert saved_negative_feedback[0].detail == ""
    assert saved_negative_feedback[0].reason == []

    # click I'd prefer not to say
    feedback_component.click_id_prefer_not_to_say()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.change_feedback_button).to_be_visible()

    saved_negative_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(saved_negative_feedback) == 1
    assert saved_negative_feedback[0].is_positive is False
    assert saved_negative_feedback[0].detail == ""
    assert saved_negative_feedback[0].reason == []

    # Change feedback
    feedback_component.click_change_feedback()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    deleted_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(deleted_feedback) == 0


@pytest.mark.django_db(transaction=True)
@pytest.mark.usefixtures("feedback_switch_active")
def test_negative_feedback_journey_with_details(page, live_server_url, vyvyan_ai_message):
    landing_page = LandingPage(page, live_server_url)

    # Sign in
    chats_page = landing_page.sign_in()

    existing_chat_page = chats_page.navigate_to_titled_chat(vyvyan_ai_message.chat.name)
    message = next(m for m in existing_chat_page.all_messages if m.element.locator("[id^='feedback-']").count())
    feedback_component = FeedbackComponent.for_message(message)
    feedback_component.wait_for_feedback_ready()

    container_id = feedback_component.container.get_attribute("id")
    message_id = container_id.removeprefix("feedback-")
    resp = page.request.get(f"{live_server_url}/chat-message/{message_id}/buttons/")
    print("STATUS:", resp.status)  # noqa: T201
    print("BODY:", resp.text())  # noqa: T201

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    initial_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(initial_feedback) == 0

    # click not quite
    feedback_component.click_not_quite()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.form).to_be_visible()

    saved_negative_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(saved_negative_feedback) == 1
    assert saved_negative_feedback[0].is_positive is False
    assert saved_negative_feedback[0].detail == ""
    assert saved_negative_feedback[0].reason == []

    # fill out and submit form
    feedback_component.select_reasons(["It was inaccurate", "It wasn't what I asked for"])
    feedback_component.input_detail(text="test 1")
    feedback_component.click_send_feedback()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.change_feedback_button).to_be_visible()

    saved_negative_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(saved_negative_feedback) == 1
    assert saved_negative_feedback[0].is_positive is False
    assert saved_negative_feedback[0].detail == "test 1"
    assert saved_negative_feedback[0].reason == ["INACCURATE", "UNASKED"]

    # Change feedback
    feedback_component.click_change_feedback()
    feedback_component.wait_for_feedback_ready()

    expect(feedback_component.not_quite_button).to_be_visible()
    expect(feedback_component.yes_button).to_be_visible()

    deleted_feedback = ChatMessageFeedback.objects.filter(message=vyvyan_ai_message)
    assert len(deleted_feedback) == 0

from collections.abc import Sequence
from datetime import datetime, timedelta
from http import HTTPStatus
from zoneinfo import ZoneInfo

import pytest
from django.contrib.auth import get_user_model
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import Client, RequestFactory
from freezegun import freeze_time

from redbox_app.redbox_core.models import Chat, ChatMessage, File, Tool
from redbox_app.redbox_core.services import chats as chats_service

User = get_user_model()

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


@pytest.mark.django_db(transaction=True)
def test_get_context(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    several_files: Sequence[File],
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice

    middleware = SessionMiddleware(lambda _: None)
    middleware.process_request(request)
    request.session.save()

    # When
    context = chats_service.get_context(request)
    chat_context = chats_service.get_context(request, chat_with_files.id)

    # Then
    assert context["chat_id"] is None
    assert context["messages"] == []
    assert len(context["chats"]) == len(Chat.get_ordered_by_last_message_date(alice))
    assert context["current_chat"] is None
    assert len(context["completed_files"]) == len(several_files)

    assert chat_context["chat_id"] == chat_with_files.id
    assert len(chat_context["messages"]) == len(ChatMessage.get_messages_ordered_by_citation_priority(chat_with_files))
    assert len(chat_context["chats"]) == len(Chat.get_ordered_by_last_message_date(alice))
    assert chat_context["current_chat"] == chat_with_files
    assert len(context["completed_files"]) == len(several_files)


@pytest.mark.django_db(transaction=True)
def test_render_chats(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice

    middleware = SessionMiddleware(lambda _: None)
    middleware.process_request(request)
    request.session.save()

    context = chats_service.get_context(request)
    chat_context = chats_service.get_context(request, chat_with_files.id)

    # When
    response = chats_service.render_chats(request, context)
    chat_response = chats_service.render_chats(request, chat_context)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert "canned-prompts" in response.content.decode()

    assert chat_response.status_code == HTTPStatus.OK
    assert "canned-prompts" not in chat_response.content.decode()


@pytest.mark.django_db(transaction=True)
def test_render_conversations(
    client: Client,
    user_with_chats_with_messages_over_time: User,
    chat_with_files: Chat,
):
    # Given
    alice = user_with_chats_with_messages_over_time
    client.force_login(alice)
    factory = RequestFactory()
    request = factory.get("/chats/")
    request.user = alice

    middleware = SessionMiddleware(lambda _: None)
    middleware.process_request(request)
    request.session.save()

    chat_context = chats_service.get_context(request, chat_with_files.id)

    # When
    response = chats_service.render_conversations(request)
    chat_response = chats_service.render_conversations(request, chat_context)

    # Then
    assert response.status_code == HTTPStatus.OK
    assert "ids-list-row--selected" not in response.content.decode()

    assert chat_response.status_code == HTTPStatus.OK
    assert "ids-list-row--selected" in chat_response.content.decode()


@pytest.mark.django_db(transaction=True)
@freeze_time(NOW)
def test_groups_chats_into_correct_buckets(alice, make_chat_with_message_at_specific_time):
    # Given
    expectations = {
        "today": [NOW - timedelta(hours=1)],
        "yesterday": [NOW - timedelta(days=1)],
        "previous_7_days": [NOW - timedelta(days=2), NOW - timedelta(days=7)],
        "previous_30_days": [NOW - timedelta(days=8), NOW - timedelta(days=30)],
        "previous_year": [NOW - timedelta(days=31), NOW - timedelta(days=365)],
        "over_a_year": [NOW - timedelta(days=366)],
    }
    for bucket, datetimes in expectations.items():
        for i, dt in enumerate(datetimes):
            make_chat_with_message_at_specific_time(name=f"{bucket}-{i}", message_at=dt, user=alice)

    # When
    grouped = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC)

    # Then
    for bucket, datetimes in expectations.items():
        names = [c.name for c in getattr(grouped, bucket)]

        # correct number in bucket
        assert len(names) == len(datetimes)
        # correct order within bucket
        assert names == sorted(names, key=lambda n: int(n.split("-")[1]))


@pytest.mark.django_db(transaction=True)
@freeze_time(NOW)
def test_group_chats_timezone_shifts_bucket_boundaries(alice, make_chat_with_message_at_specific_time):
    # Given
    message_at = NOW.replace(hour=0, minute=30)
    make_chat_with_message_at_specific_time("late-night", message_at, alice)
    los_angeles = ZoneInfo("America/Los_Angeles")

    # When
    grouped_utc = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC)
    grouped_la = chats_service.get_filtered_and_grouped_chats(alice, tz=los_angeles)

    # Then
    assert [c.name for c in grouped_utc.today] == ["late-night"]
    assert [c.name for c in grouped_la.yesterday] == ["late-night"]
    chat = grouped_la.yesterday[0]
    assert chat.local_last_message_datetime == message_at.astimezone(los_angeles)
    assert chat.last_message_datetime == message_at


@pytest.mark.django_db(transaction=True)
@freeze_time(NOW)
def test_excludes_empty_archived_and_other_users_chats(alice, bob, make_chat_with_message_at_specific_time):
    # Given
    make_chat_with_message_at_specific_time("visible", NOW - timedelta(hours=1), alice)
    Chat.objects.create(user=alice, name="empty")  # no messages
    archived = make_chat_with_message_at_specific_time("archived", NOW - timedelta(hours=1), alice)
    Chat.objects.filter(id=archived.id).update(archived=True)
    make_chat_with_message_at_specific_time("bobs-chat", NOW - timedelta(hours=1), bob)

    # When
    grouped = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC)

    # Then
    assert [c.name for c in grouped.today] == ["visible"]


@pytest.mark.django_db(transaction=True)
@freeze_time(NOW)
def test_filters_by_tool_and_name_query(alice, make_chat_with_message_at_specific_time):
    # Given
    tool = Tool.objects.create(name="summariser")  # adjust required fields
    make_chat_with_message_at_specific_time("Invest lens", NOW - timedelta(hours=1), alice, tool=tool)
    make_chat_with_message_at_specific_time("What lens for my camera", NOW - timedelta(hours=2), alice)
    make_chat_with_message_at_specific_time("What should I have for dinner?", NOW - timedelta(hours=3), alice)

    # When
    all_chats = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC)
    by_tool = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC, tool=tool)
    by_name = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC, chat_name_query="lens")

    # Then
    assert len(all_chats.today) == 3

    assert len(by_tool.today) == 1
    assert [c.name for c in by_tool.today] == ["Invest lens"]
    assert by_tool.today[0].tool == "summariser"

    assert len(by_name.today) == 2
    assert {c.name for c in by_name.today} == {"Invest lens", "What lens for my camera"}


@pytest.mark.django_db(transaction=True)
@freeze_time(NOW)
def test_future_message_is_clamped_to_today(alice, make_chat_with_message_at_specific_time):
    # Given
    make_chat_with_message_at_specific_time("from-the-future", NOW + timedelta(minutes=5), alice)

    # When
    grouped = chats_service.get_filtered_and_grouped_chats(alice, tz=UTC)

    # Then — must not raise, lands in today
    assert [c.name for c in grouped.today] == ["from-the-future"]

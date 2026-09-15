from datetime import datetime, timedelta
from http import HTTPStatus
from zoneinfo import ZoneInfo

import pytest
from bs4 import BeautifulSoup
from django.contrib.auth import get_user_model
from django.test import Client
from freezegun import freeze_time
from waffle.testutils import override_flag

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

User = get_user_model()


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_all_chats_view_renders_grouped_chats(alice: User, client: Client, make_chat_with_message_at_specific_time):
    # Given
    client.force_login(alice)
    make_chat_with_message_at_specific_time("recent chat", NOW - timedelta(hours=1), alice)
    make_chat_with_message_at_specific_time("old chat", NOW - timedelta(days=400), alice)

    # When
    with freeze_time(NOW):
        response = client.get("/chats/all/")

    # Then
    assert response.status_code == HTTPStatus.OK

    soup = BeautifulSoup(response.content, "html.parser")
    assert soup.find(id="chat-results") is not None
    headings = [h2.text for h2 in soup.select("#chat-results h2")]
    assert headings == ["Today", "Over a year"]
    assert soup.find("a", string="recent chat") is not None


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_all_chats_view_requires_login(client: Client):
    # When
    response = client.get("/chats/all/")

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.url.startswith("/auth/login/")


@pytest.mark.django_db
def test_all_chats_view_404s_when_flag_disabled(alice, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get("/chats/all/")

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_search_view_filters_by_query(alice, client: Client, make_chat_with_message_at_specific_time):
    # Given
    client.force_login(alice)
    make_chat_with_message_at_specific_time("Weekly report", NOW - timedelta(hours=1), alice)
    make_chat_with_message_at_specific_time("Holiday plans", NOW - timedelta(hours=2), alice)

    # When
    with freeze_time(NOW):
        response = client.get("/chats/search/", {"q": "  weekly  "})

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    links = [a.text for a in soup.select("table a")]
    assert links == ["Weekly report"]


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_search_view_uses_tz_param_for_grouping(alice, client: Client, make_chat_with_message_at_specific_time):
    # Given — 00:30 UTC today is yesterday evening in Los Angeles
    client.force_login(alice)
    make_chat_with_message_at_specific_time("late-night", NOW.replace(hour=0, minute=30), alice)

    # When
    with freeze_time(NOW):
        response = client.get("/chats/search/", {"tz": "America/Los_Angeles"})

    # Then
    soup = BeautifulSoup(response.content, "html.parser")
    assert [h2.text for h2 in soup.find_all("h2")] == ["Yesterday"]


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
@pytest.mark.parametrize("bad_tz", ["Not/AZone", ""])
def test_search_view_falls_back_to_utc_on_bad_tz(
    alice: User, client: Client, bad_tz: str, make_chat_with_message_at_specific_time
):
    # Given
    client.force_login(alice)
    make_chat_with_message_at_specific_time("late-night", NOW.replace(hour=0, minute=30), alice)

    # When
    with freeze_time(NOW):
        response = client.get("/chats/search/", {"tz": bad_tz})

    # Then — grouped as UTC, no 500
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content, "html.parser")
    assert [h2.text for h2 in soup.find_all("h2")] == ["Today"]


@pytest.mark.django_db
@override_flag("enable_chats_redesign", active=True)
def test_search_view_shows_empty_state(alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get("/chats/search/", {"q": "nonexistent"})

    # Then
    soup = BeautifulSoup(response.content, "html.parser")
    assert "No chats found" in soup.get_text()

import datetime
from unittest import mock

import pytest
import pytz

from redbox_app.jinja2 import (
    environment,
    humanise_expiry,
    humanize_short_timedelta,
    humanize_timedelta,
    markdown,
    remove_refs,
    to_user_timezone,
    url,
)


class TestMarkdown:
    def test_markdown_basic(self):
        # Given
        text = "**Bold** text"

        # When
        result = markdown(text)

        # Then
        assert result == '<p class=""><strong>Bold</strong> text'

    def test_markdown_with_class(self):
        # Given
        text = "**Bold** text"

        # When
        result = markdown(text, cls="custom-class")

        # Then
        assert result == '<p class="custom-class"><strong>Bold</strong> text'


class TestHumaniseExpiry:
    def test_humanise_expiry_future(self):
        # Given
        delta = datetime.timedelta(days=2)

        # When
        result = humanise_expiry(delta)

        # Then
        assert result == "2 days"

    def test_humanise_expiry_past(self):
        # Given
        delta = datetime.timedelta(days=-2)

        # When
        result = humanise_expiry(delta)

        # Then
        assert result == "2 days ago"


def test_url_pass_kwargs_and_args():
    # Given / When / Then all together
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        url("some_view_name", 1, key="value")
    assert str(excinfo.value) == "Use *args or **kwargs, not both."


def test_humanise_timedelta():
    # Given
    delta = datetime.timedelta(days=1, hours=6)

    # When
    result = humanize_timedelta(delta)

    # Then
    assert result == "a day"


class TestHumanizeShortTimedelta:
    def test_within_limit(self):
        # Given (120 minutes = 2 hours)
        minutes = 120

        # When
        result = humanize_short_timedelta(minutes=minutes)

        # Then
        assert result == "2 hours"

    def test_exceeds_limit(self):
        # Given (12100 minutes > 200 hours (12000 minutes))
        minutes = 12100

        # When
        result = humanize_short_timedelta(minutes=minutes, hours_limit=200)

        # Then
        assert result == "More than 200 hours"

    def test_custom_message(self):
        # Given
        minutes = 12100

        # When
        result = humanize_short_timedelta(minutes=minutes, hours_limit=200, too_large_msg="Too much time")

        # Then
        assert result == "Too much time"


def test_to_user_timezone():
    # Given - Create a UTC datetime
    utc_dt = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.UTC)

    london_tz = pytz.timezone("Europe/London")

    with mock.patch("pytz.timezone", return_value=london_tz):
        # When
        result = to_user_timezone(utc_dt)

        # Then
        expected = utc_dt.astimezone(london_tz).strftime("%H:%M %d/%m/%Y")
        assert result == expected


@pytest.mark.parametrize(
    "text, expected",  # noqa: PT006
    [
        ("This is a text with ref_1 reference", "This is a text with reference"),
        ("This is a text with ref_2 reference", "This is a text with reference"),
        ("This is a text with ref_3 reference", "This is a text with reference"),
        ("This has ref_4 and also ref_5 in it", "This has and also in it"),
        ("No refs here", "No refs here"),
    ],
)
def test_remove_refs(text, expected):
    # Given in parameters

    # When
    result = remove_refs(text)

    # Then
    assert result == expected


class TestEnvironment:
    def test_creates_jinja_env(self):
        # Given, When
        env = environment()

        # Then
        assert hasattr(env, "filters")
        assert hasattr(env, "globals")

        # Test that filters are registered
        assert "static" in env.filters
        assert "url" in env.filters
        assert "humanise_expiry" in env.filters
        assert "remove_refs" in env.filters
        assert "template_localtime" in env.filters
        assert "to_user_timezone" in env.filters
        assert "environment" in env.filters
        assert "security" in env.filters

        # Test that globals are registered
        assert "static" in env.globals
        assert "url" in env.globals
        assert "humanise_expiry" in env.globals
        assert "remove_refs" in env.globals
        assert "template_localtime" in env.globals
        assert "to_user_timezone" in env.globals
        assert "environment" in env.globals
        assert "security" in env.globals
        assert "google_analytics_tag" in env.globals
        assert "google_analytics_link" in env.globals
        assert "google_analytics_iframe_src" in env.globals

    def test_passes_options(self):
        # Given, When
        env = environment(trim_blocks=True)

        # Then
        assert env.trim_blocks is True

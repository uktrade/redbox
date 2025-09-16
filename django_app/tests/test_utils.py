from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest
from django.http import HttpRequest, HttpResponse
from django.utils import timezone

from redbox_app.redbox_core.utils import get_date_group, render_with_oob


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (timezone.now().date(), "Today"),
        ((timezone.now() - timedelta(days=1)).date(), "Previous"),
        ((timezone.now() - timedelta(days=2)).date(), "Previous"),
        ((timezone.now() - timedelta(days=7)).date(), "Previous"),
        ((timezone.now() - timedelta(days=8)).date(), "Previous"),
        ((timezone.now() - timedelta(days=30)).date(), "Previous"),
        ((timezone.now() - timedelta(days=31)).date(), "Previous"),
    ],
)
def test_date_group_calculation(given: date, expected: str):
    # Given

    # When
    actual = get_date_group(given)

    # Then
    assert actual == expected


@pytest.mark.django_db()
def test_render_with_oob_single_template(monkeypatch):
    # Given
    mock_render_to_string = MagicMock(return_value="<div>Rendered Template</div>")
    monkeypatch.setattr("redbox_app.redbox_core.utils.render_to_string", mock_render_to_string)

    mock_request = MagicMock(spec=HttpRequest)
    templates = [{"template": "test_template.html", "context": {"key": "value"}, "request": mock_request}]

    # When
    response = render_with_oob(templates)

    # Then
    assert isinstance(response, HttpResponse)
    assert response.content.decode() == "<div>Rendered Template</div>"
    mock_render_to_string.assert_called_once_with("test_template.html", {"key": "value"}, mock_request, using="jinja2")


@pytest.mark.django_db()
def test_render_with_oob_multiple_templates(monkeypatch):
    # Given
    mock_render_to_string = MagicMock(side_effect=["<div>First Template</div>", "<div>Second Template</div>"])
    monkeypatch.setattr("redbox_app.redbox_core.utils.render_to_string", mock_render_to_string)

    mock_request = MagicMock(spec=HttpRequest)
    templates = [
        {"template": "template1.html", "context": {"key1": "value1"}, "request": mock_request},
        {"template": "template2.html", "context": {"key2": "value2"}, "request": mock_request, "engine": "django"},
    ]

    # When
    response = render_with_oob(templates)

    # Then
    assert isinstance(response, HttpResponse)
    assert response.content.decode() == "<div>First Template</div><div>Second Template</div>"
    assert mock_render_to_string.call_count == 2

    # Check the calls were made with correct parameters
    calls = mock_render_to_string.call_args_list
    assert calls[0][0] == ("template1.html", {"key1": "value1"}, mock_request)
    assert calls[0][1] == {"using": "jinja2"}
    assert calls[1][0] == ("template2.html", {"key2": "value2"}, mock_request)
    assert calls[1][1] == {"using": "django"}


@pytest.mark.django_db()
def test_render_with_oob_empty_list(monkeypatch):
    # Given
    mock_render_to_string = MagicMock()
    monkeypatch.setattr("redbox_app.redbox_core.utils.render_to_string", mock_render_to_string)

    templates = []

    # When
    response = render_with_oob(templates)

    # Then
    assert isinstance(response, HttpResponse)
    assert response.content.decode() == ""
    mock_render_to_string.assert_not_called()

from datetime import date, timedelta

import pytest
from django.utils import timezone

from redbox_app.redbox_core.utils import get_date_group


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

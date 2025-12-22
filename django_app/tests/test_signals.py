import pytest
from django.conf import settings
from django.utils import timezone
from pytest_mock import MockerFixture

from redbox_app.redbox_core.models import ChatMessage, MonitorWebSearchResults


class TestWebSearchLimit:
    @pytest.mark.parametrize(("no_of_api_calls", "is_email_sent"), [(1, False), (3, True)])
    @pytest.mark.django_db
    def test_no_email_when_under_limit(
        self, no_of_api_calls, is_email_sent, chat_message: ChatMessage, mocker: MockerFixture
    ):
        """Test that no email is sent when we're under the daily limit."""
        mock_send_email = mocker.patch(
            "redbox_app.redbox_core.signals.notifications_service.send_email", return_value="email sent"
        )
        settings.WEB_SEARCH_API_LIMIT = 2
        # Create fewer records than the limit
        for _ in range(no_of_api_calls):
            MonitorWebSearchResults.objects.create(
                chat_message=chat_message,
                user_text="Fake",
                web_search_urls="www.fake.com",
                web_search_api_count=1,
                created_at=timezone.now(),
            )

        if not is_email_sent:
            mock_send_email.assert_not_called()

        if is_email_sent:
            mock_send_email.assert_called_once()

    @pytest.mark.django_db
    def test_only_counts_today(self, chat_message: ChatMessage, mocker: MockerFixture):
        """Test that only today's records are counted toward the limit."""
        mock_send_email = mocker.patch(
            "redbox_app.redbox_core.signals.notifications_service.send_email", return_value="email sent"
        )
        # Create a record with yesterday's date
        MonitorWebSearchResults.objects.create(
            chat_message=chat_message,
            user_text="Fake",
            web_search_urls="www.fake.com",
            web_search_api_count=1,
            created_at=timezone.now().date() - timezone.timedelta(days=1),
        )

        settings.WEB_SEARCH_API_LIMIT = 2
        # Create records up to the limit for today
        for _ in range(settings.WEB_SEARCH_API_LIMIT):
            MonitorWebSearchResults.objects.create(
                chat_message=chat_message,
                user_text="Fake",
                web_search_urls="www.fake.com",
                web_search_api_count=1,
                created_at=timezone.now(),
            )

        # Now an email should have been sent
        mock_send_email.assert_called_once()

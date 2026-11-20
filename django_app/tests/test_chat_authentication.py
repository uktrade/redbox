from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from django.test import RequestFactory
from django.urls import reverse
from django.utils import timezone

from redbox_app.redbox_core.views.chat_views import ChatsView


def test_expired_sso_token_redirects_to_authentication():
    """
    A user with an authenticated Django session and expired SSO token is not shown the chat page.
    """

    # GIVEN:
    # A request to the chats page
    request = RequestFactory().get("/chats/")

    # with an authenticated Django user
    request.user = SimpleNamespace(is_authenticated=True)

    # with a SSO token that expired 5 seconds ago
    request.session = {
        "_authbroker_token": {
            "access_token": "fake_test_token",
            "expires_at": timezone.now().timestamp() - 5,
        }
    }

    # WHEN: The chats page is requested
    response = ChatsView.as_view()(request)

    # THEN The user is redirected before chat renders
    redirect_url = urlsplit(response["Location"])
    assert parse_qs(redirect_url.query)["next"] == ["/chats/"]
    assert response.status_code == 302
    assert redirect_url.path == reverse("reauthenticate")

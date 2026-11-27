from urllib.parse import urlencode, urlsplit

import pytest
from playwright.sync_api import Page, expect
from tests.playwright.pages import LandingPage


@pytest.mark.django_db(transaction=True)
def test_reauthentication_preserves_destination(
    page: Page,
    live_server_url,
):
    """Reauthenticate returns the user to the correct chat page."""
    LandingPage(page, live_server_url).sign_in()
    destination = "/chats/?q=fake%20prompt"
    reauthenticate_url = f"{live_server_url}/reauthenticate/?{urlencode({'next': destination})}"

    page.goto(reauthenticate_url)

    expect(page).to_have_url(f"{live_server_url}{destination}")
    expect(page.locator("#message")).to_have_text("fake prompt")


@pytest.mark.django_db(transaction=True)
def test_expired_sso_redirects_back_to_chat(page: Page, live_server_url, expire_sso_session):
    """An expired SSO token causes reauthentication then routes to the preserved URL."""
    LandingPage(page, live_server_url).sign_in()
    expire_sso_session()
    destination = "/chats/?q=fake%20prompt"

    with page.expect_response(
        lambda response: urlsplit(response.url).path == "/reauthenticate/" and response.status == 302
    ):
        page.goto(f"{live_server_url}{destination}")

    expect(page).to_have_url(f"{live_server_url}{destination}")
    expect(page.locator("#message")).to_have_text("fake prompt")

import pytest
from playwright.sync_api import Page, expect
from tests.playwright.pages import LandingPage


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("destination", ["/chats/", "/chats/?q=fake%20prompt"])
def test_prompt_draft_restored_after_reauthentication(
    page: Page, live_server_url, expire_sso_session, destination: str
):
    """An unsent prompt is restored after reauthentication."""

    page.route_web_socket("**/ws/chat/", lambda ws: ws.send('{"type":"auth_expired","data":"Session expired"}'))

    LandingPage(page, live_server_url).sign_in()

    # Go to destination page
    destination_url = f"{live_server_url}{destination}"
    page.goto(destination_url)
    expect(page).to_have_url(destination_url)

    # Prompt from URL query param (if any) should appear in chat box
    message_input = page.locator("#message")

    if "?q=" in destination:
        expect(message_input).to_have_text("fake prompt")

    # Enter a prompt
    user_prompt = "fake prompt from user"
    message_input.fill(user_prompt)
    expect(message_input).to_have_text(user_prompt)

    # Expire session
    expire_sso_session()

    # Submit the prompt
    page.locator('.send-button[type="submit"]').click()

    # Reauth should return user to the original chat page
    expect(page).to_have_url(f"{live_server_url}{destination}")

    # User's draft prompt should automatically appear in chat box
    expect(page.locator("#message")).to_have_text(user_prompt)

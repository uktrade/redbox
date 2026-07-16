from bs4 import BeautifulSoup
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import Client

User = get_user_model()


def test_support_view_contains_contact_email_and_version_number(client: Client):
    # Given
    version = settings.REDBOX_VERSION

    # When
    response = client.get("/support/")

    # Then
    soup = BeautifulSoup(response.content)
    mailto_links = [
        a.get("href", "").removeprefix("mailto:") for a in soup.find_all("a") if a.get("href", "").startswith("mailto:")
    ]
    assert mailto_links
    assert version in str(response.content)


def test_feedback_view_contains_contact_email(client: Client):
    # Given
    contact_email = settings.CONTACT_EMAIL

    # When
    response = client.get("/feedback/")

    # Then
    soup = BeautifulSoup(response.content)
    mailto_links = [
        a.get("href", "").removeprefix("mailto:") for a in soup.find_all("a") if a.get("href", "").startswith("mailto:")
    ]
    assert mailto_links[0] == contact_email


def test_feedback_view_shows_backlink_when_not_signed_in(client: Client):
    # Given
    # An unauthenticated client

    # When
    response = client.get("/feedback/")

    # Then
    soup = BeautifulSoup(response.content)
    back_links = soup.find_all("a", class_="govuk-back-link")
    assert len(back_links) == 1


def test_feedback_view_hides_backlink_when_signed_in(alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get("/feedback/")

    # Then
    soup = BeautifulSoup(response.content)
    back_links = soup.find_all("a", class_="govuk-back-link")
    assert len(back_links) == 0

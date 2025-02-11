from bs4 import BeautifulSoup
from django.conf import settings
from django.test import Client


def test_api_v0_messages(client: Client):
    # Given
    version = settings.REDBOX_VERSION

    # When
    response = client.get("/api/v0/messages/")
    print(BeautifulSoup(response.content))

    assert False

    # Then
    # soup = BeautifulSoup(response.content)
    # mailto_links = [
    #     a.get("href", "").removeprefix("mailto:") for a in soup.find_all("a") if a.get("href", "").startswith("mailto:")
    # ]
    # assert mailto_links
    # assert version in str(response.content)

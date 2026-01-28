import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    Chat,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_first_time_user(client: Client, bob: User, chat_with_alice: Chat):
    # Given
    client.force_login(chat_with_alice.user)

    # When
    response_1 = chat_with_alice.user.first_time_user
    response_2 = bob.first_time_user

    # Then
    assert response_1 is False
    assert response_2 is True

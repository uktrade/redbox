import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import Team
from redbox_app.redbox_core.services import teams as teams_service

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_create_team(client: Client, alice: User):
    # Given
    client.force_login(alice)
    team_name_1 = "Test Team 1"
    team_name_2 = "Test Team 2"

    # When
    team_1 = teams_service.create_team(team_name_1, "DDaT")
    team_2 = teams_service.create_team(team_name_2, "DDaT", alice)

    team_1_obj = Team.objects.filter(team_name=team_name_1).first()
    team_2_obj = Team.objects.filter(team_name=team_name_2).first()

    # Then
    assert team_1 == team_1_obj
    assert team_2 == team_2_obj
    assert len(team_2_obj.get_members()) > 0

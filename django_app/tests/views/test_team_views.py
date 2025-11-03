import logging
from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    Team,
    UserTeamMembership,
)

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_user_can_edit_their_own_team_members(redbox_team: Team, alice: User, bob: User, client: Client):
    # Given
    client.force_login(alice)
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)
    membership = redbox_team.add_member(bob, UserTeamMembership.RoleType.MEMBER)

    # When
    response = client.get(f"/team/edit-member-row/{membership.id}/")
    response_2 = client.post(
        f"/team/edit-member/{membership.id}/", data={"role-type": UserTeamMembership.RoleType.ADMIN.value}
    )

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content)
    select = soup.find("select", {"name": "role-type"})
    assert select is not None
    assert len(select.find_all("option")) > 0
    assert response_2.status_code == HTTPStatus.OK
    assert redbox_team.is_admin(bob)


@pytest.mark.django_db
def test_user_cannot_edit_other_teams_members(redbox_team: Team, alice: User, bob: User, client: Client):
    # Given
    client.force_login(bob)
    membership = redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)

    # When
    response = client.get(f"/team/edit-member-row/{membership.id}/")

    # Then
    assert response.status_code == HTTPStatus.FORBIDDEN


@pytest.mark.django_db
def test_user_cannot_edit_nonexistent_members(bob: User, client: Client):
    # Given
    client.force_login(bob)

    # When
    response = client.get("/team/edit-member-row/67/")

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
def test_user_can_delete_their_own_team_members(redbox_team: Team, alice: User, bob: User, client: Client):
    # Given
    client.force_login(alice)
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)
    membership = redbox_team.add_member(bob)

    # When
    response = client.delete(f"/team/delete-member/{membership.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_user_cannot_delete_other_teams_members(redbox_team: Team, alice: User, bob: User, client: Client):
    # Given
    client.force_login(alice)
    membership = redbox_team.add_member(bob, UserTeamMembership.RoleType.ADMIN)

    # When
    response = client.delete(f"/team/delete-member/{membership.id}/")

    # Then
    assert response.status_code == HTTPStatus.FORBIDDEN


@pytest.mark.django_db
def test_user_cannot_delete_nonexistent_team_members(alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.delete("/team/delete-member/67/")

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db(transaction=True)
def test_user_can_create_team(alice: User, client: Client):
    # Given
    client.force_login(alice)
    team_name = "Test Team"

    # When
    response = client.post(
        "/team/create-team/",
        data={
            "team-name": team_name,
            "directorate": "DDaT",
        },
    )
    response_2 = client.post("/team/create-team/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert response_2.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    team = Team.objects.filter(team_name=team_name).first()
    assert team.is_admin(alice)


@pytest.mark.django_db(transaction=True)
def test_user_can_add_team_members(redbox_team: Team, alice: User, bob: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.post(
        f"/team/{redbox_team.id}/add-member/",
        data={
            "user-id": bob.id,
            "role-type": UserTeamMembership.RoleType.ADMIN.value,
        },
    )
    response_2 = client.post(
        f"/team/{redbox_team.id}/add-member/?next=/chats/",
        data={
            "user-id": alice.id,
            "role-type": UserTeamMembership.RoleType.ADMIN.value,
        },
    )

    # Then
    assert response.status_code == HTTPStatus.OK
    assert redbox_team.is_admin(bob)
    assert response_2.status_code == HTTPStatus.FOUND
    assert redbox_team.is_admin(alice)
    assert response_2.headers.get("Location") == "/chats/"


@pytest.mark.django_db(transaction=True)
def test_user_can_delete_team(alice: User, redbox_team: Team, client: Client):
    # Given
    client.force_login(alice)
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)

    # When
    response = client.delete(f"/team/{redbox_team.id}/delete-team/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert not Team.objects.filter(id=redbox_team.id).exists()

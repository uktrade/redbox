import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    Team,
    UserTeamMembership,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_get_members(client: Client, alice: User, bob: User, redbox_team: Team):
    # Given
    client.force_login(alice)

    # When
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)
    redbox_team.add_member(bob)

    # Then
    assert UserTeamMembership.objects.filter(
        user=alice, team=redbox_team, role_type=UserTeamMembership.RoleType.ADMIN
    ).exists()

    assert len(redbox_team.get_members()) == 2


@pytest.mark.django_db(transaction=True)
def test_eligible_users(client: Client, alice: User, bob: User, peter_rabbit: User, redbox_team: Team):
    # Given
    client.force_login(alice)

    # When
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)

    # Then
    eligible_users = redbox_team.eligible_users()

    assert alice not in eligible_users
    assert bob in eligible_users
    assert peter_rabbit in eligible_users


@pytest.mark.django_db(transaction=True)
def test_add_member(client: Client, alice: User, bob: User, redbox_team: Team):
    # Given
    client.force_login(alice)

    # When
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)
    redbox_team.add_member(bob)

    # Then
    assert UserTeamMembership.objects.filter(
        user=alice, team=redbox_team, role_type=UserTeamMembership.RoleType.ADMIN
    ).exists()
    assert UserTeamMembership.objects.filter(
        user=bob, team=redbox_team, role_type=UserTeamMembership.RoleType.MEMBER
    ).exists()


@pytest.mark.django_db(transaction=True)
def test_is_admin(client: Client, alice: User, bob: User, redbox_team: Team):
    # Given
    client.force_login(alice)

    # When
    redbox_team.add_member(alice, UserTeamMembership.RoleType.ADMIN)
    redbox_team.add_member(bob)

    # Then
    assert redbox_team.is_admin(alice)
    assert not redbox_team.is_admin(bob)

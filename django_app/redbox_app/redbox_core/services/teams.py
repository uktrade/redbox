import logging
from uuid import UUID

from django.contrib.auth import get_user_model

from redbox_app.redbox_core.models import Team, UserTeamMembership

User = get_user_model()
logger = logging.getLogger(__name__)


def create_team(team_name: str, directorate: str, user: User | UUID | None = None):
    if user and not isinstance(user, User):
        user = User.objects.get(pk=user)

    team = Team(team_name=team_name, directorate=directorate)
    team.save()

    if user:
        team.add_member(user, UserTeamMembership.RoleType.ADMIN)
    return team

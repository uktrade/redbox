import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from tests.models.test_chat_model import User

from redbox_app.redbox_core.models import Chat, File, UserTeamMembership, UserTool
from redbox_app.setting_enums import Environment
from redbox_app.settings import MOCK_SSO_USERNAME

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Command(BaseCommand):
    # This email is hardcoded in the https://github.com/uktrade/mock-sso repo
    mock_user_email = MOCK_SSO_USERNAME
    help = "Migrate the chats, files and tools belonging to specific user id to the mock sso user"

    def add_arguments(self, parser):
        parser.add_argument("user_id", type=str)

    def handle(self, **options):
        if not Environment.is_local:
            error_message = "This command can only be run on a local environment"
            raise CommandError(error_message)

        mock_sso_user = User.objects.filter(username=self.mock_user_email).first()
        if mock_sso_user is None:
            error_message = f"User with username '{self.mock_user_email}' not found"
            raise CommandError(error_message)

        user_id = options["user_id"]
        user = User.objects.filter(id=user_id).first()
        if user is None:
            error_message = f"User with id '{user_id}' not found"
            raise CommandError(error_message)

        self._migrate_objects(mock_sso_user, user)

    def _migrate_objects(self, mock_sso_user, user):
        with transaction.atomic():
            # Update any UserTool that the mock sso user doesn't already have
            UserTool.objects.filter(user=user).exclude(pk__in=UserTool.objects.filter(user=mock_sso_user)).update(
                user=mock_sso_user
            )
            Chat.objects.filter(user=user).update(user=mock_sso_user)
            File.objects.filter(user=user).update(user=mock_sso_user)
            UserTeamMembership.objects.filter(user=user).update(user=mock_sso_user)

            logger.info("Related objects updated for user %s", user)

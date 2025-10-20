import logging
from uuid import UUID

from django.conf import settings
from django.contrib.auth import get_user_model
from notifications_python_client.notifications import NotificationsAPIClient

from redbox_app.redbox_core.models import Team

User = get_user_model()
logger = logging.getLogger(__name__)


def send_team_addition_email(admin: User | UUID, user: User | UUID, team: Team | UUID):
    if not isinstance(admin, User):
        admin = User.objects.get(pk=admin)
    if not isinstance(user, User):
        user = User.objects.get(pk=user)
    if not isinstance(admin, Team):
        team = Team.objects.get(id=team)

    notifications_client = NotificationsAPIClient(settings.GOVUK_NOTIFY_API_KEY)
    team_addition_template = settings.GOVUK_NOTIFY_TEAM_ADDITION_EMAIL_TEMPLATE_ID
    recipient_email = user.email

    personalisation = {
        "team_name": team.team_name,
        "team_admin": admin.name or admin.email,
        "correspondence_url": settings.CURRENT_SITE_URL,
    }
    response = notifications_client.send_email_notification(
        email_address=recipient_email,
        template_id=team_addition_template,
        personalisation=personalisation,
    )
    logger.info("Team addition email sent to %s", recipient_email)
    return response


def send_email(recipient_email: str, subject: str, body: str):
    notifications_client = NotificationsAPIClient(settings.GOVUK_NOTIFY_API_KEY)
    plain_email_template = settings.GOVUK_NOTIFY_PLAIN_EMAIL_TEMPLATE_ID

    response = notifications_client.send_email_notification(
        email_address=recipient_email,
        template_id=plain_email_template,
        personalisation={"subject": subject, "body": body},
    )
    logger.info("Email sent to %s", recipient_email)
    return response

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
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


def send_email(
    recipient_email: str, subject: str, body: str, reference: Any | None = None, check_if_sent: bool = False
):
    if check_if_sent:
        # check if an email has already been sent today
        notifications_client = NotificationsAPIClient(settings.GOVUK_NOTIFY_API_KEY)
        notifications = notifications_client.get_all_notifications(
            template_type="email", status="delivered", reference=reference
        )["notifications"]
        has_today_email = [
            notification
            for notification in notifications
            if (datetime.fromisoformat(notification["sent_at"]).date() == timezone.now().date())
            and (notification["email_address"] == recipient_email)
        ]
        if has_today_email:
            logger.info("Email has alredy been sent to %s today", recipient_email)
            return None

    notifications_client = NotificationsAPIClient(settings.GOVUK_NOTIFY_API_KEY)
    plain_email_template = settings.GOVUK_NOTIFY_PLAIN_EMAIL_TEMPLATE_ID

    response = notifications_client.send_email_notification(
        email_address=recipient_email,
        template_id=plain_email_template,
        personalisation={"subject": subject, "body": body},
        reference=reference,
    )
    logger.info("Email sent to %s", recipient_email)
    return response


def send_low_credit_email(credit: float):
    if credit <= settings.WEB_SEARCH_CREDIT_LIMIT:
        logger.info("Sending credit notification email to admin..")
        send_email(
            recipient_email=settings.ADMIN_EMAIL,
            subject="Web search API credit is low",
            body=f"Current API credit: ${credit} is lower than"
            f"the credit limit of {settings.WEB_SEARCH_CREDIT_LIMIT}.",
            reference="web_search_credit",
            check_if_sent=True,
        )


def send_api_limit_exceed_email(api_count: int):
    if api_count > settings.WEB_SEARCH_API_LIMIT:
        logger.info("Sending api limit notification email to admin..")
        send_email(
            recipient_email=settings.ADMIN_EMAIL,
            subject="Web search API call exceeded limit",
            body=f"Current API count: {api_count} exceeds "
            f"the daily limit of {settings.WEB_SEARCH_API_LIMIT} per day.",
            reference="web_search_api",
            check_if_sent=True,
        )

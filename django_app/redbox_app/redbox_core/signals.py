import logging

from django.conf import settings
from django.db.models import Sum
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from redbox_app.redbox_core.models import MonitorWebSearchResults
from redbox_app.redbox_core.services import notifications as notifications_service

logger = logging.getLogger(__name__)


@receiver(post_save, sender=MonitorWebSearchResults)
def web_search_notification(sender, instance, created, **kwargs):  # noqa: ARG001
    """
    Runs automatically after an Web Search API call is saved.
    Only triggers when the no. of daily call exceed limit.
    """
    logger.info("Checking Kagi credit")
    if created:  # Only check on creation, not updates
        today = timezone.now().date()
        count_today = (
            MonitorWebSearchResults.objects.filter(created_at__date=today).aggregate(Sum("web_search_api_count"))[
                "web_search_api_count__sum"
            ]
            or 0
        )
        if count_today > settings.WEB_SEARCH_API_LIMIT:
            logger.info("Sending email to admin..")
            notifications_service.send_email(
                recipient_email=settings.ADMIN_EMAIL,
                subject="Calls exceeded etc...",
                body=f"Current API count: {count_today} exceeds "
                f"the daily limit of {settings.WEB_SEARCH_API_LIMIT} per day.",
                reference="web_search_api",
                check_if_sent=True,
            )

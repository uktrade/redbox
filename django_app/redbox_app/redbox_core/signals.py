import logging

from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from redbox_app.redbox_core.services import notifications as notifications_service

from .models import MonitorWebSearchResults

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
        count_today = MonitorWebSearchResults.objects.filter(created_at__date=today).count()
        if count_today > settings.WEB_SEARCH_API_LIMIT:
            notifications_service.send_email(
                recipient=settings.ADMIN_EMAIL,
                subject="Calls exceeded etc...",
                body=f"{count_today} exceed the daily limit {settings.WEB_SEARCH_API_LIMIT}",
            )

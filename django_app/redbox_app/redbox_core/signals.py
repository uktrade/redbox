from django.db.models import Sum
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from redbox_app.redbox_core.models import MonitorWebSearchResults
from redbox_app.redbox_core.services.notifications import send_api_limit_exceed_email


@receiver(post_save, sender=MonitorWebSearchResults)
def web_search_notification(sender, instance, created, **kwargs):  # noqa: ARG001
    """
    Runs automatically after an Web Search API call is saved.
    Only triggers when the no. of daily call exceed limit.
    """
    if created:  # Only check on creation, not updates
        today = timezone.now().date()
        count_today = (
            MonitorWebSearchResults.objects.filter(created_at__date=today).aggregate(Sum("web_search_api_count"))[
                "web_search_api_count__sum"
            ]
            or 0
        )
        send_api_limit_exceed_email(api_count=count_today)

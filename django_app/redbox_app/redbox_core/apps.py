import logging

from django.apps import AppConfig
from django.conf import settings
from django.db.models.signals import post_save

from redbox_app.setting_enums import LOCAL_HOSTS

logger = logging.getLogger(__name__)


class RedboxCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "redbox_app.redbox_core"

    def ready(self):
        from redbox_app.redbox_core import signals

        post_save.connect(signals.web_search_notification)

        try:
            from django.contrib.sites.models import Site

            site = Site.objects.get_current()

            if site.domain in LOCAL_HOSTS:
                settings.CURRENT_SITE_URL = f"http://{site.domain}:8080"
            else:
                settings.CURRENT_SITE_URL = f"https://{site.domain}"

        except Exception as e:
            logger.exception("Failed to set CURRENT_SITE_URL", exc_info=e)
            site = None
            settings.CURRENT_SITE_URL = None

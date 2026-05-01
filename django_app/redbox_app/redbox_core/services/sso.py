import logging
from http import HTTPStatus

import requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpRequest

from redbox_app.redbox_core.models import UserSSO, UserSSOAttribute

User = get_user_model()
logger = logging.getLogger(__name__)


def fetch_sso_payload(request: HttpRequest):
    data = None
    error_message = "Failed to fetch SSO data"

    if not request.user.is_authenticated:
        logger.warning("%s: %s", error_message, "Not authenticated")
        return data

    session = request.session
    authbroker_token = session.get("_authbroker_token", {}) or {}
    access_token = authbroker_token.get("access_token")

    if not access_token:
        logger.warning("%s: %s", error_message, "Missing access_token")
        return data

    url = f"{settings.AUTHBROKER_URL}/api/v1/user/me/"
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != HTTPStatus.OK:
            logger.warning("%s: %s %s", error_message, response.status_code, response.text[:500])
            return data

        return response.json()

    except Exception:
        logger.exception("%s: %s", error_message, "Failed to call authbroker endpoint")
        return data


def sync_sso_data(request: HttpRequest) -> bool:
    data = fetch_sso_payload(request)

    if not data:
        return False

    sso, created = UserSSO.objects.get_or_create(user=request.user)

    if created:
        logger.info("Created UserSSO record for: %s", request.user)

    # Emails
    sso.email = data.get("email") or ""
    sso.contact_email = data.get("contact_email") or ""
    sso.email_user_id = data.get("email_user_id") or ""
    sync_related_emails(sso=sso, related_emails=data.get("related_emails", []))

    # Name
    sso.first_name = data.get("first_name") or ""
    sso.last_name = data.get("last_name") or ""

    sso.save()
    logger.info("Synced UserSSO data for: %s", request.user)

    return True


def sync_related_emails(sso: UserSSO, related_emails):
    UserSSOAttribute.objects.filter(sso=sso, type=UserSSOAttribute.AttributeType.RELATED_EMAILS).delete()

    UserSSOAttribute.objects.bulk_create(
        [
            UserSSOAttribute(
                sso=sso,
                type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
                value=email,
            )
            for email in related_emails
        ]
    )

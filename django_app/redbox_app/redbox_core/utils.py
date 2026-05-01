import logging
import uuid
from datetime import date
from http import HTTPStatus

import requests
import waffle
from django import forms
from django.conf import settings
from django.core.exceptions import FieldError
from django.http import Http404, HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.types import RenderTemplateItem

logger = logging.getLogger(__name__)


def get_date_group(on: date) -> str:
    today = timezone.now().date()
    if on == today:
        return "Today"
    return "Previous"


def render_with_oob(templates: list[RenderTemplateItem]) -> HttpResponse:
    """
    Render multiple templates with their own context, request and optional engine.
    Using HTMX Out of bounds swap method.

    Args:
        templates (List[RenderTemplateItem]): A list of objects like:
            {
                "template": str,
                "context": dict,
                "request": HttpRequest,
                "engine": Optional[str]
            }
    Returns:
        HttpResponse: All rendered templates concatenated into a single response.
    """

    html = ""
    for template_item in templates:
        template = template_item["template"]
        context = template_item["context"]
        request = template_item["request"]
        engine = template_item.get("engine", "jinja2")  # Default to jinja2

        html += render_to_string(template, context, request, using=engine)

    return HttpResponse(html)


def save_forms(form_dict: forms.BaseForm | dict[str, forms.BaseForm]):
    results = {}

    if isinstance(form_dict, forms.BaseForm):
        results["form"] = form_dict.is_valid()
        if results["form"]:
            form_dict.save()
    else:
        for name, form in form_dict.items():
            results[name] = form.is_valid()
            if results[name]:
                form.save()

    return results


def resolve_instance(value, model, lookup="pk", raise_404=False):
    if value is None:
        return None
    if isinstance(value, model):
        return value

    try:
        return model.objects.get(**{lookup: value})
    except (ValueError, FieldError, model.DoesNotExist) as err:
        if raise_404:
            msg = f"{model.__name__} not found"
            raise Http404(msg) from err
        msg = f"Cannot resolve {model.__name__} from value: {lookup}='{value}'"
        raise ValueError(msg) from err


def parse_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if not value or value == "None":
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        return None


def user_has_ofi_email(token: str) -> bool:
    if not token:
        return False

    url = f"{settings.AUTHBROKER_URL}/api/v1/user/me/"
    headers = {
        "Authorization": f"Bearer {token}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=5)

        if resp.status_code != HTTPStatus.OK:
            logger.warning("SSO check failed: %s %s", resp.status_code, resp.text[:500])
            return False

        data = resp.json()

        # all emails
        related_emails = data.get("related_emails", []) or []
        main_email = data.get("email") or ""
        contact_email = data.get("contact_email") or ""
        all_emails = [*related_emails, main_email, contact_email]

        return any(email and str(email).lower().endswith("@officeforinvestment.gov.uk") for email in all_emails)

    except Exception:
        logger.exception("Failed to call authbroker endpoint")
        return False


def user_has_invest_lens_access(request) -> bool:
    if not request.user.is_authenticated:
        return False

    if request.user.is_superuser:
        return True

    session = request.session
    authbroker_token = session.get("_authbroker_token", {}) or {}
    access_token = authbroker_token.get("access_token")

    if user_has_ofi_email(access_token):
        return True

    flag_name = flags.ENABLE_INVEST_LENS

    if waffle.flag_is_active(request, flag_name):
        return True

    flag = waffle.get_waffle_flag_model().objects.get(name=flag_name)

    if hasattr(flag, "get_extra_emails"):
        user_email = getattr(request.user, "email", None)
        if user_email:
            extra_emails = flag.get_extra_emails()
            return user_email.strip().lower() in extra_emails

    return False

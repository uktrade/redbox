from datetime import date

from django.core.exceptions import FieldError
from django.http import Http404
from django.utils import timezone


def get_date_group(on: date) -> str:
    today = timezone.now().date()
    if on == today:
        return "Today"
    return "Previous"


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


def strip_domain(email: str) -> str:
    return email.rsplit("@", maxsplit=1)[-1]

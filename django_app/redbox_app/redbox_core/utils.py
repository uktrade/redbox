from datetime import date

from django.utils import timezone


def get_date_group(on: date) -> str:
    today = timezone.now().date()
    if on == today:
        return "Today"
    return "Previous"

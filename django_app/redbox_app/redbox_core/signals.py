from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django.utils import timezone


@receiver(user_logged_in)
def update_last_active(sender, user, request, **kwargs):  # noqa: ARG001
    user.last_active = timezone.now()
    user.save(update_fields=["last_active"])

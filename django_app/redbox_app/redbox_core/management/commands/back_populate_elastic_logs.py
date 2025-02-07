from django.core.management import BaseCommand

from redbox_app.redbox_core.models import ChatMessage


class Command(BaseCommand):
    help = """This is a one-off command to back populate opensearch logs."""

    def handle(self, *args, **kwargs):  # noqa:ARG002
        for chat_message in ChatMessage.objects.all():
            chat_message.log()

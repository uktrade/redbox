import requests
from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "test sso /api/v1/user/me/ endpoint using a session token (get it from dev tools chrome)"

    def add_arguments(self, parser):
        parser.add_argument("session_key", type=str)

    def handle(self, **options):
        session_key = options["session_key"]

        try:
            session = Session.objects.get(session_key=session_key)
            data = session.get_decoded()

            token = data.get("_authbroker_token", {}).get("access_token")

            if not token:
                self.stdout.write(self.style.ERROR("no access key in session"))
                return

            url = f"{settings.AUTHBROKER_URL}/api/v1/user/me/"
            headers = {
                "Authorization": f"Bearer {token}",
            }

            resp = requests.get(url, headers=headers, timeout=3)

            self.stdout.write(self.style.SUCCESS(f"status is {resp.status_code}"))
            self.stdout.write(resp.text)

        except Session.DoesNotExist:
            self.stdout.write(self.style.ERROR("session not found check you got the right cookie from dev tools"))

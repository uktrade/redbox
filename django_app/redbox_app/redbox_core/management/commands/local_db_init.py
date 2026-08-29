from django.core.management import BaseCommand, call_command

from redbox_app.setting_enums import Environment


class NotLocalError(OSError):
    pass


class Command(BaseCommand):
    help = "Initialise db with data from fixtures"

    def handle(self, *_args, **_options):
        if not Environment.is_local:
            raise NotLocalError

        self.stdout.write("Running migrations...")
        call_command("migrate")

        self.stdout.write("Loading tools...")
        call_command("loaddata", "tools")

        self.stdout.write("Loading tool settings...")
        call_command("loaddata", "tool_settings")

        self.stdout.write("Loading agents...")
        call_command("loaddata", "agents")

        self.stdout.write(self.style.SUCCESS("Local db ready!"))

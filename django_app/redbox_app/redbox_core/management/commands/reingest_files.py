import datetime
import logging

from django.core.management import BaseCommand

from redbox_app.redbox_core.models import File, FileTool
from redbox_app.redbox_core.services.documents import reingest_file

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = """This is an ad-hoc command when changes to the AI pipeline (e.g. a new embedding strategy)
    mean we need to regenerate chunks for all the current files.
    """

    def add_arguments(self, parser):
        """sync only to be used for testing"""
        parser.add_argument("start_date", type=datetime.date.fromisoformat)
        parser.add_argument("end_date", type=datetime.date.fromisoformat)

    def handle(self, **options):
        start_date = options["start_date"]
        end_date = options["end_date"]
        end_datetime = end_date.strftime("%Y-%m-%d 23:59:59")

        logger.debug("Reingesting files between '%s' and '%s'", start_date, end_datetime)

        queryset = (
            File.objects.filter(status=File.Status.complete)
            .exclude(created_at__gt=end_datetime)
            .exclude(created_at__lt=start_date)
            .exclude(id__in=FileTool.objects.filter(file_type=FileTool.FileType.ADMIN).values("file__id"))
        )

        if queryset.count() == 0:
            logger.debug("No files found between '%s' and '%s'", start_date, end_datetime)
            return

        for file in queryset:
            reingest_file(file)

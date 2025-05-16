# Custom generated migration will need to be amended for contribution back to i.ai

from django.db import migrations, models


# Custom generated migration will need to be amended for contribution back to i.ai

import logging
logger = logging.getLogger(__name__)


def populate_existing_users_username(apps, schema_editor):
    User = apps.get_model("redbox_core", "User")

    for user in User.objects.all():
        user.username = user.email
        user.save(update_fields=["username"])

class Migration(migrations.Migration):

    dependencies = [
        ("redbox_core", "0071_create_username_field"),
    ]

    operations = [
        migrations.RunPython(populate_existing_users_username, reverse_code=migrations.RunPython.noop),
    ]

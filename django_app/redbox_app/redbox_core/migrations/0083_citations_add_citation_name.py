from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0082_user_uk_or_us_english'),
    ]

    operations = [
        migrations.AddField(
            model_name="citation",
            name="citation_name",
            field=models.TextField(
                blank=True,
                help_text="the reference that the citation refers to - Will be replaced with the link going forward",
                null=True,
            ),
        ),
    ]

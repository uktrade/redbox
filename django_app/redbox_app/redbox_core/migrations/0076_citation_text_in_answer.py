# Generated by Django 5.1.3 on 2024-12-13 14:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        (
            "redbox_core",
            "0075_citation_text_in_answer_alter_user_business_unit_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="citation",
            name="text_in_answer",
            field=models.TextField(
                blank=True,
                help_text="the part of the answer the citation refers too - useful for adding in footnotes",
                null=True,
            ),
        ),
    ]
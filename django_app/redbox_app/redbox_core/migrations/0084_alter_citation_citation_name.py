from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        (
            "redbox_core",
            "0075_citation_text_in_answer_alter_user_business_unit_and_more",
        ),
    ]

    operations = [
        migrations.AlterField(
            model_name="citation",
            name="citation_name",
            field=models.TextField(
                blank=True,
                help_text="the unique name of the citation in the format 'ref_N' where N is a strictly incrementing number starting from 1",
                null=True,
            ),
        ),
    ]

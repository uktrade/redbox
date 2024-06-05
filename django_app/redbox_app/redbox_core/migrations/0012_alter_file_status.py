# Generated by Django 5.0.6 on 2024-06-05 06:46

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("redbox_core", "0011_alter_file_processing_status"),
    ]

    operations = [
        migrations.AlterField(
            model_name="file",
            name="status",
            field=models.CharField(
                choices=[
                    ("uploaded", "Uploaded"),
                    ("parsing", "Parsing"),
                    ("chunking", "Chunking"),
                    ("embedding", "Embedding"),
                    ("indexing", "Indexing"),
                    ("complete", "Complete"),
                    ("unknown", "Unknown"),
                    ("deleted", "Deleted"),
                    ("errored", "Errored"),
                ]
            ),
        ),
    ]
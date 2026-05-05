from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("redbox_core", "0113_migrate_waffle_flags"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                DROP TABLE IF EXISTS waffle_flag_users CASCADE;
                DROP TABLE IF EXISTS waffle_flag_groups CASCADE;
                DROP TABLE IF EXISTS waffle_flag CASCADE;
            """,
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]

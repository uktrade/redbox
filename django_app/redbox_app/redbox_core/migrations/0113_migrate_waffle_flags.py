from django.db import migrations


def copy_flags(apps, schema_editor):
    CustomFlag = apps.get_model("redbox_core", "CustomFlag")

    try:
        WaffleFlag = apps.get_model("waffle", "Flag")
    except LookupError:
        return

    for old in WaffleFlag.objects.all():
        new_flag, _ = CustomFlag.objects.get_or_create(
            name=old.name,
            defaults={
                "everyone": old.everyone,
                "percent": old.percent,
                "testing": old.testing,
                "superusers": old.superusers,
                "staff": old.staff,
                "authenticated": old.authenticated,
                "languages": old.languages,
                "rollout": old.rollout,
                "note": old.note,
                "created": old.created,
                "modified": old.modified,
            },
        )

        if old.groups.exists():
            new_flag.groups.set(old.groups.all())
        if old.users.exists():
            new_flag.users.set(old.users.all())


def reverse_copy_flags(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("redbox_core", "0112_customflag"),
    ]

    operations = [
        migrations.RunPython(copy_flags, reverse_copy_flags),
    ]

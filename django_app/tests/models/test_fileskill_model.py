from collections.abc import Sequence

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    File,
    FileSkill,
    Skill,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_default_file_type(client: Client, alice: User, default_skill: Skill, several_files: Sequence[File]):
    # Given
    client.force_login(alice)

    # When
    for file in several_files:
        file_skill = FileSkill(file=file, skill=default_skill)
        file_skill.save()

    file_skills = FileSkill.objects.filter(skill=default_skill)

    # Then
    assert all(file.file_type == file.FileType.MEMBER for file in file_skills)

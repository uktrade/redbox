from collections.abc import Sequence

import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from redbox_app.redbox_core.models import (
    File,
    FileTool,
    Tool,
)

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_default_file_type(client: Client, alice: User, default_tool: Tool, several_files: Sequence[File]):
    # Given
    client.force_login(alice)

    # When
    for file in several_files:
        file_tool = FileTool(file=file, tool=default_tool)
        file_tool.save()

    file_tools = FileTool.objects.filter(tool=default_tool)

    # Then
    assert all(file.file_type == file.FileType.MEMBER for file in file_tools)

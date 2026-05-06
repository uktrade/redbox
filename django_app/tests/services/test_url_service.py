import uuid

import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import Chat, Tool
from redbox_app.redbox_core.services import url as url_service

User = get_user_model()


@pytest.mark.django_db(transaction=True)
def test_get_chat_url(client: Client, chat_with_alice: Chat, default_tool: Tool):
    # Given
    client.force_login(chat_with_alice.user)
    chat_id = chat_with_alice.id
    slug = default_tool.slug

    # When
    new_chat_link = url_service.get_chat_url()
    new_tool_chat_link = url_service.get_chat_url(slug=slug)
    chat_link = url_service.get_chat_url(chat_id=chat_id)
    tool_chat_link = url_service.get_chat_url(chat_id=chat_id, slug=slug)

    # Then
    assert new_chat_link == reverse("chats")
    assert new_tool_chat_link == reverse("chats", kwargs={"slug": slug})
    assert chat_link == reverse("chats", kwargs={"chat_id": chat_id})
    assert tool_chat_link == reverse("chats", kwargs={"slug": slug, "chat_id": chat_id})


@pytest.mark.django_db(transaction=True)
def test_get_citation_url(client: Client, alice: User, default_tool: Tool):
    # Given
    client.force_login(alice)
    message_id = uuid.uuid4()
    citation_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    slug = default_tool.slug

    # When
    citation_link = url_service.get_citation_url(
        message_id=message_id,
        citation_id=citation_id,
        chat_id=chat_id,
    )
    tool_citation_link = url_service.get_citation_url(
        message_id=message_id,
        citation_id=citation_id,
        chat_id=chat_id,
        slug=slug,
    )

    # Then
    assert citation_link == reverse(
        "citations", kwargs={"chat_id": chat_id, "message_id": message_id}, fragment=str(citation_id)
    )
    assert tool_citation_link == reverse(
        "citations", kwargs={"slug": slug, "chat_id": chat_id, "message_id": message_id}, fragment=str(citation_id)
    )


def test_get_upload_url(default_tool: Tool):
    slug = default_tool.slug

    assert url_service.get_upload_url() == reverse("document-upload")
    assert url_service.get_upload_url(slug=slug) == reverse("document-upload", kwargs={"slug": slug})


def test_get_tool_settings_url(default_tool: Tool):
    slug = default_tool.slug

    assert url_service.get_tool_settings_url(slug) == reverse("tool-settings", kwargs={"slug": slug})


def test_tool_access_rule_urls(default_tool: Tool):
    slug = default_tool.slug
    rule_id = uuid.uuid4()

    assert url_service.get_add_tool_access_rule_url(slug) == reverse("add-tool-access-rule", kwargs={"slug": slug})
    assert url_service.get_edit_tool_access_rule_url(slug, rule_id) == reverse(
        "edit-tool-access-rule", kwargs={"slug": slug, "rule_id": rule_id}
    )
    assert url_service.get_delete_tool_access_rule_url(slug, rule_id) == reverse(
        "delete-tool-access-rule", kwargs={"slug": slug, "rule_id": rule_id}
    )


def test_bulk_and_preview_urls(default_tool: Tool):
    slug = default_tool.slug

    assert url_service.get_bulk_add_user_tool_url(slug) == reverse("bulk-add-user-tool", kwargs={"slug": slug})
    assert url_service.get_affected_users_preview_url(slug) == reverse(
        "tool-access-rule-preview", kwargs={"slug": slug}
    )


def test_get_tool_access_rule_value_input_url():
    assert url_service.get_tool_access_rule_value_input_url() == reverse("tool-access-rule-value-input")


def test_user_tool_urls(default_tool: Tool):
    slug = default_tool.slug
    user_tool_id = uuid.uuid4()

    assert url_service.get_edit_user_tool_row_url(slug, user_tool_id) == reverse(
        "edit-user-tool-row", kwargs={"slug": slug, "user_tool_id": user_tool_id}
    )
    assert url_service.get_edit_user_tool_url(slug, user_tool_id) == reverse(
        "edit-user-tool", kwargs={"slug": slug, "user_tool_id": user_tool_id}
    )

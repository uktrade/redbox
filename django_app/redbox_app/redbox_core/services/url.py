import uuid

from django.urls import reverse


def get_chat_url(chat_id: uuid.UUID | None = None, slug: str | None = None) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug
    if chat_id:
        kwargs["chat_id"] = chat_id

    return reverse("chats", kwargs=kwargs)


def get_citation_url(
    message_id: uuid.UUID, chat_id: uuid.UUID, citation_id: uuid.UUID | None = None, slug: str | None = None
) -> str:
    kwargs = {"message_id": message_id, "chat_id": chat_id}
    fragment = str(citation_id) if citation_id else None

    if slug:
        kwargs["slug"] = slug

    return reverse("citations", kwargs=kwargs, fragment=fragment)


def get_upload_url(slug: str | None = None) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    return reverse("document-upload", kwargs=kwargs)


def get_tool_settings_url(slug: str) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    return reverse("tool-settings", kwargs=kwargs)


def get_add_tool_access_rule_url(slug: str) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    return reverse("add-tool-access-rule", kwargs=kwargs)


def get_edit_tool_access_rule_url(slug: str, rule_id) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    if rule_id:
        kwargs["rule_id"] = rule_id

    return reverse("edit-tool-access-rule", kwargs=kwargs)


def get_delete_tool_access_rule_url(slug: str, rule_id) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    if rule_id:
        kwargs["rule_id"] = rule_id

    return reverse("delete-tool-access-rule", kwargs=kwargs)


def get_bulk_add_user_tool_url(slug: str) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    return reverse("bulk-add-user-tool", kwargs=kwargs)


def get_affected_users_preview_url(slug: str) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    return reverse("tool-access-rule-preview", kwargs=kwargs)


def get_tool_access_rule_value_input_url() -> str:
    return reverse("tool-access-rule-value-input")


def get_edit_user_tool_row_url(slug: str, user_tool_id: uuid.UUID) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    if user_tool_id:
        kwargs["user_tool_id"] = user_tool_id

    return reverse("edit-user-tool-row", kwargs=kwargs)


def get_edit_user_tool_url(slug: str, user_tool_id: uuid.UUID) -> str:
    kwargs = {}

    if slug:
        kwargs["slug"] = slug

    if user_tool_id:
        kwargs["user_tool_id"] = user_tool_id

    return reverse("edit-user-tool", kwargs=kwargs)

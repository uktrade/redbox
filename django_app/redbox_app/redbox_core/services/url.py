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

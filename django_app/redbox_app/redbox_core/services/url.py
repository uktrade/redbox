import uuid

from django.urls import reverse


def get_chat_url(chat_id: uuid.UUID | None = None, skill_slug: str | None = None) -> str:
    kwargs = {}

    if skill_slug:
        kwargs["skill_slug"] = skill_slug
    if chat_id:
        kwargs["chat_id"] = chat_id

    return reverse("chats", kwargs=kwargs)


def get_citation_url(
    message_id: uuid.UUID, citation_id: uuid.UUID, skill_slug: str | None = None, chat_id: uuid.UUID | None = None
) -> str:
    kwargs = {"message_id": message_id}

    if skill_slug:
        kwargs["skill_slug"] = skill_slug
    if chat_id:
        kwargs["chat_id"] = chat_id

    return reverse("citations", kwargs=kwargs, fragment=str(citation_id))

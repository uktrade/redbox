import uuid

from django.urls import reverse


def get_chat_url(chat_id: uuid.UUID | None = None, skill_slug: str | None = None) -> str:
    kwargs = {}

    if skill_slug:
        kwargs["skill_slug"] = skill_slug
    if chat_id:
        kwargs["chat_id"] = chat_id

    return reverse("chats", kwargs=kwargs)

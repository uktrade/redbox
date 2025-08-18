from typing import TypedDict

from django.http import HttpRequest


class RenderTemplateItem(TypedDict):
    template: str
    context: dict
    request: HttpRequest
    engine: str | None

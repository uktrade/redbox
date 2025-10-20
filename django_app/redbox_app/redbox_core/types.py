from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from django.http import HttpRequest

# Valid icon path mappings
FILE_EXTENSION_MAPPING: dict[str, str] = {
    ".eml": "mail",
    ".html": "html",
    ".json": "file-json",
    ".md": "text-snippet",
    ".msg": "mail",
    ".rst": "wysiwyg",
    ".rtf": "text-snippet",
    ".txt": "text-snippet",
    ".xml": "code",
    ".csv": "csv",
    ".doc": "docs",
    ".docx": "docs",
    ".epub": "menu-book",
    ".odt": "odt",
    ".pdf": "pcture-as-pdf",
    ".ppt": "co-present",
    ".pptx": "co-present",
    ".tsv": "tsv",
    ".xlsx": "table-view",
    ".htm": "html",
}

APPROVED_FILE_EXTENSIONS = list(FILE_EXTENSION_MAPPING.keys())


class RenderTemplateItem(TypedDict):
    template: str
    context: dict
    request: HttpRequest
    engine: str | None


@dataclass
class TabConfig:
    id: str
    title: str
    template: str
    get_context: Callable[[HttpRequest], dict] = lambda _: {}
    handle_post: Callable[[HttpRequest], Any] = lambda _: None


class TabRegistry:
    def __init__(self, tabs: list[TabConfig]):
        self._tabs = tabs
        self._lookup = {tab.id: tab for tab in tabs}

    def __iter__(self):
        return iter(self._tabs)

    def __getitem__(self, key: str) -> TabConfig:
        return self._lookup[key]

    def get(self, key: str, default=None):
        return self._lookup.get(key, default)

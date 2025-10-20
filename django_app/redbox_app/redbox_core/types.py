from typing import TypedDict

from django.http import HttpRequest


class RenderTemplateItem(TypedDict):
    template: str
    context: dict
    request: HttpRequest
    engine: str | None


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

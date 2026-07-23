import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import UUID
from zoneinfo import ZoneInfo


from django.http import HttpRequest
from django.utils import timezone

if TYPE_CHECKING:
    from redbox_app.redbox_core.models import Chat


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

    def get_context(self, request: HttpRequest) -> list[dict]:
        """
        Returns template-ready tab context.

        :param request:
        :type request: HttpRequest
        :return:
        :rtype: list[dict]
        """
        return [
            {
                "id": tab.id,
                "title": tab.title,
                "template": tab.template,
                "context": tab.get_context(request),
            }
            for tab in self._tabs
        ]


@dataclass(frozen=True)
class UIFragment:
    id: str
    template: str


FRAGMENTS = {
    "chat-feed": UIFragment(
        id="chat-feed",
        template="chat/chat_feed.html",
    ),
    "chat-cta": UIFragment(
        id="chat-cta",
        template="chat/cta.html",
    ),
    "conversations": UIFragment(
        id="conversations",
        template="side_panel/conversations.html",
    ),
    "your-documents": UIFragment(
        id="your-documents",
        template="side_panel/your_documents.html",
    ),
}


STREAM_REF_RE = re.compile(
    r"[\[\(\{<]\s*ref_(\d+)\s*[\]\)\}>]|\bref_(\d+)\b",
    re.IGNORECASE,
)


@dataclass
class CitationMap:
    """
    Streaming-only citation state.
    Maps ref_n tokens -> stable footnote numbers.
    """

    counter: int = 1
    map: dict[str, int] = field(default_factory=dict)

    def resolve(self, ref: str) -> int:
        """
        Get or assign a footnote number for a ref token.
        """
        if ref not in self.map:
            self.map[ref] = self.counter
            self.counter += 1
        return self.map[ref]


class StreamingTextBuffer:
    """
    Holds the last incomplete token between streamed chunks.
    """

    def __init__(self):
        self.tail = ""

    def process(self, chunk: str) -> str:
        text = self.tail + chunk

        # If the text ends with whitespace, everything is safe.
        if text and text[-1].isspace():
            self.tail = ""
            return text

        # No whitespace at all yet.
        if not any(c.isspace() for c in text):
            self.tail = text
            return ""

        # Split on the last whitespace boundary.
        split_at = max(
            text.rfind(" "),
            text.rfind("\n"),
            text.rfind("\t"),
        )

        safe = text[: split_at + 1]
        self.tail = text[split_at + 1 :]

        return safe

    def flush(self) -> str:
        text = self.tail
        self.tail = ""
        return text


@dataclass(slots=True)
class MessageResponse:
    chat_message_id: str

    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class MessageCreatedResponse(MessageResponse):
    chat_message_role: str
    html: str


@dataclass(slots=True)
class MessageUpdateResponse(MessageResponse):
    sr_text: str
    html: str


@dataclass(slots=True)
class MessageCompletedResponse(MessageResponse):
    html: str
    title: str
    session_id: str


@dataclass(slots=True)
class MessageActivityResponse(MessageResponse):
    activity_event_message: str


type ChatStreamEvent = Literal[
    "message_created",
    "message_update",
    "message_completed",
    "message_activity",
    "auth_expired",
    "route",
    "error",
    "session-id",
]


@dataclass
class FilterChat:
    id: UUID
    name: str
    tool: str | None
    last_message_datetime: datetime
    local_last_message_datetime: datetime

    @classmethod
    def from_chat(cls, chat: "Chat", tz: ZoneInfo) -> "FilterChat":
        last_message_datetime = chat.latest_message_date
        return cls(
            id=chat.id,
            name=chat.name,
            tool=chat.tool.name if chat.tool else None,
            last_message_datetime=last_message_datetime,
            local_last_message_datetime=timezone.localtime(last_message_datetime, tz),
        )


@dataclass
class GroupedChats:
    today: list[FilterChat] = field(default_factory=list)
    yesterday: list[FilterChat] = field(default_factory=list)
    previous_7_days: list[FilterChat] = field(default_factory=list)
    previous_30_days: list[FilterChat] = field(default_factory=list)
    previous_year: list[FilterChat] = field(default_factory=list)
    over_a_year: list[FilterChat] = field(default_factory=list)

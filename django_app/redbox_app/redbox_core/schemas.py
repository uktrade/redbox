from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID
from zoneinfo import ZoneInfo

from django.utils import timezone

if TYPE_CHECKING:
    from redbox_app.redbox_core.models import Chat


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

from enum import Enum, StrEnum


class ChatRoute(StrEnum):
    search = "search"
    gadget = "agent"
    chat = "chat"
    chat_with_docs = "summarise"
    chat_with_docs_map_reduce = "chat/documents/large"
    newroute = "newroute"
    summarise = "summarise"
    tabular = "tabular"
    legislation = "legislation"


class ErrorRoute(StrEnum):
    files_too_large = "error/files_too_large"


class ToolEnum(Enum):
    search = "search"
    summarise = "summarise"


class DecisionEnum(Enum):
    modify = "modify"
    reject = "reject"
    approve = "approve"
    more_info = "more_info"

from enum import StrEnum


class ChatRoute(StrEnum):
    search = "search"
    gadget = "agent"
    chat = "chat"
    chat_with_docs = "summarise"
    chat_with_docs_map_reduce = "chat/documents/large"


class ErrorRoute(StrEnum):
    files_too_large = "error/files_too_large"

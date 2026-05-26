from enum import StrEnum
from typing import List
from pydantic import BaseModel, Field
from redbox.api.wrapper import SensitiveValue
from redbox.models.file import ChunkCreatorType
from langchain_core.messages import AIMessage, ToolCall
from langchain.tools import StructuredTool
from concurrent.futures import Future


# Input


class FutureResultType(StrEnum):
    """Future result type of tool call."""

    UNKNOWN = "unknown"
    SYNC = "sync"
    MCP_ASYNC = "mcp_async"


class ToolCallRequest:
    """ToolRunner grouped tool call objects."""

    class Base(BaseModel):
        future_result_type: FutureResultType = FutureResultType.UNKNOWN

    class Sync(Base):
        future_result_type: FutureResultType = FutureResultType.SYNC
        tool_call: ToolCall

    class MCPAsync(BaseModel):
        future_result_type: FutureResultType = FutureResultType.MCP_ASYNC

        mcp_server: str
        access_token: SensitiveValue
        creator_type: ChunkCreatorType
        tool_calls: list[ToolCall]

        class Config:
            arbitrary_types_allowed = True


class GroupedToolCallRequests(BaseModel):
    sync_calls: List[ToolCallRequest.Sync]
    mcp_async_server_calls: List[ToolCallRequest.MCPAsync]

    class Config:
        arbitrary_types_allowed = True


class ValidatedToolCall(BaseModel):
    name: str
    tool: StructuredTool
    args: dict

    class Config:
        arbitrary_types_allowed = True


# Output


class ToolCallResult:
    """Result of tool execution."""

    class Base(BaseModel):
        tool_name: str = Field(description="The name of the executed tool.")
        metadata: dict = Field(default={}, description="Metadata from tool execution.")

    class Success(Base):
        """Successful result of tool execution."""

        response: AIMessage = Field(description="AIMessage response generated from tool execution.")

    class Failure(Base):
        """Failed result of tool execution."""

        error: str = Field(default=None, description="Error from tool execution.")


class SubmittedToolCallRequest(BaseModel):
    name: str
    result_type: FutureResultType
    future: Future
    future_args: dict
    metadata: dict

    class Config:
        arbitrary_types_allowed = True


class SubmittedToolCallRequests(BaseModel):
    futures: list[SubmittedToolCallRequest]
    failures: list[ToolCallResult.Failure]


class Result(BaseModel):
    """Result of parallel tool execution."""

    results: List[ToolCallResult.Success] = Field(
        default=[], description="List of responses generated from tool executions."
    )
    failures: List[ToolCallResult.Failure] = Field(default=[], description="List of failures from tool executions.")

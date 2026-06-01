from enum import StrEnum
from typing import List
from pydantic import BaseModel, Field
from redbox.api.wrapper import SensitiveValue
from redbox.models.file import ChunkCreatorType
from langchain_core.messages import AIMessage, ToolCall
from concurrent.futures import Future


# -- ToolCall Requests & Results --


class FutureResultType(StrEnum):
    """Future result type of tool call request."""

    SYNC = "sync"
    MCP_ASYNC = "mcp_async"


class ToolCallRequest:
    """Tool call request objects."""

    class Base(BaseModel):
        future_result_type: FutureResultType

    class Sync(Base):
        """Sync tool call requests."""

        future_result_type: FutureResultType = FutureResultType.SYNC
        tool_call: ToolCall

    class MCPAsync(BaseModel):
        """MCP Async tool call requests."""

        future_result_type: FutureResultType = FutureResultType.MCP_ASYNC

        mcp_server: str
        access_token: SensitiveValue
        creator_type: ChunkCreatorType
        tool_calls: list[ToolCall]

        class Config:
            arbitrary_types_allowed = True


class RunRequestCalls(BaseModel):
    """Grouped tool call request objects."""

    sync: List[ToolCallRequest.Sync]
    mcp_async: List[ToolCallRequest.MCPAsync]

    class Config:
        arbitrary_types_allowed = True


class ToolCallResult:
    """Result of single tool execution."""

    class Base(BaseModel):
        tool_name: str = Field(description="The name of the executed tool.")
        metadata: dict = Field(default={}, description="Metadata from tool execution.")

    class Success(Base):
        """Successful result of single tool execution."""

        response: AIMessage = Field(description="AIMessage response generated from tool execution.")

    class Failure(Base):
        """Failed result of single tool execution."""

        error: str = Field(description="Error from tool execution.")


class ParsedRunRequest(BaseModel):
    calls: RunRequestCalls
    failures: list[ToolCallResult.Failure]


class SubmittedToolCallRequest(BaseModel):
    """Submitted tool call future."""

    name: str
    result_type: FutureResultType
    future: Future
    future_args: dict
    metadata: dict

    class Config:
        arbitrary_types_allowed = True


class SubmittedRunRequest(BaseModel):
    """Submitted tool call futures and failures."""

    futures: list[SubmittedToolCallRequest]
    failures: list[ToolCallResult.Failure]


# -- ToolRunner Result --


class Result(BaseModel):
    """Result of parallel tool execution."""

    results: List[ToolCallResult.Success] = Field(
        default=[], description="List of responses generated from tool executions."
    )
    failures: List[ToolCallResult.Failure] = Field(default=[], description="List of failures from tool executions.")

from enum import StrEnum
from typing import List
from pydantic import BaseModel, ConfigDict, Field
from redbox.api.wrapper import SensitiveValue
from redbox.models.file import ChunkCreatorType
from langchain_core.messages import AIMessage, ToolCall


# Input


class FutureResultType(StrEnum):
    """Future result type of tool call."""

    UNKNOWN = "unknown"
    SYNC = "sync"
    MCP_ASYNC = "mcp_async"


class ToolRunnerToolCall:
    """ToolRunner tool call."""

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

        model_config = ConfigDict(arbitrary_types_allowed=True)


# Output


class ToolResult:
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


class ToolRunnerResult(BaseModel):
    """Result of parallel tool execution."""

    results: List[ToolResult.Success] = Field(
        default=[], description="List of responses generated from tool executions."
    )
    failures: List[ToolResult.Failure] = Field(default=[], description="List of failures from tool executions.")

import logging
from typing import Tuple, Optional
from pydantic import BaseModel, ConfigDict
from redbox.api.wrapper import SensitiveValue
from redbox.models.file import ChunkCreatorType
from langchain_core.messages import ToolCall
from langchain.tools import StructuredTool

log = logging.getLogger(__name__)


class RunnerToolCall:
    class Sync(BaseModel):
        tool_call: ToolCall

    class MCPAsync(BaseModel):
        mcp_server: str
        access_token: SensitiveValue
        creator_type: ChunkCreatorType
        tool_calls: list[ToolCall]

        model_config = ConfigDict(arbitrary_types_allowed=True)


def group_tool_calls(
    tool_calls: list[ToolCall], tools: list[StructuredTool]
) -> Tuple[RunnerToolCall.Sync, RunnerToolCall.MCPAsync]:
    sync_tools: list[RunnerToolCall.Sync] = []
    mcp_async_tools: dict[str, RunnerToolCall.MCPAsync] = {}

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        selected_tool: Optional[StructuredTool] = next((tool for tool in tools if tool.name == tool_name), None)

        if selected_tool.func and not selected_tool.coroutine:
            sync_tools.append(RunnerToolCall.Sync(tool_call=tool_call))
        else:
            mcp_url = selected_tool.metadata["url"]
            creator_type = selected_tool.metadata["creator_type"]
            sso_access_token = selected_tool.metadata["sso_access_token"]

            if mcp_url:
                if mcp_url in mcp_async_tools.keys():
                    mcp_async_tools[mcp_url].tool_calls.append(tool_call)
                else:
                    mcp_async_tools[mcp_url] = RunnerToolCall.MCPAsync(
                        mcp_server=mcp_url,
                        access_token=sso_access_token,
                        creator_type=creator_type,
                        tool_calls=[tool_call],
                    )

    return sync_tools, mcp_async_tools.values()

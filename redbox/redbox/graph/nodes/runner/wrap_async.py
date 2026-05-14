import logging
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from openai import BaseModel
from langchain_core.messages import ToolCall
from redbox.models.file import ChunkCreatorType
from langchain.tools import StructuredTool
from redbox.api.format import format_mcp_tool_response


log = logging.getLogger(__name__)


def _get_mcp_headers(sso_access_token: str | None = None) -> dict[str, str]:
    if not sso_access_token:
        log.warning("_get_mcp_headers - Datahub MCP sso_access_token is None")
        return {}
    token = sso_access_token.strip()
    if not token:
        return {}
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


class MCPToolInvoke(BaseModel):
    name: str
    tool: StructuredTool
    calls: list[ToolCall]


def wrap_async_tools(invokes: list[MCPToolInvoke]):
    """
    Returns a synchronous function that properly wraps an async tool

    Args:
        tool_name: The name of the tool to invoke

    Returns:
        A function that synchronously executes the async tool
    """

    INIT_TIMEOUT, TOOL_LOADING_TIMEOUT, INVOKE_TIMEOUT = 10, 15, 60

    def wrapper():
        tool = invokes[0].tool

        # get mcp tool url
        mcp_url = tool.metadata["url"]
        creator_type = tool.metadata["creator_type"]

        try:
            sso_access_token = tool.metadata["sso_access_token"].get()
        except Exception as e:
            log.error(f"wrap_async_tool - Failed to retrieve sso_access_token: {e}")
            raise

        if not sso_access_token:
            log.error("wrap_async_tool - MCP sso_access_token is None")

        headers = _get_mcp_headers(sso_access_token)

        async def run_tools():
            try:
                async with streamablehttp_client(mcp_url, headers=headers or None) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        init_result = await asyncio.wait_for(session.initialize(), timeout=INIT_TIMEOUT)
                        server_name = init_result.serverInfo.name
                        server_version = init_result.serverInfo.version

                        # Get tools
                        tools = await asyncio.wait_for(load_mcp_tools(session), timeout=TOOL_LOADING_TIMEOUT)

                        results: dict[str, list] = {}
                        for invoke in invokes:
                            log.info(
                                f"wrap_async_tool - Calling tool '{invoke.name}' on MCP server {server_name}@{server_version}"
                            )

                            selected_tool = next((t for t in tools if t.name == invoke.name), None)
                            if not selected_tool:
                                raise ValueError(f"tool with name '{invoke.name}' not found")

                            log.warning(f"wrap_async_tool - tool found with name '{invoke.name}'")

                            for call in invoke.calls:
                                # remove intermediate step argument if it is not required by tool
                                args = call.get("args", {})
                                if "is_intermediate_step" not in selected_tool.args_schema.get(
                                    "required", []
                                ) and args.get("is_intermediate_step"):
                                    args.pop("is_intermediate_step")
                                    log.warning(f"wrap_async_tool - updated args: {call}")

                                log.warning(f"wrap_async_tool - args '{args}'")
                                result = await asyncio.wait_for(selected_tool.ainvoke(args), timeout=INVOKE_TIMEOUT)

                                log.warning(f"wrap_async_tool - MCP Tool '{invoke.name}' result: {result}")

                                if creator_type == ChunkCreatorType.datahub:
                                    log.warning(
                                        f"wrap_async_tool - Formatting MCP tool response for creator_type='{creator_type}'"
                                    )
                                    results[invoke.name] = results.get(invoke.name, []) + [
                                        (
                                            format_mcp_tool_response(
                                                tool_response=result,
                                                creator_type=creator_type,
                                            ),
                                            args,
                                        )
                                    ]
                                else:
                                    log.warning(
                                        f"wrap_async_tool - Returning raw MCP tool response for creator_type='{creator_type}'"
                                    )
                                    results[invoke.name] = results.get(invoke.name, [])

                        return results

            except asyncio.TimeoutError:
                log.error("wrap_async_tool - Tools timed out")
                raise

            except asyncio.CancelledError:
                log.warning("wrap_async_tool - Tools cancelled")
                raise

            except Exception as e:
                log.error(
                    f"wrap_async_tool - MCP execution failed at '{mcp_url}': {e}",
                    exc_info=True,
                )
                raise

        try:
            return asyncio.run(run_tools())
        except Exception as e:
            log.error(f"wrap_async_tool - Unhandled error running MCP tools '{mcp_url}': {e}", exc_info=True)
            raise

    return wrapper


def wrap_async_tool(tool, tool_name):
    """
    Returns a synchronous function that properly wraps an async tool

    Args:
        tool_name: The name of the tool to invoke

    Returns:
        A function that synchronously executes the async tool
    """

    INIT_TIMEOUT, TOOL_LOADING_TIMEOUT, INVOKE_TIMEOUT = 10, 15, 60

    def wrapper(args):
        # get mcp tool url
        mcp_url = tool.metadata["url"]
        creator_type = tool.metadata["creator_type"]

        try:
            sso_access_token = tool.metadata["sso_access_token"].get()
        except Exception as e:
            log.error(f"wrap_async_tool - Failed to retrieve sso_access_token: {e}")
            raise

        if not sso_access_token:
            log.error("wrap_async_tool - MCP sso_access_token is None")

        headers = _get_mcp_headers(sso_access_token)

        async def run_tool():
            try:
                async with streamablehttp_client(mcp_url, headers=headers or None) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        init_result = await asyncio.wait_for(session.initialize(), timeout=INIT_TIMEOUT)
                        server_name = init_result.serverInfo.name
                        server_version = init_result.serverInfo.version

                        log.info(
                            f"wrap_async_tool - Calling tool '{tool_name}' on MCP server {server_name}@{server_version}"
                        )

                        # Get tools
                        tools = await asyncio.wait_for(load_mcp_tools(session), timeout=TOOL_LOADING_TIMEOUT)

                        selected_tool = next((t for t in tools if t.name == tool_name), None)
                        if not selected_tool:
                            raise ValueError(f"tool with name '{tool_name}' not found")

                        # remove intermediate step argument if it is not required by tool
                        if "is_intermediate_step" not in selected_tool.args_schema.get("required", []) and args.get(
                            "is_intermediate_step"
                        ):
                            args.pop("is_intermediate_step")
                            log.warning(f"wrap_async_tool - updated args: {args}")

                        log.warning(f"wrap_async_tool - tool found with name '{tool_name}'")
                        log.warning(f"wrap_async_tool - args '{args}'")
                        result = await asyncio.wait_for(selected_tool.ainvoke(args), timeout=INVOKE_TIMEOUT)

                        log.warning(f"wrap_async_tool - MCP Tool '{tool_name}' result: {result}")

                        if creator_type == ChunkCreatorType.datahub:
                            log.warning(
                                f"wrap_async_tool - Formatting MCP tool response for creator_type='{creator_type}'"
                            )
                            return format_mcp_tool_response(
                                tool_response=result,
                                creator_type=creator_type,
                            )

                        log.warning(
                            f"wrap_async_tool - Returning raw MCP tool response for creator_type='{creator_type}'"
                        )
                        return result

            except asyncio.TimeoutError:
                log.error(f"wrap_async_tool - Tool '{tool_name}' timed out")
                raise

            except asyncio.CancelledError:
                log.warning(f"wrap_async_tool - Tool '{tool_name}' cancelled")
                raise

            except Exception as e:
                log.error(
                    f"wrap_async_tool - MCP execution failed for '{tool_name}' at '{mcp_url}': {e}",
                    exc_info=True,
                )
                raise

        try:
            return asyncio.run(run_tool())
        except Exception as e:
            log.error(f"wrap_async_tool - Unhandled error running tool '{tool_name}': {e}", exc_info=True)
            raise

    return wrapper

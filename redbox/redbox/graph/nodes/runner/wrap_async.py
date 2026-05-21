import logging
import asyncio
from typing import List, Any
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from redbox.graph.nodes.runner.tool_calls import RunnerToolCall
from redbox.models.file import ChunkCreatorType
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


async def execute_mcp_tools_async(mcp_input: RunnerToolCall.MCPAsync) -> List[Any]:
    """
    Execute multiple MCP tools in a single session.

    Args:
        mcp_input: MCPAsync object containing server URL, credentials, and tool calls

    Returns:
        List of results from each tool call in order
    """
    INIT_TIMEOUT, TOOL_LOADING_TIMEOUT, INVOKE_TIMEOUT = 10, 15, 60

    mcp_url = mcp_input.mcp_server
    creator_type = mcp_input.creator_type

    try:
        sso_access_token = mcp_input.access_token.get()
    except Exception as e:
        log.error(f"execute_mcp_tools_async - Failed to retrieve sso_access_token: {e}")
        raise

    if not sso_access_token:
        log.error("execute_mcp_tools_async - MCP sso_access_token is None")
        raise ValueError("MCP sso_access_token is required")

    headers = _get_mcp_headers(sso_access_token)
    results = []

    try:
        async with streamablehttp_client(mcp_url, headers=headers or None) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                # Initialize the connection once
                init_result = await asyncio.wait_for(session.initialize(), timeout=INIT_TIMEOUT)
                server_name = init_result.serverInfo.name
                server_version = init_result.serverInfo.version

                log.info(f"execute_mcp_tools_async - Connected to MCP server {server_name}@{server_version}")

                # Load tools once
                tools = await asyncio.wait_for(load_mcp_tools(session), timeout=TOOL_LOADING_TIMEOUT)

                # Create a lookup map for faster access
                tools_map = {t.name: t for t in tools}

                # Execute each tool call in sequence
                for i, tool_call in enumerate(mcp_input.tool_calls):
                    tool_name = tool_call.tool_name
                    args = tool_call.args.copy()

                    log.info(
                        f"execute_mcp_tools_async - Executing tool {i + 1}/{len(mcp_input.tool_calls)}: '{tool_name}'"
                    )

                    selected_tool = tools_map.get(tool_name)
                    if not selected_tool:
                        error_msg = f"Tool '{tool_name}' not found on server"
                        log.error(f"execute_mcp_tools_async - {error_msg}")
                        results.append({"error": error_msg, "tool_name": tool_name})
                        continue

                    # Remove intermediate step argument if not required by tool
                    if "is_intermediate_step" not in selected_tool.args_schema.get("required", []) and args.get(
                        "is_intermediate_step"
                    ):
                        args.pop("is_intermediate_step")
                        log.debug("execute_mcp_tools_async - Removed is_intermediate_step from args")

                    log.debug(f"execute_mcp_tools_async - Invoking '{tool_name}' with args: {args}")

                    try:
                        result = await asyncio.wait_for(selected_tool.ainvoke(args), timeout=INVOKE_TIMEOUT)

                        log.info(f"execute_mcp_tools_async - Tool '{tool_name}' completed successfully")

                        # Format result if needed
                        if creator_type == ChunkCreatorType.datahub:
                            log.debug(
                                f"execute_mcp_tools_async - Formatting response for creator_type='{creator_type}'"
                            )
                            formatted_result = format_mcp_tool_response(
                                tool_response=result,
                                creator_type=creator_type,
                            )
                            results.append(formatted_result)
                        else:
                            results.append(result)

                    except asyncio.TimeoutError:
                        error_msg = f"Tool '{tool_name}' timed out after {INVOKE_TIMEOUT}s"
                        log.error(f"execute_mcp_tools_async - {error_msg}")
                        results.append({"error": error_msg, "tool_name": tool_name})

                    except Exception as e:
                        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                        log.error(f"execute_mcp_tools_async - {error_msg}", exc_info=True)
                        results.append({"error": error_msg, "tool_name": tool_name})

    except asyncio.TimeoutError as e:
        log.error(f"execute_mcp_tools_async - Session initialization/setup timed out: {e}")
        raise

    except asyncio.CancelledError:
        log.warning("execute_mcp_tools_async - Session cancelled")
        raise

    except Exception as e:
        log.error(
            f"execute_mcp_tools_async - MCP session failed for '{mcp_url}': {e}",
            exc_info=True,
        )
        raise

    return results


def execute_mcp_tools(mcp_input: RunnerToolCall.MCPAsync) -> List[Any]:
    """
    Synchronous wrapper for executing multiple MCP tools.

    Args:
        mcp_input: MCPAsync object containing server URL, credentials, and tool calls

    Returns:
        List of results from each tool call in order
    """

    def wrapper():
        try:
            return asyncio.run(execute_mcp_tools_async(mcp_input))
        except Exception as e:
            log.error(f"execute_mcp_tools - Unhandled error: {e}", exc_info=True)
            raise

    return wrapper

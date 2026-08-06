import asyncio
import logging
from typing import Any, Dict, List, Tuple

import httpx
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import redbox.graph.nodes.runner.models as tr_models
from redbox.api.format import format_mcp_tool_response
from redbox.models.file import ChunkCreatorType

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


async def execute_mcp_tools_async(
    mcp_input: tr_models.ToolCallRequest.MCPAsync,
) -> Tuple[Dict[str, Any], List[tr_models.ToolCallResult.Failure]]:
    """
    Execute multiple MCP tools in a single session.

    Args:
        mcp_input: MCPAsync object containing server URL, credentials, and tool calls

    Returns: Tuple[Dict[str, Any], List[tr_models.ToolCallResult.Failure]]
        Tuple of tool results and failures.

        Result[0] - a dictionary of tool results, with tool_call ID as key and result as value.
        Result[1] - a list of tool call failures.
    """
    results: Dict[str, Any] = {}
    failures: List[tr_models.ToolCallResult.Failure] = []

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
    try:
        async with httpx.AsyncClient(headers=headers or {}) as http_client:
            async with streamable_http_client(
                mcp_url,
                http_client=http_client,
            ) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    # Initialize the connection once
                    init_result = await asyncio.wait_for(
                        session.initialize(),
                        timeout=INIT_TIMEOUT,
                    )
                    server_name = init_result.serverInfo.name
                    server_version = init_result.serverInfo.version

                    log.info(f"execute_mcp_tools_async - Connected to MCP server {server_name}@{server_version}")

                    # Load tools once
                    tools = await asyncio.wait_for(
                        load_mcp_tools(session),
                        timeout=TOOL_LOADING_TIMEOUT,
                    )

                    # Create a lookup map for faster access
                    tools_map = {t.name: t for t in tools}

                    # Execute each tool call in sequence
                    for i, tool_call in enumerate(mcp_input.tool_calls):
                        tool_call_id = tool_call.get("id")
                        tool_name = tool_call.get("name")
                        args = tool_call.get("args").copy()

                        log.info(
                            f"execute_mcp_tools_async - Executing tool {i + 1}/{len(mcp_input.tool_calls)}: '{tool_name}'"
                        )

                        selected_tool = tools_map.get(tool_name)
                        if not selected_tool:
                            error_msg = f"Tool '{tool_name}' not found on server"
                            log.error(f"execute_mcp_tools_async - {error_msg}")

                            failures.append(
                                tr_models.ToolCallResult.Failure(
                                    tool_name=tool_name,
                                    metadata={"tool_args": args},
                                    error=f"MCP Async tool '{tool_name}' not found on server '{mcp_url}'",
                                )
                            )
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
                                results[tool_call_id] = formatted_result

                            else:
                                results[tool_call_id] = result

                        except asyncio.TimeoutError:
                            error_msg = f"Tool '{tool_name}' timed out after {INVOKE_TIMEOUT}s"
                            log.error(f"execute_mcp_tools_async - {error_msg}")

                            failures.append(
                                tr_models.ToolCallResult.Failure(
                                    tool_name=tool_name,
                                    metadata={"tool_args": args},
                                    error=f"MCP Async tool '{tool_name}' timed out on server '{mcp_url}' - asyncio.TimeoutError",
                                )
                            )

                        except Exception as e:
                            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                            log.error(f"execute_mcp_tools_async - {error_msg}", exc_info=True)

                            failures.append(
                                tr_models.ToolCallResult.Failure(
                                    tool_name=tool_name,
                                    metadata={"tool_args": args},
                                    error=f"MCP Async tool '{tool_name}' got an unknown exception server '{mcp_url}' - {e}",
                                )
                            )

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

    return results, failures


def execute_mcp_tools(
    mcp_input: tr_models.ToolCallRequest.MCPAsync,
) -> Tuple[Dict[str, Any], List[tr_models.ToolCallResult.Failure]]:
    """
    Synchronous wrapper for executing multiple MCP tools.

    Args:
        mcp_input: MCPAsync object containing server URL, credentials, and tool calls

    Returns: Tuple[Dict[str, Any], List[tr_models.ToolCallResult.Failure]]
        Tuple of tool results and failures.

        Result[0] - a dictionary of tool results, with tool_call ID as key and result as value.
        Result[1] - a list of tool call failures.
    """

    def wrapper():
        try:
            return asyncio.run(execute_mcp_tools_async(mcp_input))
        except Exception as e:
            log.error(
                f"execute_mcp_tools - Unhandled error when calling MCP server '{mcp_input.mcp_server}': {e}",
                exc_info=True,
            )
            raise

    return wrapper

import logging
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

import redbox.graph.nodes.runner.exceptions as tool_exceptions

log = logging.getLogger(__name__)


def wrap_async_tool(tool, tool_name, tool_timeout):
    """
    Returns a synchronous function that properly wraps an async tool

    Args:
        tool_name: The name of the tool to invoke

    Returns:
        A function that synchronously executes the async tool
    """

    def wrapper(args):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        mcp_url = tool.metadata.get("url")
        if not mcp_url:
            raise tool_exceptions.ToolInputValidationError(
                tool_name=tool_name,
                field="metadata.url",
                value=mcp_url,
                reason="MCP tool is missing a 'url' in its metadata",
            )

        # Define the async operation
        async def run_tool():
            try:
                # tool need to be executed within the connection context manager
                async with streamablehttp_client(mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        try:
                            await session.initialize()
                        except Exception as e:
                            raise tool_exceptions.ToolFailedError(
                                tool_name=tool_name,
                                cause=e,
                                stderr=f"MCP session initialisation failed against {mcp_url}",
                            ) from e

                        # Get tools
                        try:
                            tools = await load_mcp_tools(session)
                        except Exception as e:
                            raise tool_exceptions.ToolFailedError(
                                tool_name=tool_name,
                                cause=e,
                                stderr="Failed to load tools from MCP session",
                            ) from e

                        selected_tool = next((t for t in tools if t.name == tool_name), None)
                        if not selected_tool:
                            raise tool_exceptions.ToolNotFoundError(
                                tool_name=tool_name, available_tools=[t.name for t in tools]
                            )

                        # Strip is_intermediate_step if the tool schema does not declare it
                        required_args = (selected_tool.args_schema or {}).get("required", [])
                        if "is_intermediate_step" not in required_args and args.get("is_intermediate_step"):
                            args.pop("is_intermediate_step")
                            log.warning(f"updated args: {args}")

                        # Validate required args are present
                        missing = [field for field in required_args if field not in args]
                        if missing:
                            raise tool_exceptions.ToolInputValidationError(
                                tool_name=tool_name,
                                reason=f"missing required argument(s): {', '.join(missing)}",
                            )

                        log.warning(f"tool found with name '{tool_name}'")
                        log.warning(f"args '{args}'")

                        try:
                            result = await selected_tool.ainvoke(args)
                        except asyncio.TimeoutError:
                            raise tool_exceptions.ToolTimeoutError(tool_name=tool_name, timeout_seconds=0)
                        except Exception as e:
                            raise tool_exceptions.ToolFailedError(tool_name=tool_name, cause=e)

                        log.warning(f"result: {result}")
                        return result

            except (
                tool_exceptions.ToolNotFoundError,
                tool_exceptions.ToolInputValidationError,
                tool_exceptions.ToolTimeoutError,
                tool_exceptions.ToolFailedError,
            ):
                raise  # re-raise typed exceptions as-is
            except asyncio.TimeoutError as e:
                raise tool_exceptions.ToolTimeoutError(tool_name=tool_name, timeout_seconds=tool_timeout) from e
            except ConnectionError as e:
                raise tool_exceptions.ToolFailedError(
                    tool_name=tool_name,
                    cause=e,
                    stderr=f"Failed to connect to MCP server at {mcp_url}",
                ) from e
            except Exception as e:
                raise tool_exceptions.ToolFailedError(tool_name=tool_name, cause=e) from e

        try:
            # Run the async function and return its result
            return loop.run_until_complete(run_tool())
        finally:
            # Clean up resources
            loop.close()

    return wrapper

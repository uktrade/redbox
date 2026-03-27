import logging
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

# from redbox.graph.nodes.sends import _get_mcp_headers
from redbox.models.file import ChunkCreatorType
from redbox.api.format import format_mcp_tool_response


log = logging.getLogger(__name__)


def _get_mcp_headers(sso_access_token: str | None = None) -> dict[str, str]:
    if not sso_access_token:
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

    def wrapper(args):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # get mcp tool url
        mcp_url = tool.metadata["url"]
        creator_type = tool.metadata["creator_type"]
        sso_access_token = tool.metadata["sso_access_token"].get()
        headers = _get_mcp_headers(sso_access_token)

        try:
            # Define the async operation
            async def run_tool():
                # tool need to be executed within the connection context manager
                async with streamablehttp_client(mcp_url, headers=headers or None) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        init_result = await session.initialize()
                        server_name = init_result.serverInfo.name
                        server_version = init_result.serverInfo.version

                        log.info(f"Calling tool '{tool_name}' on MCP server {server_name}@{server_version}")

                        # Get tools
                        tools = await load_mcp_tools(session)

                        selected_tool = next((t for t in tools if t.name == tool_name), None)
                        if not selected_tool:
                            raise ValueError(f"tool with name '{tool_name}' not found")

                        # remove intermediate step argument if it is not required by tool
                        if "is_intermediate_step" not in selected_tool.args_schema["required"] and args.get(
                            "is_intermediate_step"
                        ):
                            args.pop("is_intermediate_step")
                            log.warning(f"updated args: {args}")

                        log.warning(f"tool found with name '{tool_name}'")
                        log.warning(f"args '{args}'")
                        result = await selected_tool.ainvoke(args)

                        log.warning(f"MCP Tool '{tool_name}' result: {result}")

                        if creator_type == ChunkCreatorType.datahub:
                            log.warning(f"Formatting MCP tool response for creator_type='{creator_type}'")
                            return format_mcp_tool_response(
                                tool_response=result,
                                creator_type=creator_type,
                            )

                        log.warning(f"Returning raw MCP tool response for creator_type='{creator_type}'")
                        return result

            # Run the async function and return its result
            return loop.run_until_complete(run_tool())
        finally:
            # Clean up resources
            loop.close()

    return wrapper

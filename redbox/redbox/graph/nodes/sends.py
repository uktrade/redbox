import logging
import threading
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.constants import Send

from redbox.models.chain import DocumentState, RedboxState, TaskStatus
from redbox.api.format import format_mcp_tool_response

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from redbox.models.file import ChunkCreatorType
from redbox.graph.nodes.runner.runner import ToolRunner

log = logging.getLogger(__name__)


def _copy_state(state: RedboxState, **updates) -> RedboxState:
    updated_model = state.model_copy(update=updates, deep=True)
    return updated_model


def build_document_group_send(target: str) -> Callable[[RedboxState], list[Send]]:
    """Builds Sends per document group."""

    def _group_send(state: RedboxState) -> list[Send]:
        group_send_states: list[RedboxState] = [
            _copy_state(
                state,
                documents=DocumentState(groups={document_group_key: document_group}),
            )
            for document_group_key, document_group in state.documents.groups.items()
        ]
        return [Send(node=target, arg=state) for state in group_send_states]

    return _group_send


def build_document_chunk_send(target: str) -> Callable[[RedboxState], list[Send]]:
    """Builds Sends per individual document"""

    def _chunk_send(state: RedboxState) -> list[Send]:
        chunk_send_states: list[RedboxState] = [
            _copy_state(
                state,
                documents=DocumentState(groups={document_group_key: {document_key: document}}),
            )
            for document_group_key, document_group in state.documents.groups.items()
            for document_key, document in document_group.items()
        ]
        return [Send(node=target, arg=state) for state in chunk_send_states]

    return _chunk_send


def build_tool_send(target: str) -> Callable[[RedboxState], list[Send]]:
    """Builds Sends per tool call."""

    def _tool_send(state: RedboxState) -> list[Send]:
        tool_send_states: list[RedboxState] = [
            _copy_state(
                state,
                messages=[AIMessage(content=state.last_message.content, tool_calls=[tool_call])],
            )
            for tool_call in state.last_message.tool_calls
        ]
        return [Send(node=target, arg=state) for state in tool_send_states]

    return _tool_send


def run_with_timeout(func, args, timeout):
    """Run a a function with a timeout and return its result or None if it times out or fails.
    This function can be used to set a timeout for tool execution"""
    result = [None]
    exception = [None]
    completed = [False]

    def target():
        try:
            result[0] = func(args)
        except Exception as e:
            exception[0] = e
        finally:
            completed[0] = True

    thread = threading.Thread(target=target)
    thread.daemon = True  # The thread will exit when the main program exits
    thread.start()
    thread.join(timeout)  # applying timeout constraint

    if not completed[0]:  # if it times out
        log.warning(f"Tool execution timed out after {timeout} seconds")
        return None
    if exception[0]:  # if the tool fails
        log.warning(f"Tool execution failed: {str(exception[0])}")
        return None
    return result[0]


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


def run_tools_parallel(
    ai_msg,
    tools,
    state,
    parallel_timeout=60,
    is_loop=False,
) -> list[AIMessage] | None:

    if not ai_msg.tool_calls:
        log.warning("No tool calls detected. Returning agent content.")
        return ai_msg.content

    try:
        max_workers = min(10, len(ai_msg.tool_calls))
        runner = ToolRunner(
            tools=tools,
            state=state,
            max_workers=max_workers,
            is_loop=is_loop,
            parallel_timeout=parallel_timeout,
        )

        return runner.run(tool_calls=ai_msg.tool_calls)

    except Exception as e:
        log.warning(
            f"Unexpected error in parallel tool execution: {str(e)}",
            exc_info=True,
        )
        return None


def no_dependencies(dependencies: list[str], plan) -> bool:
    """
    Check if all dependencies are completed
    return True if no dependencies i.e. []
    return True if depencies are completed or failed e.g. ['completed']
    return False if there are dependencies in pending, scheduled, running e.g. ['pending']
    """
    if dependencies:
        deps = [
            dep
            for dep in dependencies
            if plan.get_task_status(dep) in [TaskStatus.PENDING, TaskStatus.SCHEDULED, TaskStatus.RUNNING]
        ]
        return len(deps) == 0
    else:
        return True


def sending_task_to_agent(state: RedboxState):
    plan = state.agent_plans
    if plan:
        # sending tasks that have no dependencies
        task_send_states: list[RedboxState] = []
        for task in plan.tasks:
            if no_dependencies(task.dependencies, plan) and (task.status == TaskStatus.PENDING):
                # update status
                task.status = TaskStatus.SCHEDULED
                state.agent_plans.update_task_status(task.id, TaskStatus.SCHEDULED)
                task_send_states += [
                    (
                        task.agent.value,
                        _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]),
                    )
                ]
                log.warning(f"Sending task: {task.id} to agent {task.agent}")
        return [Send(node=target, arg=state) for target, state in task_send_states]

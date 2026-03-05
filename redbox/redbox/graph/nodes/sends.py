import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed, Future
from typing import Callable, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolCall
from langgraph.constants import Send

from redbox.models.chain import DocumentState, RedboxState, TaskStatus
from redbox.graph.nodes import exceptions

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
import json

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


# def run_with_timeout(func, args, timeout):
#     """Run a a function with a timeout and return its result or None if it times out or fails.
#     This function can be used to set a timeout for tool execution"""
#     result = [None]
#     exception = [None]
#     completed = [False]

#     def target():
#         try:
#             result[0] = func(args)
#         except Exception as e:
#             exception[0] = e
#         finally:
#             completed[0] = True

#     thread = threading.Thread(target=target)
#     thread.daemon = True  # The thread will exit when the main program exits
#     thread.start()
#     thread.join(timeout)  # applying timeout constraint

#     if not completed[0]:  # if it times out
#         log.warning(f"Tool execution timed out after {timeout} seconds")
#         return None
#     if exception[0]:  # if the tool fails
#         log.warning(f"Tool execution failed: {str(exception[0])}")
#         return None
#     return result[0]


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

        try:
            # Define the async operation
            async def run_tool():
                # tool need to be executed within the connection context manager
                async with streamablehttp_client(mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        await session.initialize()
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
                        log.warning("result")
                        log.warning(result)
                        return result

            # Run the async function and return its result
            return loop.run_until_complete(run_tool())
        finally:
            # Clean up resources
            loop.close()

    return wrapper


def run_tools_parallel(ai_msg, tools, state, parallel_timeout=60, is_loop=False):
    run_id = str(uuid4())[:8]
    log_stub = f"[run_tools_parallel run_id='{run_id}']"
    log.warning(f"{log_stub} Starting tool execution.")

    if not ai_msg.tool_calls:
        # No tool calls
        log.warning(f"{log_stub} No tool calls detected. Returning agent content.")
        return ai_msg.content

    log.warning(
        f"{log_stub} {len(ai_msg.tool_calls)} tool call(s) detected: {[tc.get('name') for tc in ai_msg.tool_calls]}"
    )

    max_workers = min(10, len(ai_msg.tool_calls))
    log.warning(f"{log_stub} Creating ThreadPoolExecutor(max_workers={max_workers})")

    # -- helpers --

    def submit_tool_future(tool_call: ToolCall) -> tuple[Future, dict] | None:
        # Find the matching tool by name
        tool_name = tool_call.get("name")
        selected_tool = next((tool for tool in tools if tool.name == tool_name), None)

        if selected_tool is None:
            raise exceptions.ToolNotFoundError(tool_name=tool_name, available_tools=[tool.name for tool in tools])
            # log.warning(f"{log_stub} Warning: No tool found for {tool_name}")
            # return None

        # Get arguments and submit the tool invocation
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            raise exceptions.ToolInputValidationError(
                tool_name=tool_name,
                field="args",
                value=args,
                reason=f"expected dict, got {type(args).__name__!r}",
            )

        log.warning(f"args: {args}")
        is_intermediate_step = "False"

        try:
            # check if tool is sync (not async). the sync tool should have sync function defined and no async coroutine
            if selected_tool.func and not selected_tool.coroutine:
                args["state"] = state
                future = executor.submit(selected_tool.invoke, args)

            # for async mcp tools
            else:
                # capture any intermediate step value decided by LLM
                if is_loop:
                    is_intermediate_step = args.get("is_intermediate_step", "False")
                    log.warning(f"intermediate step: {is_intermediate_step}")
                future = executor.submit(wrap_async_tool(selected_tool, tool_name), args)
        except Exception as e:
            raise exceptions.ToolFailedError(tool_name=tool_name, cause=e) from e

        return future, {"name": tool_name, "intermediate_step": is_intermediate_step}

    def parse_tool_future(future: Future, tool_timeout: float) -> Optional[AIMessage]:
        result = None

        future_tool_name = futures[future]["name"]
        is_intermediate_step = futures[future]["intermediate_step"]

        try:
            response = future.result()
        except TimeoutError as e:
            raise exceptions.ToolTimeoutError(future_tool_name, timeout_seconds=tool_timeout) from e
        except Exception as e:
            raise exceptions.ToolFailedError(future_tool_name, cause=e) from e

        log.warning(f"{log_stub} This is what I got from tool '{future_tool_name}': {response}")

        if response is None:
            raise exceptions.ToolFailedError(
                future_tool_name, stderr="Tool returned None — may have failed or timed out"
            )

        log.warning(f"{log_stub} {future_tool_name} response not None")

        if (not is_loop and isinstance(response, str)) or (
            is_loop and isinstance(response, tuple)
        ):  # when is_loop=True, result output should be a Tuple
            log.warning(f"{log_stub} {future_tool_name} my non-transformed response: {response}")
            result = response

        elif is_loop and isinstance(response, str):
            try:
                result_dict = json.loads(response)
                is_empty = result_dict.get("total") == 0  # Check if response has no records
                log.warning(f"{log_stub} {future_tool_name} is_empty {is_empty}")
            except json.JSONDecodeError:
                is_empty = response in ["", "None", "[]"]  # Check if response is an empty string/None/empty array

            status = "fail" if is_empty else "pass"  # Set status based on emptiness

            if is_empty:
                log.warning(f"No records  returned from {future_tool_name} tool")
                response = "Error message: Empty response"

            # Create transformed response and append to responses
            transformed_response = (response, status, is_intermediate_step)
            log.warning("my transformed response")
            log.warning(transformed_response)
            result = transformed_response

        else:
            raise exceptions.ToolOutputValidationError(
                future_tool_name,
                reason=(
                    f"expected str (is_loop=False) or tuple (is_loop=True), "
                    f"got {type(response).__name__!r} — is_loop={is_loop}"
                ),
            )

        raw_res = response
        if isinstance(raw_res, tuple):
            raw_res = raw_res[0]

        if not raw_res or not isinstance(raw_res, str) or not raw_res.strip():
            raise exceptions.ToolOutputValidationError(
                future_tool_name,
                reason=f"empty or whitespace-only response body: {repr(raw_res)}",
            )

        return AIMessage(result)

    # Dict to store futures and related metadata
    futures = {}
    try:
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tool invocations to the executor
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call.get("name")
                try:
                    res = submit_tool_future(tool_call=tool_call)
                    if res is None:
                        continue

                    future, args = res
                    futures[future] = args

                except exceptions.ToolNotFoundError as e:
                    log.warning(f"{log_stub} '{tool_name}' not found: {e}")

                except exceptions.ToolInputValidationError as e:
                    log.warning(f"{log_stub} '{tool_name}' invalid input: {e}")

                except exceptions.ToolRunnerError as e:
                    log.warning(f"{log_stub} '{tool_name}' failed to submit: {e}")

            # Collect responses as tools complete
            responses = []
            failed_tools: list[str] = []
            for future in as_completed(futures.keys(), timeout=parallel_timeout):
                future_tool_name = futures[future]["name"]

                try:
                    response = parse_tool_future(future=future, tool_timeout=parallel_timeout)
                    if response is not None:
                        responses.append(response)

                except exceptions.ToolTimeoutError as e:
                    log.warning(f"{log_stub} '{future_tool_name}' timed out after {e.timeout_seconds:.1f}s.")
                    failed_tools.append(future_tool_name)

                except exceptions.ToolOutputValidationError as e:
                    log.warning(f"{log_stub} '{future_tool_name}' returned an invalid output: {e}")
                    failed_tools.append(future_tool_name)

                except exceptions.ToolFailedError as e:
                    log.warning(f"{log_stub} '{future_tool_name}' failed: {e}")
                    if e.cause:
                        log.debug(f"{log_stub} '{future_tool_name}' cause: {e.cause}", exc_info=e.cause)
                    failed_tools.append(future_tool_name)

                except exceptions.ToolRunnerError as e:
                    # Catch-all for any other typed tool errors (e.g. ToolPermissionError, ToolRateLimitError)
                    log.warning(f"{log_stub} '{future_tool_name}' tool runner error: {e}")
                    failed_tools.append(future_tool_name)

            if failed_tools:
                log.warning(f"{log_stub} {len(failed_tools)} tool(s) failed: {', '.join(failed_tools)}")

            if not responses:
                log.warning(
                    f"{log_stub} Every tool execution has failed or timed out. "
                    f"Failed tools: {', '.join(failed_tools) or 'unknown'}."
                )
                return None

            log.warning(
                f"{log_stub} Completed. Successful: {len(responses)}, "
                f"Failed: {len(failed_tools)}. Responses: {responses}"
            )
            return responses

    except TimeoutError:
        log.warning(f"{log_stub} Global parallel tool execution timed out after {parallel_timeout} seconds.")
        return None
    except Exception as e:
        log.warning(f"{log_stub} Unexpected error in parallel tool execution: {str(e)}", exc_info=True)
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
                    (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
                ]
                log.warning(f"Sending task: {task.id} to agent {task.agent}")

        return [Send(node=target, arg=state) for target, state in task_send_states]

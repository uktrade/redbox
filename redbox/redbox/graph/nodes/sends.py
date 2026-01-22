import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Callable
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.constants import Send

from redbox.models.chain import DocumentState, RedboxState, TaskStatus

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


def run_tools_parallel(ai_msg, tools, state, parallel_timeout=60, per_tool_timeout=60, result_timeout=60):
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

    # Dict to store futures and related metadata
    futures = {}

    try:
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tool invocations to the executor
            for tool_call in ai_msg.tool_calls:
                # Find the matching tool by name
                tool_name = tool_call.get("name")
                selected_tool = next((tool for tool in tools if tool.name == tool_name), None)

                if selected_tool is None:
                    log.warning(f"{log_stub} Warning: No tool found for {tool_name}")
                    continue

                # Get arguments and submit the tool invocation
                args = tool_call.get("args", {})
                args["state"] = state

                future = executor.submit(run_with_timeout, selected_tool.invoke, args, per_tool_timeout)
                futures[future] = {"name": tool_name}

            # Collect responses as tools complete
            responses = []
            for future in as_completed(futures.keys(), timeout=parallel_timeout):
                future_tool_name = futures[future]["name"]

                try:
                    response = future.result(timeout=result_timeout)
                    log.warning(f"{log_stub} This is what I got from tool '{future_tool_name}': {response}")
                    if response is not None:  # if response is not None, meaning tool did not fail or timeout
                        responses.append(AIMessage(response))
                    else:
                        log.warning(f"{future_tool_name} Tool has failed or timed out")
                        continue

                    raw_res = response
                    if isinstance(raw_res, tuple):
                        raw_res = raw_res[0]

                    if not raw_res or not raw_res.strip():
                        log.warning(
                            f"{log_stub} '{future_tool_name}' Tool returned empty/whitespace response: {repr(raw_res)}"
                        )

                except TimeoutError:
                    log.warning(
                        f"{log_stub} '{future_tool_name}' Results retrieval from tool timed out after {result_timeout} seconds."
                    )

                except Exception as e:
                    log.warning(f"{log_stub} '{future_tool_name}' Tool invocation error: {e}")

            if responses:
                log.warning(
                    f"{log_stub} Completed. Successful parallel tool responses: {len(responses)}. Responses: {responses}"
                )
                return responses
            else:
                log.warning(
                    f"{log_stub} Every tool execution has failed or timed out after {per_tool_timeout} seconds."
                )
                return None

    except TimeoutError:
        log.warning(f"{log_stub} Global parallel tool execution timed out after {parallel_timeout} seconds.")
        return None
    except Exception as e:
        log.warning(f"{log_stub} Unexpected error in parallel tool execution: {str(e)}", exc_info=True)
        return None


def sending_task_to_agent(state: RedboxState):
    plan = state.agent_plans
    if plan:
        # sending tasks that have no dependencies
        task_send_states: list[RedboxState] = [
            (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            for task in plan.tasks
            if not task.dependencies and (task.status == TaskStatus.PENDING)
        ]
        return [Send(node=target, arg=state) for target, state in task_send_states]


def sending_specific_task_to_agent(task, state: RedboxState):
    # sending tasks that have no dependencies
    for task in state.agent_plans.tasks:
        if task.status == TaskStatus.PENDING:
            task_send_states: list[RedboxState] = [
                (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            ]
            return [Send(node=target, arg=state) for target, state in task_send_states]

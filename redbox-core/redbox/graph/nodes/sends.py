import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.constants import Send

from redbox.models.chain import DocumentState, RedboxState

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


def run_tools_parallel(ai_msg, tools, state, timeout=30):
    # Create a list to store futures
    futures = []
    try:
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(10, len(ai_msg.tool_calls))) as executor:
            # Submit tool invocations to the executor
            for tool_call in ai_msg.tool_calls:
                # Find the matching tool by name
                selected_tool = next((tool for tool in tools if tool.name == tool_call.get("name")), None)

                if selected_tool is None:
                    log.warning(f"Warning: No tool found for {tool_call.get('name')}")
                    continue

                # Get arguments and submit the tool invocation
                args = tool_call.get("args", {})
                args["state"] = state
                future = executor.submit(selected_tool.invoke, args)
                futures.append(future)

            # Collect responses as tools complete
            responses = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    response = future.result()
                    responses.append(AIMessage(response))
                except Exception as e:
                    print(f"Tool invocation error: {e}")

            return responses
    except TimeoutError:
        log.error(f"Tool execution timed out after {timeout} seconds")
        responses.append(AIMessage("Some tools timed out during execution"))
    except Exception as e:
        log.error(f"Unexpected error in tool execution: {str(e)}", exc_info=True)
        responses.append(AIMessage(f"Execution error: {str(e)}"))


def sending_task_to_agent(state: RedboxState):
    plan = state.agent_plans
    if plan:
        task_send_states: list[RedboxState] = [
            (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            for task in plan.tasks
        ]
        return [Send(node=target, arg=state) for target, state in task_send_states]

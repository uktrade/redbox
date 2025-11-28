import asyncio
import logging
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


async def run_tools_parallel(ai_msg, tools, state, timeout=60):
    if not ai_msg.tool_calls:
        return ai_msg.content

    async def run_tool(tool, args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, tool.invoke, args)

    tasks = []

    for tc in ai_msg.tool_calls:
        name = tc.get("name")
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            log.warning(f"No tool found: {name}")
            continue

        args = tc.get("args", {})
        args["state"] = state

        tasks.append(run_tool(tool, args))

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout)
        return [AIMessage(r) for r in results]

    except asyncio.TimeoutError:
        log.warning("Tool execution timed out")
        return []

    except Exception as e:
        log.warning(f"Tool execution error: {e}")
        return []


def sending_task_to_agent(state: RedboxState):
    plan = state.agent_plans
    if plan:
        task_send_states: list[RedboxState] = [
            (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            for task in plan.tasks
        ]
        return [Send(node=target, arg=state) for target, state in task_send_states]

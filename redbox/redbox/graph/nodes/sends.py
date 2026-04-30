import logging
from typing import Callable

from langchain_core.messages import AIMessage
from langgraph.constants import Send

from redbox.models.chain import DocumentState, RedboxState, TaskStatus


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

        try:
            result = runner.run(tool_calls=ai_msg.tool_calls)
            return result.responses
        finally:
            runner.executor.shutdown(wait=True)

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

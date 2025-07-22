import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
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


async def run_tools_parallel(ai_msg, tools, state, timeout=30):
    """Execute tool calls in parallel, supporting both synchronous and asynchronous tools with a timeout."""
    responses = []
    tools_by_name = {tool.name: tool for tool in tools}

    async def execute_tool(tool, args):
        try:
            result = None
            log.debug(f"Executing tool {tool.name} with args: {args}")
            # Check if the tool has an async coroutine
            is_async = hasattr(tool, "coroutine") and asyncio.iscoroutinefunction(tool.coroutine)
            if is_async:
                # Unpack args according to the tool's schema if it's a StructuredTool
                if hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        schema_instance = tool.args_schema(**args)
                        tool_args = schema_instance.dict()
                    except Exception as e:
                        log.error(f"Error parsing args for {tool.name}: {str(e)}")
                        return AIMessage(content=f"Error: Invalid arguments for {tool.name}")
                else:
                    tool_args = args
                result = await asyncio.wait_for(tool.ainvoke(tool_args), timeout=timeout)
            else:
                async def sync_tool_wrapper():
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        return await asyncio.get_event_loop().run_in_executor(executor, lambda: tool.invoke(args))
                result = await asyncio.wait_for(sync_tool_wrapper(), timeout=timeout)

            log.debug(f"Raw tool result for {tool.name}: {result}")
            if getattr(tool, "response_format", None) == "content_and_artifact":
                if not isinstance(result, tuple) or len(result) != 2:
                    log.error(f"Tool {tool.name} returned invalid content_and_artifact format: {type(result)}")
                    return AIMessage(content=f"Error: Invalid response format from {tool.name}")
                content, artifact = result
                if not isinstance(content, str) or not isinstance(artifact, list):
                    log.error(f"Tool {tool.name} returned invalid types: content={type(content)}, artifact={type(artifact)}")
                    return AIMessage(content=f"Error: Invalid response types from {tool.name}")
                for doc in artifact:
                    if not isinstance(doc, Document) or not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
                        log.error(f"Invalid Document in {tool.name} artifact: {doc}")
                        return AIMessage(content=f"Error: Invalid Document format from {tool.name}")
                return AIMessage(content=content, additional_kwargs={"artifact": artifact})
            return result
        except asyncio.TimeoutError:
            log.warning(f"Tool {tool.name} timed out after {timeout} seconds")
            return AIMessage(content=f"Error: Tool {tool.name} timed out")
        except Exception as e:
            log.warning(f"Tool {tool.name} invocation error: {str(e)}", exc_info=True)
            return AIMessage(content=f"Error: {str(e)}")

    tasks = []
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call.get("name")
        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            log.warning(f"No tool found for {tool_name}")
            responses.append(AIMessage(content=f"Error: Tool {tool_name} not found"))
            continue
        args = tool_call.get("args", {})
        args["state"] = state
        tasks.append(execute_tool(selected_tool, args))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, AIMessage):
                responses.append(result)
            elif isinstance(result, Exception):
                log.warning(f"Unexpected error in tool execution: {str(result)}", exc_info=True)
                responses.append(AIMessage(content=f"Error: {str(result)}"))
            elif isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                log.debug(f"Processing content_and_artifact result: content={content[:100]}..., artifact={artifact}")
                responses.append(AIMessage(content=content, additional_kwargs={"artifact": artifact}))
            else:
                log.warning(f"Unexpected result type from tool execution: {type(result)}")
                responses.append(AIMessage(content=f"Error: Unexpected result type {type(result)}"))

    if not responses:
        responses.append(ai_msg)

    log.debug(f"Final tool responses: {responses}")
    return responses


def sending_task_to_agent(state: RedboxState):
    plan = state.agent_plans
    if plan:
        task_send_states: list[RedboxState] = [
            (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            for task in plan.tasks
        ]
        return [Send(node=target, arg=state) for target, state in task_send_states]

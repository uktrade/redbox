import json
import logging
import re
import textwrap
import time
from collections.abc import Callable
from functools import reduce
from random import uniform
from typing import Any, Iterable
from uuid import uuid4

from botocore.exceptions import EventStreamError
from langchain.schema import StrOutputParser
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.types import Command

from redbox.chains.activity import log_activity
from redbox.chains.components import (
    get_basic_metadata_retriever,
    get_chat_llm,
    get_structured_response_with_citations_parser,
    get_structured_response_with_planner_parser,
    get_tokeniser,
)
from redbox.chains.parser import ClaudeParser
from redbox.chains.runnables import CannedChatLLM, build_llm_chain, chain_use_metadata, create_chain_agent
from redbox.graph.nodes.sends import run_tools_parallel
from redbox.models import ChatRoute
from redbox.models.chain import (
    AgentTask,
    FeedbackEvalDecision,
    DocumentState,
    MultiAgentPlan,
    PromptSet,
    RedboxState,
    RequestMetadata,
)
from redbox.models.graph import ROUTE_NAME_TAG, RedboxActivityEvent, RedboxEventType
from redbox.models.prompts import (
    PLANNER_FORMAT_PROMPT,
    PLANNER_PROMPT,
    PLANNER_QUESTION_PROMPT,
    REPLAN_PROMPT,
    USER_FEEDBACK_EVAL_PROMPT,
)
from redbox.models.settings import get_settings
from redbox.transform import combine_agents_state, combine_documents, flatten_document_state

log = logging.getLogger(__name__)
re_keyword_pattern = re.compile(r"@(\w+)")


# Patterns: functions that build processes

## Core patterns


def build_retrieve_pattern(
    retriever: VectorStoreRetriever,
    structure_func: Callable[[list[Document]], DocumentState],
    final_source_chain: bool = False,
) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a function that uses state["request"] and state["text"] to set state["documents"].

    Uses structure_func to order the retriever documents for the state.
    """
    return RunnableParallel({"documents": retriever | structure_func})


def build_chat_pattern(
    prompt_set: PromptSet,
    tools: list[StructuredTool] | None = None,
    final_response_chain: bool = False,
) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that uses the state to set state["text"].

    If tools are supplied, can also set state["tool_calls"].
    """

    def _chat(state: RedboxState) -> dict[str, Any]:
        llm = get_chat_llm(state.request.ai_settings.chat_backend, tools=tools)
        return build_llm_chain(
            prompt_set=prompt_set,
            llm=llm,
            final_response_chain=final_response_chain,
        ).invoke(state)

    return _chat


def build_merge_pattern(
    prompt_set: PromptSet,
    tools: list[StructuredTool] | None = None,
    final_response_chain: bool = False,
) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that uses state.request and state.documents to return one item in state.documents.

    When combined with chunk send, will replace each Document with what's returned from the LLM.

    When combined with group send, with combine all Documents and use the metadata of the first.

    When used without a send, the first Document received defines the metadata.

    If tools are supplied, can also set state["tool_calls"].
    """
    tokeniser = get_tokeniser()
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 1.5

    @RunnableLambda
    def _merge(state: RedboxState) -> dict[str, Any]:
        llm = get_chat_llm(state.request.ai_settings.chat_backend, tools=tools)

        if not state.documents.groups:
            return {"documents": None}

        flattened_documents = flatten_document_state(state.documents)
        merged_document = reduce(lambda left, right: combine_documents(left, right), flattened_documents)

        merge_state = RedboxState(
            request=state.request,
            documents=DocumentState(
                groups={merged_document.metadata["uuid"]: {merged_document.metadata["uuid"]: merged_document}}
            ),
        )

        retries = 0
        while retries < MAX_RETRIES:
            try:
                merge_response = build_llm_chain(
                    prompt_set=prompt_set, llm=llm, final_response_chain=final_response_chain
                ).invoke(merge_state)

                # if reaches a successful citation, exit the loop
                break

            except EventStreamError as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    raise e
                wait_time = BACKOFF_FACTOR**retries + uniform(0, 1)
                time.sleep(wait_time)

        merged_document.page_content = merge_response["messages"][-1].content
        request_metadata = merge_response["metadata"]
        merged_document.metadata["token_count"] = tokeniser(merged_document.page_content)

        group_uuid = next(iter(state.documents.groups or {}), uuid4())
        document_uuid = merged_document.metadata.get("uuid", uuid4())

        # Clear old documents, add new one
        document_state = state.documents.groups.copy()

        for group in document_state:
            for document in document_state[group]:
                document_state[group][document] = None

        document_state[group_uuid][document_uuid] = merged_document

        return {"documents": DocumentState(groups=document_state), "metadata": request_metadata}

    return _merge


def build_stuff_pattern(
    prompt_set: PromptSet,
    output_parser: Runnable = None,
    format_instructions: str = "",
    tools: list[StructuredTool] | None = None,
    final_response_chain: bool = False,
    additional_variables: dict = {},
    summary_multiagent_flag: bool = False,
) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that uses state.request and state.documents to set state.messages.

    If tools are supplied, can also set state.tool_calls.
    """

    @RunnableLambda
    def _stuff(state: RedboxState) -> dict[str, Any]:
        llm = get_chat_llm(state.request.ai_settings.chat_backend, tools=tools)

        events = [
            event
            for event in build_llm_chain(
                prompt_set=prompt_set,
                llm=llm,
                output_parser=output_parser,
                format_instructions=format_instructions,
                final_response_chain=final_response_chain,
                additional_variables=additional_variables,
                summary_multiagent_flag=summary_multiagent_flag,
            ).stream(state)
        ]
        return sum(events, {})

    return _stuff


## Utility patterns


def build_set_route_pattern(route: ChatRoute) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that sets state["route_name"]."""

    def _set_route(state: RedboxState) -> dict[str, Any]:
        return {"route_name": route}

    return RunnableLambda(_set_route).with_config(tags=[ROUTE_NAME_TAG])


def build_set_self_route_from_llm_answer(
    conditional: Callable[[str], bool],
    true_condition_state_update: dict,
    false_condition_state_update: dict,
    final_route_response: bool = True,
) -> Runnable[RedboxState, dict[str, Any]]:
    """A Runnable which sets the route based on a conditional on state['text']"""

    @RunnableLambda
    def _set_self_route_from_llm_answer(state: RedboxState):
        llm_response = state.last_message.content
        if conditional(llm_response):
            return true_condition_state_update
        else:
            return false_condition_state_update

    runnable = _set_self_route_from_llm_answer
    if final_route_response:
        runnable = _set_self_route_from_llm_answer.with_config(tags=[ROUTE_NAME_TAG])
    return runnable


def build_passthrough_pattern() -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that uses state["request"] to set state["text"]."""

    @RunnableLambda
    def _passthrough(state: RedboxState) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=state.request.question)],
        }

    return _passthrough


def build_set_text_pattern(text: str, final_response_chain: bool = False) -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that can arbitrarily set state["messages"] to a value."""
    llm = CannedChatLLM(messages=[AIMessage(content=text)])
    _llm = llm.with_config(tags=["response_flag"]) if final_response_chain else llm

    @RunnableLambda
    def _set_text(state: RedboxState) -> dict[str, Any]:
        set_text_chain = _llm | StrOutputParser()

        return {"messages": state.messages + [HumanMessage(content=set_text_chain.invoke(text))]}

    return _set_text


def build_set_metadata_pattern() -> Runnable[RedboxState, dict[str, Any]]:
    """A Runnable which calculates the static request metadata from the state"""

    @RunnableLambda
    def _set_metadata_pattern(state: RedboxState):
        flat_docs = flatten_document_state(state.documents)
        return {
            "metadata": RequestMetadata(
                selected_files_total_tokens=sum(map(lambda d: d.metadata.get("token_count", 0), flat_docs)),
                number_of_selected_files=len(state.request.s3_keys),
            )
        }

    return _set_metadata_pattern


def build_error_pattern(text: str, route_name: str | None) -> Runnable[RedboxState, dict[str, Any]]:
    """A Runnable which sets text and route to record an error"""

    @RunnableLambda
    def _error_pattern(state: RedboxState):
        return build_set_text_pattern(text, final_response_chain=True).invoke(state) | build_set_route_pattern(
            route_name
        ).invoke(state)

    return _error_pattern


# Raw processes: functions that need no building


def clear_documents_process(state: RedboxState) -> dict[str, Any]:
    if documents := state.documents:
        return {"documents": DocumentState(groups={group_id: None for group_id in documents.groups})}


def report_sources_process(state: RedboxState) -> None:
    """A Runnable which reports the documents in the state as sources."""
    if citations_state := state.citations:
        dispatch_custom_event(RedboxEventType.on_citations_report, citations_state)
    elif document_state := state.documents:
        dispatch_custom_event(RedboxEventType.on_source_report, flatten_document_state(document_state))


def empty_process(state: RedboxState) -> None:
    return None


def build_log_node(message: str) -> Runnable[RedboxState, dict[str, Any]]:
    """A Runnable which logs the current state in a compact way"""

    @RunnableLambda
    def _log_node(state: RedboxState):
        log.info(
            json.dumps(
                {
                    "user_uuid": str(state.request.user_uuid),
                    "document_metadata": {
                        group_id: {doc_id: d.metadata for doc_id, d in group_documents.items()}
                        for group_id, group_documents in state.documents.group
                    },
                    "messages": (textwrap.shorten(state.last_message.content, width=32, placeholder="...")),
                    "route": state.route_name,
                    "message": message,
                }
            )
        )
        return None

    return _log_node


def build_activity_log_node(
    log_message: RedboxActivityEvent
    | Callable[[RedboxState], Iterable[RedboxActivityEvent]]
    | Callable[[RedboxState], Iterable[RedboxActivityEvent]],
):
    """
    A Runnable which emits activity events based on the state. The message should either be a static message to log, or a function which returns an activity event or an iterator of them
    """

    @RunnableLambda
    def _activity_log_node(state: RedboxState):
        if isinstance(log_message, RedboxActivityEvent):
            log_activity(log_message)
        else:
            response = log_message(state)
            if isinstance(response, RedboxActivityEvent):
                log_activity(response)
            else:
                for message in response:
                    log_activity(message)
        return None

    return _activity_log_node


def lm_choose_route(state: RedboxState, parser: ClaudeParser):
    """
    LLM choose the route (search/summarise) based on user question and file metadata
    """
    chain = chain_use_metadata(system_prompt=state.request.ai_settings.llm_decide_route_prompt, parser=parser)
    res = chain.invoke(state)
    return res.next.value


def create_planner(is_streamed=False):
    metadata = None

    @RunnableLambda
    def _get_metadata(state: RedboxState):
        nonlocal metadata
        env = get_settings()
        retriever = get_basic_metadata_retriever(env)
        metadata = retriever.invoke(state)
        return state

    @RunnableLambda
    def _stream_planner_agent(state: RedboxState):
        planner_output_parser, format_instructions = get_structured_response_with_planner_parser()
        agent = build_stuff_pattern(
            prompt_set=PromptSet.Planner,
            output_parser=planner_output_parser,
            format_instructions=format_instructions,
            final_response_chain=False,
            additional_variables={"metadata": metadata},
        )
        return agent

    @RunnableLambda
    def _create_planner(state: RedboxState):
        orchestration_agent = create_chain_agent(
            system_prompt=PLANNER_PROMPT + "\n" + PLANNER_QUESTION_PROMPT + "\n" + PLANNER_FORMAT_PROMPT,
            use_metadata=True,
            tools=None,
            parser=ClaudeParser(pydantic_object=MultiAgentPlan),
            using_only_structure=False,
        )
        return orchestration_agent

    if is_streamed:
        return _get_metadata | _stream_planner_agent
    else:
        return _create_planner


def my_planner(allow_plan_feedback=False, node_after_streamed: str = "human", node_afer_replan: str = "sending_task"):
    @RunnableLambda
    def _create_planner(state: RedboxState):
        no_tasks_auto = 1

        if state.user_feedback:
            # replanning
            plan_prompt = REPLAN_PROMPT
            plan = state.request.chat_history[-1].get("text")
            # if we save plans we can use this
            # plan = state.agent_plans[-1].model_dump_json()
            user_input = state.user_feedback.replace("@newroute ", "")
            orchestration_agent = create_chain_agent(
                system_prompt=plan_prompt,
                use_metadata=True,
                tools=None,
                parser=ClaudeParser(pydantic_object=MultiAgentPlan),
                using_only_structure=False,
                _additional_variables={"previous_plan": plan, "user_feedback": user_input},
            )
            res = orchestration_agent.invoke(state)
            state.agent_plans = res
            # reset user feedback
            state.user_feedback = ""
            return Command(update=state, goto=node_afer_replan)
        else:
            # run planner agent
            orchestration_agent = create_planner(is_streamed=False)
            res = orchestration_agent.invoke(state)
            state.agent_plans = res

            # send task to worker agent if we have only 1 task
            if len(state.agent_plans.tasks) > no_tasks_auto:
                if allow_plan_feedback:
                    # stream task
                    return Command(update=state, goto=node_after_streamed)
                else:
                    return Command(update=state, goto=node_afer_replan)
            else:
                return Command(update=state, goto=node_afer_replan)

    return _create_planner


def build_agent(agent_name: str, system_prompt: str, tools: list, use_metadata: bool = False, max_tokens: int = 5000):
    @RunnableLambda
    def _build_agent(state: RedboxState):
        parser = ClaudeParser(pydantic_object=AgentTask)
        try:
            task = parser.parse(state.last_message.content)
        except Exception as e:
            print(f"Cannot parse in {agent_name}: {e}")
        activity_node = build_activity_log_node(
            RedboxActivityEvent(message=f"{agent_name} is completing task: {task.task}")
        )
        activity_node.invoke(state)

        worker_agent = create_chain_agent(
            system_prompt=system_prompt,
            use_metadata=use_metadata,
            parser=None,
            tools=tools,
            _additional_variables={"task": task.task, "expected_output": task.expected_output},
        )
        ai_msg = worker_agent.invoke(state)
        result = run_tools_parallel(ai_msg, tools, state)
        result_content = "".join([res.content for res in result])
        # truncate results to max_token
        result = f"<{agent_name}_Result>{result_content[:max_tokens]}</{agent_name}_Result>"
        return {"agents_results": result}

    return _build_agent


def create_evaluator():
    def _create_evaluator(state: RedboxState):
        _additional_variables = {"agents_results": combine_agents_state(state.agents_results)}
        citation_parser, format_instructions = get_structured_response_with_citations_parser()
        evaluator_agent = build_stuff_pattern(
            prompt_set=PromptSet.NewRoute,
            tools=None,
            output_parser=citation_parser,
            format_instructions=format_instructions,
            final_response_chain=False,
            additional_variables=_additional_variables,
        )
        return evaluator_agent

    return _create_evaluator


def invoke_custom_state(
    custom_graph,
    agent_name: str,
    all_chunks_retriever: VectorStoreRetriever,
    use_as_agent: bool,
    debug: bool = False,
    max_tokens: int = 5000,
):
    @RunnableLambda
    def _invoke_custom_state(state: RedboxState):
        # transform the state to the subgraph state
        subgraph = custom_graph(all_chunks_retriever=all_chunks_retriever, use_as_agent=use_as_agent, debug=debug)
        subgraph_state = state.model_copy()
        agent_task = json.loads(subgraph_state.last_message.content)
        subgraph_state.request.question = (
            agent_task["task"] + f"\nReturn response that is no longer than {max_tokens} tokens."
        )

        # set activity log
        activity_node = build_activity_log_node(
            RedboxActivityEvent(message=f"{agent_name} is completing task: {agent_task["task"]}")
        )
        activity_node.invoke(state)

        ## invoke the subgraph
        response = subgraph.invoke(subgraph_state)  # the LLM response is streamed

        return response

    return _invoke_custom_state


def delete_plan_message():
    @RunnableLambda
    def _delete_plan_message(state: RedboxState):
        last_message = state.messages[-1]
        return {"messages": [RemoveMessage(id=last_message.id)]}

    return _delete_plan_message


def build_user_feedback_evaluation():
    @RunnableLambda
    def _build_user_feedback_evaluation(state: RedboxState):
        decision_agent = create_chain_agent(
            system_prompt=USER_FEEDBACK_EVAL_PROMPT,
            parser=ClaudeParser(pydantic_object=FeedbackEvalDecision),
            _additional_variables={"feedback": state.user_feedback},
            use_metadata=False,
        )
        res = decision_agent.invoke(state)
        return res.next.value

    return _build_user_feedback_evaluation


def stream_plan():
    @RunnableLambda
    def _stream_plan(state: RedboxState):
        for t in state.agent_plans.tasks:
            dispatch_custom_event(RedboxEventType.response_tokens, data=f"{t.task}\n\n")

    return _stream_plan

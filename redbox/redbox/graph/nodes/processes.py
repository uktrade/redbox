import json
import logging
import os
import random
import re
import sqlite3
import textwrap
import time
from collections.abc import Callable
from functools import reduce
from io import StringIO
from random import uniform
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd
from botocore.exceptions import EventStreamError
from langchain.schema import StrOutputParser
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.types import Command

from redbox.chains.activity import log_activity
from redbox.chains.components import (
    get_base_chat_llm,
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
    AgentEnum,
    AgentTask,
    DocumentState,
    FeedbackEvalDecision,
    MultiAgentPlan,
    PromptSet,
    RedboxState,
    RequestMetadata,
    get_plan_fix_prompts,
    get_plan_fix_suggestion_prompts,
)
from redbox.models.graph import ROUTE_NAME_TAG, RedboxActivityEvent, RedboxEventType
from redbox.models.prompts import (
    PLANNER_FORMAT_PROMPT,
    PLANNER_PROMPT,
    PLANNER_QUESTION_PROMPT,
    REPLAN_PROMPT,
    TABULAR_FORMAT_PROMPT,
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


def remove_evaluator_task(state: RedboxState):
    if len(state.agent_plans.tasks) > 0:
        if state.agent_plans.tasks[-1].agent == AgentEnum.Evaluator_Agent:
            state.tasks_evaluator = state.agent_plans.tasks[-1].task + " " + state.agent_plans.tasks[-1].expected_output
            state.agent_plans.tasks.pop(-1)
    return state


def my_planner(
    allow_plan_feedback=False,
    node_after_streamed: str = "human",
    node_afer_replan: str = "sending_task",
    node_for_no_task="evaluator",
):
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
            # remove evaluator agent task
            remove_evaluator_task(state)
            # reset user feedback
            state.user_feedback = ""
            return Command(update=state, goto=node_afer_replan)
        else:
            # run planner agent
            orchestration_agent = create_planner(is_streamed=False)
            res = orchestration_agent.invoke(state)
            state.agent_plans = res

            if res.tasks[-1].agent == AgentEnum.Evaluator_Agent:
                # if there are 0 or 1 task
                if len(res.tasks[:-1]) <= no_tasks_auto:
                    remove_evaluator_task(state)
                    if len(state.agent_plans.tasks) == 0:
                        return Command(update=state, goto=node_for_no_task)
                    else:
                        return Command(update=state, goto=node_afer_replan)
                else:  # there is more than 1 tasks
                    if allow_plan_feedback:
                        # stream task
                        return Command(update=state, goto=node_after_streamed)
                    else:
                        remove_evaluator_task(state)
                        return Command(update=state, goto=node_afer_replan)
            else:
                log.error("The evaluator is not the last agent!")
                return Command(update=state, goto=node_for_no_task)

    return _create_planner


def build_agent(agent_name: str, system_prompt: str, tools: list, use_metadata: bool = False, max_tokens: int = 5000):
    def remove_task_dependecies(state: RedboxState, task_id: str):
        for task in state.agent_plans.tasks:
            if task_id in task.dependencies:
                task.dependencies.remove(task_id)

    @RunnableLambda
    def _build_agent(state: RedboxState):
        parser = ClaudeParser(pydantic_object=AgentTask)
        try:
            task = parser.parse(state.last_message.content)
        except Exception as e:
            print(f"Cannot parse in {agent_name}: {e}")

        # check if dependencies are met before running task
        if not task.dependencies:
            activity_node = build_activity_log_node(
                RedboxActivityEvent(message=f"{agent_name} is completing task: {task.task}")
            )
            activity_node.invoke(state)

            worker_agent = create_chain_agent(
                system_prompt=system_prompt,
                use_metadata=use_metadata,
                parser=None,
                tools=tools,
                _additional_variables={
                    "task": task.task,
                    "expected_output": task.expected_output,
                    "agents_results": state.agents_results,
                },
            )
            ai_msg = worker_agent.invoke(state)
            result = run_tools_parallel(ai_msg, tools, state)
            if type(result) is str:
                result_content = result
            elif type(result) is list:
                result_content = "".join([res.content for res in result])
            else:
                log.error(f"Worker agent return incompatible data type {type(result)}")
            # truncate results to max_token
            result = f"<{agent_name}_Result>{result_content[:max_tokens]}</{agent_name}_Result>"
            # remove task and dependecies
            state.agents_results = result
            remove_task_dependecies(state=state, task_id=task.id)
            if task in state.agent_plans.tasks:
                state.agent_plans.tasks.remove(task)
            return {"agents_results": result, "agent_plans": state.agent_plans}

    return _build_agent.with_retry(stop_after_attempt=3)


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

        # invoking this subgraph will change original state.question - we correct the state question in subsequent nodes

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
            _additional_variables={"plan": state.agent_plans, "feedback": state.user_feedback},
            use_metadata=False,
        )
        res = decision_agent.invoke(state)
        return res.next.value

    return _build_user_feedback_evaluation


def stream_plan():
    """Stream task descriptions and also save the plan in messages"""

    @RunnableLambda
    def _stream_plan(state: RedboxState):
        prefix_texts, suffix_texts = get_plan_fix_prompts()
        dispatch_custom_event(RedboxEventType.response_tokens, data=f"{random.choice(prefix_texts)}\n\n")
        for i, t in enumerate(state.agent_plans.tasks):
            if t.agent != AgentEnum.Evaluator_Agent:
                dispatch_custom_event(RedboxEventType.response_tokens, data=f"{i+1}. {t.task}\n\n")
        dispatch_custom_event(RedboxEventType.response_tokens, data=f"\n\n{random.choice(suffix_texts)}")
        return state

    return _stream_plan


def stream_suggestion():
    @RunnableLambda
    def _stream_suggestion(state: RedboxState):
        texts = get_plan_fix_suggestion_prompts()
        dispatch_custom_event(RedboxEventType.response_tokens, data=f"{random.choice(texts)}")
        return state

    return _stream_suggestion


def extract_response(answer: str) -> tuple[list[str], str]:
    """Extracts the response from a the tabular agent using the 'html'-like Tabular_Agent_Result tags."""
    match = re.search(r"<Tabular_Agent_Result>(.*?)</Tabular_Agent_Result>", answer, re.DOTALL)
    out = match.group(1) if match else "I was unable to parse the result. Please try again."

    def stream_chars(text):
        return split_into_random_chunks(text)

    return stream_chars(out), out


def split_into_random_chunks(text, min_len=20, max_len=50):
    """Splits response into chunks of random character size."""
    chunks = []
    i = 0
    while i < len(text):
        chunk_len = random.randint(min_len, max_len)
        chunks.append(text[i : i + chunk_len])
        i += chunk_len
    return chunks


def stream_tabular_failure():
    """Stream the failure of the tabular agent and fallback to newroute"""

    @RunnableLambda
    def _stream_tabular_failure(state: RedboxState):
        stream = split_into_random_chunks(
            "Unfortunately, I am unable to output a response using the Tabular Agent. I will try a different approach.\n\n",
            5,
            20,
        )
        for char in stream:
            dispatch_custom_event(RedboxEventType.response_tokens, data=char)
        state.request.question = state.request.question.lstrip("@tabular")
        return state

    return _stream_tabular_failure


def stream_tabular_response():
    """Stream the tabular analysis response and save the analysis in messages"""

    @RunnableLambda
    def _stream_tabular_response(state: RedboxState):
        stream, answer = extract_response(state.agents_results[-1].content)
        for char in stream:
            dispatch_custom_event(RedboxEventType.response_tokens, data=char)
        state.messages.append(AIMessage(content=answer))
        return state

    return _stream_tabular_response


def combine_question_evaluator() -> Runnable[RedboxState, dict[str, Any]]:
    """Returns a Runnable that uses state["request"] to set state["text"]."""

    @RunnableLambda
    def _combine_question(state: RedboxState) -> dict[str, Any]:
        state.request.question = "\n\n".join([task.content for task in state.tasks_evaluator])
        return state

    return _combine_question


def create_or_update_db_from_tabulars(state: RedboxState) -> RedboxState:
    """
    Initialise a database or use existing one if valid.

    If state.request.db_location is None, creates a new database at a generated path.
    If state.request.db_location exists, checks if documents have changed since last use.
    Only regenerates the database if the file doesn't exist or documents have changed.
    """
    should_create_db = True
    db_path = state.request.db_location

    # Check if we have an existing db_path
    if db_path:
        # Check if the file exists
        if os.path.exists(db_path) and not state.documents_changed():
            # If the file exists and documents haven't changed, no need to recreate
            should_create_db = False
    else:
        # Generate a new db path if self.db_location is none
        user_uuid = str(state.request.user_uuid) if state.request.user_uuid else uuid4()
        db_path = f"generated_db_{user_uuid}.db"  # Initialise the database at a location that uses the user's UUID or a random one if it's not available

    # Create or update database if needed
    if should_create_db:
        # Creating/updating database at db_path
        conn = sqlite3.connect(db_path)
        doc_texts, doc_names = get_all_tabular_docs(state)
        _ = load_texts_to_db(doc_texts, doc_names=doc_names, conn=conn)
        conn.commit
        conn.close

        # Store the current_documents to reflect current state
        state.store_document_state()
        state.request.db_location = db_path
    return state


def get_all_tabular_docs(state: RedboxState) -> tuple[list[str], list[str]]:
    """Gets the file names and text for all tabular files in the redbox state"""
    all_texts, all_names = [], []
    for doc_state in state.documents:
        for group in doc_state[1].values():
            if group:
                for doc in group.values():
                    if isinstance(doc, Document):
                        try:
                            # Tabular Retriever will only get csvs or excel files.
                            all_texts.append(doc.page_content)
                            all_names.append(
                                os.path.splitext(os.path.basename(doc.metadata["uri"]))[0]
                            )  # Get file name
                        except Exception as e:
                            log.info(f"{doc} could not be parsed! \n\n{e}")
                            continue

    return all_texts, all_names


def detect_tabular_docs(state: RedboxState) -> bool:
    """Returns True if a tabular document is selected"""
    for doc_state in state.documents:
        for group in doc_state[1].values():
            if group:
                for doc in group.values():
                    uri = doc.metadata.get("uri", "")
                    if uri.endswith((".csv", ".xlsx", ".xls")):
                        return True
    return False


def extract_table_names_and_text(doc_text: str) -> tuple[str, str]:
    """Excel Files start with Sheet Names stored alongside text so we extract this if so."""
    if doc_text.startswith("<table_name>"):
        pattern = r"<table_name>(.*?)</table_name>(.*)"
        match = re.match(pattern, doc_text, re.DOTALL)
        if match:
            table_name = match.group(1)
            extracted_text = match.group(2).strip()
            return table_name, extracted_text
    return None, doc_text


def load_texts_to_db(doc_texts: list[str], doc_names: list[str], conn: sqlite3.Connection) -> list[str]:
    """Load document texts as tables in a database"""
    table_names = []
    for idx, (doc_text, doc_name) in enumerate(zip(doc_texts, doc_names)):
        try:
            clean_doc_name = sanitise_file_name(doc_name)
            sheet_name, doc_text = extract_table_names_and_text(doc_text)
            table_name = f"{clean_doc_name}_table_{idx+1}" if not sheet_name else f"{clean_doc_name}_{sheet_name}"
            parse_doc_text_as_db_table(doc_text, table_name, conn=conn)
            table_names.append(table_name)
        except Exception as e:
            log.exception(f"Failed to load table for Document '{table_name}': {e}")
    return table_names


def parse_doc_text_as_db_table(doc_text: str, table_name: str, conn: sqlite3.Connection):
    """Convert document text to SQL"""
    try:
        df = pd.read_csv(StringIO(doc_text), sep=",")

        df.to_sql(table_name, conn, if_exists="replace", index=False)
    except Exception as e:
        log.exception(f"Failed to load table '{table_name}': {e}")


def sanitise_file_name(file_name: str) -> str:
    """Removes Spaces and special characters from file names"""
    return re.sub(r"\W+", "", file_name.replace(" ", ""))


def build_tabular_agent(agent_name: str = "Tabular Agent", max_tokens: int = 5000):
    @RunnableLambda
    def _build_tabular_agent(state: RedboxState):
        state = create_or_update_db_from_tabulars(state=state)
        parser = ClaudeParser(pydantic_object=AgentTask)

        try:
            task = parser.parse(state.last_message.content)
            # Log activity
            activity_node = build_activity_log_node(
                RedboxActivityEvent(message=f"{agent_name} is completing task: {task.task}")
            )
            activity_node.invoke(state)
        except Exception as e:
            log.error(f"Cannot parse in {agent_name}: {e}")
            task = AgentTask(task=state.request.question, expected_output="Analysis of tabular data")

        # Get the structured response parser
        # steps_taken_parser, _ = get_structured_response_with_steps_taken_parser()

        try:
            # Create SQL database agent with structured output format
            db = SQLDatabase.from_uri(f"sqlite:///{state.request.db_location}")
            llm = get_base_chat_llm(model=state.request.ai_settings.chat_backend)

            # Customise the agent to return structured responses
            agent = create_sql_agent(
                llm=llm,
                db=db,
                verbose=False,
                agent_executor_kwargs={
                    "handle_parsing_errors": True,
                },
            )

            agent_result = agent.invoke(
                {"input": TABULAR_FORMAT_PROMPT.format(question=task.task.replace("@tabular ", ""))}
            )

            output = agent_result["output"]
            result = output

            # Truncate to max_tokens
            result_content = result[:max_tokens]

            formatted_result = f"<Tabular_Agent_Result>{result_content}</Tabular_Agent_Result>"
            return {"agents_results": formatted_result, "tasks_evaluator": task.task + "\n" + task.expected_output}

        except Exception as e:
            log.error(f"Error generating agent output: {e}")

            formatted_result = "<Tabular_Agent_Result>Error analysing tabular data</Tabular_Agent_Result>"
            return {"agents_results": formatted_result, "tasks_evaluator": task.task + "\n" + task.expected_output}

    return _build_tabular_agent

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
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.types import Command

from redbox.chains.activity import log_activity
from redbox.chains.components import get_chat_llm, get_structured_response_with_citations_parser, get_tokeniser
from redbox.chains.parser import ClaudeParser
from redbox.chains.runnables import CannedChatLLM, build_llm_chain, chain_use_metadata, create_chain_agent
from redbox.graph.nodes.sends import run_tools_parallel
from redbox.models import ChatRoute
from redbox.models.chain import (
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
from redbox.models.prompts import USER_FEEDBACK_EVAL_PROMPT
from redbox.transform import bedrock_tokeniser, combine_agents_state, combine_documents, flatten_document_state

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
    # metadata = None
    document_filenames = []

    # @RunnableLambda
    # def _get_metadata(state: RedboxState):
    #     nonlocal metadata
    #     env = get_settings()
    #     retriever = get_basic_metadata_retriever(env)
    #     metadata = retriever.invoke(state)
    #     return state

    @RunnableLambda
    def _document_filenames(state: RedboxState):
        nonlocal document_filenames
        document_filenames = [doc.split("/")[1] if "/" in doc else doc for doc in state.request.s3_keys]
        return state

    # @RunnableLambda
    # def _stream_planner_agent(state: RedboxState):
    #     planner_output_parser, format_instructions = get_structured_response_with_planner_parser()
    #     agent = build_stuff_pattern(
    #         prompt_set=PromptSet.Planner,
    #         output_parser=planner_output_parser,
    #         format_instructions=format_instructions,
    #         final_response_chain=False,
    #         additional_variables={"metadata": metadata, "document_filenames": document_filenames},
    #     )
    #     return agent

    @RunnableLambda
    def _create_planner(state: RedboxState):
        planner_prompt = state.request.ai_settings.planner_prompt_with_format
        orchestration_agent = create_chain_agent(
            system_prompt=planner_prompt,
            use_metadata=True,
            tools=None,
            parser=ClaudeParser(pydantic_object=MultiAgentPlan),
            using_only_structure=False,
            using_chat_history=True,
            _additional_variables={"document_filenames": document_filenames},
        )
        return orchestration_agent

    # if is_streamed:
    #     return _get_metadata | _document_filenames | _stream_planner_agent
    # else:
    return _document_filenames | _create_planner


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
            plan_prompt = state.request.ai_settings.replanner_prompt
            plan = state.request.chat_history[-1].get("text")
            # if we save plans we can use this
            # plan = state.agent_plans[-1].model_dump_json()
            user_input = state.user_feedback.replace("@newroute ", "")
            document_filenames = [doc.split("/")[1] if "/" in doc else doc for doc in state.request.s3_keys]
            orchestration_agent = create_chain_agent(
                system_prompt=plan_prompt,
                use_metadata=True,
                tools=None,
                parser=ClaudeParser(pydantic_object=MultiAgentPlan),
                using_only_structure=False,
                _additional_variables={
                    "previous_plan": plan,
                    "user_feedback": user_input,
                    "document_filenames": document_filenames,
                },
                using_chat_history=True,
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
            # if there is no task, go to evaluator
            elif len(state.agent_plans.tasks) == 0:
                return Command(update=state, goto=node_for_no_task)
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
        if type(result) is str:
            result_content = result
        elif type(result) is list:
            result_content = []
            current_token_counts = 0
            for res in result:
                current_token_counts += bedrock_tokeniser(res.content)
                if current_token_counts <= max_tokens:
                    result_content.append(res.content)
            result_content = " ".join(result_content)
            result = f"<{agent_name}_Result>{result_content}</{agent_name}_Result>"
        else:
            log.error(f"Worker agent return incompatible data type {type(result)}")
        return {"agents_results": result, "tasks_evaluator": task.task + "\n" + task.expected_output}

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
            RedboxActivityEvent(message=f"{agent_name} is completing task: {agent_task['task']}")
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
            dispatch_custom_event(RedboxEventType.response_tokens, data=f"{i + 1}. {t.task}\n\n")
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
    user_uuid = str(state.request.user_uuid) if state.request.user_uuid else uuid4()
    db_path = f"generated_db_{user_uuid}.db"  # Initialise the database at a location that uses the user's UUID or a random one if it's not available
    state.request.db_location = db_path

    # Check if we have an existing db_path
    # Check if the file exists
    if os.path.exists(db_path) and not state.documents_changed():
        # If the file exists and documents haven't changed, no need to recreate
        should_create_db = False

    # Create or update database if needed
    if should_create_db:
        # delete existing database
        if os.path.exists(db_path):
            os.remove(db_path)
        # Creating/updating database at db_path
        conn = sqlite3.connect(db_path)
        doc_texts, doc_names = get_all_tabular_docs(state)
        _ = load_texts_to_db(doc_texts, doc_names=doc_names, conn=conn)
        conn.commit()
        conn.close()

    state.request.previous_s3_keys = sorted(state.request.s3_keys)

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
            table_name = f"{clean_doc_name}_table_{idx + 1}" if not sheet_name else f"{clean_doc_name}_{sheet_name}"
            parse_doc_text_as_db_table(doc_text, table_name, conn=conn)
            table_names.append(table_name)
        except Exception as e:
            log.exception(f"Failed to load table for Document '{table_name}': {e}")
    return table_names


def detect_header(doc_text: str):
    # split the content into lines
    lines = doc_text.splitlines()
    # find the first line that look like an actual header
    header_row = None
    for i, line in enumerate(lines):
        values = [value.strip() for value in line.split(",")]
        # if there are more than 1 columns and first few columns are not empty
        trimmed_values = values[:10]
        if (
            len(values) > 1
            and all(not value.lower().startswith("unnamed") for value in trimmed_values)
            and "".join(trimmed_values)
            and sum([1 if not val else 0 for val in trimmed_values]) < 2
        ):
            header_row = i
            break

    if header_row is None:  # in case there is only one column
        header_row = 0
    text_from_header = "\n".join(lines[header_row:])
    return text_from_header


def delete_null_values(df: pd.DataFrame):
    # delete rows where all values are null
    df.dropna(axis=0, how="all", inplace=True)
    # delete columns where all values are null
    df.dropna(axis=1, how="all", inplace=True)
    return df


def parse_doc_text_as_db_table(doc_text: str, table_name: str, conn: sqlite3.Connection):
    """Convert document text to SQL"""
    try:
        text_from_header = detect_header(doc_text)
        df = pd.read_csv(StringIO(text_from_header), sep=",")
        df_cleaned = delete_null_values(df)

        if df_cleaned.empty or len(df_cleaned.columns) == 0:
            log.warning(f"skipping table '{table_name}' as there are no valid columns after cleaning {len(doc_text)}")
            return

        df_cleaned.to_sql(table_name, conn, if_exists="replace", index=False)
    except Exception as e:
        log.exception(f"Failed to load table '{table_name}': {e}")


def sanitise_file_name(file_name: str) -> str:
    """Removes Spaces and special characters from file names"""
    return re.sub(r"\W+", "", file_name.replace(" ", ""))


def get_tabular_agent(
    agent_name: str = "Tabular Agent", max_tokens: int = 5000, tools=list[StructuredTool], max_attempt=int
):
    @RunnableLambda
    def _build_tabular_agent(state: RedboxState):
        # update activity
        try:
            # retrieve tabular agent task
            tasks = state.agent_plans.tasks
            for task_level in tasks:
                if task_level.agent.value == "Tabular_Agent":
                    task = task_level.task
                    expected_output = task_level.expected_output
        except Exception as e:
            log.error(f"Cannot parse in {agent_name}: {e}")
            task = state.request.question

        # Log activity
        activity_node = build_activity_log_node(RedboxActivityEvent(message=f"{agent_name} is completing task: {task}"))
        activity_node.invoke(state)

        # Create SQL database agent with structured output format
        # db = SQLDatabase.from_uri(f"sqlite:///{state.request.db_location}")
        # call tabular agent
        success = "fail"
        num_iter = 0
        sql_error = ""
        is_intermediate_step = False
        messages = []
        while (success == "fail" or is_intermediate_step) and num_iter < max_attempt:
            worker_agent = build_stuff_pattern(
                prompt_set=PromptSet.Tabular,
                tools=tools,
                final_response_chain=False,
                additional_variables={"sql_error": sql_error, "db_schema": state.tabular_schema},
            )
            ai_msg = worker_agent.invoke(state)

            messages.append(AIMessage(ai_msg["messages"][-1].content))
            try:
                messages.append(
                    AIMessage(f"Here is the SQL query: {ai_msg['messages'][-1].tool_calls[-1]['args']['sql_query']}")
                )
            except Exception:
                log.info("no sql query input to tool")

            num_iter += 1
            if isinstance(ai_msg["messages"][-1].content, str):
                tabular_context = ai_msg["messages"][-1].content
            else:
                tabular_context = ""
            tool_output = run_tools_parallel(ai_msg["messages"][-1], tools, state)

            results = tool_output[-1].content  # this is a tuple

            # Truncate to max_tokens. only using one tool here.
            # retrieve result from database or sql error
            result = results[0][:max_tokens]  # saving this as a new variable as tuples are immutable.
            success = results[1]

            is_intermediate_step = eval(results[2])

            if success == "fail":
                sql_error = result  # capture sql error
                messages.append(
                    AIMessage(f"The SQL query failed to execute correctly. Here is the error message: {sql_error}")
                )
            else:
                if is_intermediate_step:
                    sql_error = ""
                    messages.append(AIMessage(f"Here are the results from the query: {result}"))
            # update state messages
            state.messages = messages

        if success == "pass":
            if not is_intermediate_step:  # if this is the final step
                formatted_result = f"<Tabular_Agent_Result>{tabular_context}\n The results of my query are: {result}</Tabular_Agent_Result>"
            else:
                formatted_result = f"<Tabular_Agent_Result>Iteration limit of {num_iter} is reached by the tabular agent. This is the tabular agent's reasoning at the last iteration: {tabular_context}</Tabular_Agent_Result>"

        else:
            formatted_result = f"<Tabular_Agent_Result>Error analysing tabular data. Here is the error from the executed SQL query: {result} </Tabular_Agent_Result>"

        return {
            "agents_results": [formatted_result],
            "tasks_evaluator": task + "\n" + expected_output,
        }

    return _build_tabular_agent


def get_tabular_schema():
    def _get_tabular_schema(state: RedboxState):
        # create db
        state = create_or_update_db_from_tabulars(state=state)
        db_path = state.request.db_location
        # get schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # get tables
        tables = cursor.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
        schema = {"tables": []}
        for (table_name,) in tables:
            cols = cursor.execute(f'PRAGMA table_info("{table_name}");').fetchall()
            # convert to JSON
            schema["tables"].append(
                {
                    "name": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "notnull": bool(col[3]),
                            "default": col[4],
                            "primary_key": bool(col[5]),
                        }
                        for col in cols
                    ],
                }
            )

        conn.close()
        db_schema = json.dumps(schema, indent=2)

        return {"tabular_schema": db_schema}

    return _get_tabular_schema

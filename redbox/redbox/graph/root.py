import logging
from typing import Dict

from langchain_core.messages import AIMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.pregel import RetryPolicy

from redbox.chains.components import get_structured_response_with_citations_parser
from redbox.chains.runnables import build_self_route_output_parser
from redbox.graph.agents.configs import AgentConfig
from redbox.graph.agents.workers import WorkerAgent
from redbox.graph.edges import (
    build_documents_bigger_than_context_conditional,
    build_keyword_detection_conditional,
    build_total_tokens_request_handler_conditional,
    is_using_search_keyword,
    multiple_docs_in_group_conditional,
)
from redbox.graph.nodes.processes import (
    build_activity_log_node,
    build_agent_with_loop,
    build_chat_pattern,
    build_error_pattern,
    build_merge_pattern,
    build_passthrough_pattern,
    build_retrieve_pattern,
    build_set_metadata_pattern,
    build_set_route_pattern,
    build_stuff_pattern,
    build_user_feedback_evaluation,
    clear_documents_process,
    combine_question_evaluator,
    create_evaluator,
    empty_process,
    get_tabular_agent,
    get_tabular_schema,
    invoke_custom_state,
    my_planner,
    report_sources_process,
    stream_plan,
    stream_suggestion,
)
from redbox.graph.nodes.sends import build_document_chunk_send, build_document_group_send, sending_task_to_agent
from redbox.models.chain import PromptSet, RedboxState
from redbox.models.chat import ChatRoute, ErrorRoute
from redbox.models.graph import ROUTABLE_KEYWORDS, RedboxActivityEvent
from redbox.models.prompts import EVAL_SUBMISSION, EVAL_SUBMISSION_QA
from redbox.models.settings import ChatLLMBackend, get_settings
from redbox.transform import structure_documents_by_file_name, structure_documents_by_group_and_indices

log = logging.getLogger(__name__)


def build_root_graph(
    all_chunks_retriever,
    parameterised_retriever,
    metadata_retriever,
    tabular_retriever,
    agent_configs,
    debug,
):
    builder = StateGraph(RedboxState)

    # nodes
    builder.add_node("chat_graph", get_chat_graph(debug=debug))
    builder.add_node("search_graph", get_search_graph(retriever=parameterised_retriever, debug=debug))
    builder.add_node(
        "summarise_graph",
        get_summarise_graph(all_chunks_retriever=all_chunks_retriever, debug=debug),
    )
    builder.add_node(
        "new_route_graph",
        build_new_route_graph(
            all_chunks_retriever=all_chunks_retriever,
            tabular_retriever=tabular_retriever,
            agent_configs=agent_configs,
            debug=debug,
        ),
    )
    builder.add_node(
        "retrieve_metadata", get_retrieve_metadata_graph(metadata_retriever=metadata_retriever, debug=debug)
    )

    builder.add_node("is_summarise_route", empty_process)
    builder.add_node("has_keyword", empty_process)
    builder.add_node("no_user_feedback", empty_process)

    builder.add_node(
        "log_user_request",
        build_activity_log_node(
            lambda s: [
                RedboxActivityEvent(
                    message=f"You selected {len(s.request.s3_keys)} file{'s' if len(s.request.s3_keys) > 1 else ''} - {','.join(s.request.s3_keys)}"
                )
                if len(s.request.s3_keys) > 0
                else "You selected no files",
            ]
        ),
    )

    # edges
    builder.add_edge(START, "log_user_request")
    builder.add_edge(START, "retrieve_metadata")
    builder.add_edge("retrieve_metadata", "no_user_feedback")
    builder.add_conditional_edges(
        "no_user_feedback", lambda s: s.user_feedback == "", {True: "has_keyword", False: "new_route_graph"}
    )
    builder.add_conditional_edges(
        "has_keyword",
        build_keyword_detection_conditional(*ROUTABLE_KEYWORDS.keys()),
        {
            ChatRoute.search: "search_graph",
            ChatRoute.newroute: "new_route_graph",
            ChatRoute.summarise: "summarise_graph",
            ChatRoute.chat: "chat_graph",
            "DEFAULT": "new_route_graph",
        },
    )

    builder.add_edge("search_graph", "is_summarise_route")
    builder.add_conditional_edges(
        "is_summarise_route", lambda s: s.route_name == ChatRoute.summarise, {True: "summarise_graph", False: END}
    )
    builder.add_edge("search_graph", END)
    builder.add_edge("new_route_graph", END)
    builder.add_edge("summarise_graph", END)
    return builder.compile()


def get_search_graph(
    retriever: VectorStoreRetriever,
    prompt_set: PromptSet = PromptSet.Search,
    debug: bool = False,
    final_sources: bool = True,
) -> CompiledGraph:
    """Creates a subgraph for retrieval augmented generation (RAG)."""
    citations_output_parser, format_instructions = get_structured_response_with_citations_parser()

    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node(
        "set_route_to_search",
        build_set_route_pattern(route=ChatRoute.search),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "llm_generate_query",
        build_chat_pattern(prompt_set=PromptSet.CondenseQuestion),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "retrieve_documents",
        build_retrieve_pattern(
            retriever=retriever,
            structure_func=structure_documents_by_group_and_indices,
            final_source_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "llm_answer_question",
        build_stuff_pattern(
            prompt_set=prompt_set,
            output_parser=citations_output_parser,
            format_instructions=format_instructions,
            final_response_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )

    builder.add_node(
        "set_route_to_summarise",
        build_set_route_pattern(route=ChatRoute.summarise),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "report_citations",
        report_sources_process,
        retry=RetryPolicy(max_attempts=3),
    )

    def rag_cannot_answer(llm_response: str):
        if isinstance(llm_response, AIMessage):
            llm_response = llm_response.content
        return "unanswerable" in llm_response.lower()

    builder.add_node(
        "check_if_RAG_can_answer",
        build_stuff_pattern(
            prompt_set=PromptSet.SelfRoute,
            format_instructions=format_instructions,
            output_parser=build_self_route_output_parser(
                match_condition=rag_cannot_answer,
                max_tokens_to_check=4,
                parser=citations_output_parser,
            ),
            final_response_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "clear_documents",
        clear_documents_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "RAG_cannot_answer",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node("is_using_search_keyword", empty_process)

    builder.add_node("empty_docs_returned", empty_process)
    # Edges
    builder.add_edge(START, "llm_generate_query")
    builder.add_edge("llm_generate_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "is_using_search_keyword")
    builder.add_conditional_edges(
        "is_using_search_keyword", is_using_search_keyword, {True: "llm_answer_question", False: "empty_docs_returned"}
    )
    builder.add_conditional_edges(
        "empty_docs_returned",
        lambda s: len(s.documents.groups) == 0,
        {True: "set_route_to_summarise", False: "check_if_RAG_can_answer"},
    )

    builder.add_edge("llm_answer_question", "report_citations")
    builder.add_edge("report_citations", "set_route_to_search")

    builder.add_edge("check_if_RAG_can_answer", "RAG_cannot_answer")

    builder.add_conditional_edges(
        "RAG_cannot_answer",
        lambda s: rag_cannot_answer(s.last_message),
        {True: "clear_documents", False: "report_citations"},
    )
    builder.add_edge("clear_documents", "set_route_to_summarise")
    builder.add_edge("set_route_to_summarise", END)
    builder.add_edge("set_route_to_search", END)

    return builder.compile(debug=debug)


def get_summarise_graph(
    all_chunks_retriever: VectorStoreRetriever, use_as_agent=False, model: ChatLLMBackend | None = None, debug=True
):
    builder = StateGraph(RedboxState)
    builder.add_node("choose_route_based_on_request_token", empty_process)
    builder.add_node("set_route_to_summarise_large_doc", build_set_route_pattern(ChatRoute.chat_with_docs_map_reduce))
    builder.add_node("set_route_to_summarise_doc", build_set_route_pattern(ChatRoute.chat_with_docs))
    builder.add_node("pass_user_prompt_to_LLM_message", build_passthrough_pattern())
    builder.add_node("clear_documents", clear_documents_process)

    builder.add_node("document_has_multiple_chunks", empty_process)
    builder.add_node("any_summarised_docs_bigger_than_context_window", empty_process)
    builder.add_node("any_document_bigger_than_context_window", empty_process)

    builder.add_node("sending_chunks_to_summarise", empty_process)
    builder.add_node("sending_summarised_chunks_checking_exceed_context", empty_process)
    builder.add_node("sending_summarised_chunks", empty_process)

    builder.add_node(
        "summarise_summarised_chunks",
        build_merge_pattern(prompt_set=PromptSet.ChatwithDocsMapReduce),
        retry=RetryPolicy(max_attempts=3),
    )

    builder.add_node(
        "files_too_large_error",
        build_error_pattern(
            text="These documents are too large to work with.",
            route_name=ErrorRoute.files_too_large,
        ),
    )

    builder.add_node(
        "retrieve_all_chunks",
        build_retrieve_pattern(
            retriever=all_chunks_retriever,
            structure_func=structure_documents_by_file_name,
            final_source_chain=True,
        ),
    )
    builder.add_node(
        "summarise_each_chunk_in_document",
        build_merge_pattern(prompt_set=PromptSet.ChatwithDocsMapReduce),
        retry=RetryPolicy(max_attempts=3),
    )

    # edges
    builder.add_edge(START, "choose_route_based_on_request_token")
    builder.add_conditional_edges(
        "choose_route_based_on_request_token",
        build_total_tokens_request_handler_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            "max_exceeded": "files_too_large_error",
            "context_exceeded": "set_route_to_summarise_large_doc",
            "pass": "set_route_to_summarise_doc",
        },
    )
    builder.add_edge("set_route_to_summarise_large_doc", "pass_user_prompt_to_LLM_message")
    builder.add_edge("set_route_to_summarise_doc", "pass_user_prompt_to_LLM_message")
    builder.add_edge("pass_user_prompt_to_LLM_message", "retrieve_all_chunks")
    builder.add_conditional_edges(
        "retrieve_all_chunks",
        lambda s: s.route_name,
        {
            ChatRoute.chat_with_docs: "summarise_document",
            ChatRoute.chat_with_docs_map_reduce: "sending_chunks_to_summarise",
        },
    )

    # summarise process
    builder.add_node(
        "summarise_document",
        build_stuff_pattern(
            prompt_set=PromptSet.ChatwithDocs,
            final_response_chain=False if use_as_agent else True,
            summary_multiagent_flag=True if use_as_agent else False,
            model=model,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_edge("summarise_document", "clear_documents")

    # summarise large documents process
    builder.add_conditional_edges(
        "sending_chunks_to_summarise",
        build_document_chunk_send("summarise_each_chunk_in_document"),
        path_map=["summarise_each_chunk_in_document"],
    )
    builder.add_edge("summarise_each_chunk_in_document", "document_has_multiple_chunks")
    builder.add_conditional_edges(
        "document_has_multiple_chunks",
        multiple_docs_in_group_conditional,
        {
            True: "sending_summarised_chunks_checking_exceed_context",
            False: "any_summarised_docs_bigger_than_context_window",
        },
    )
    builder.add_conditional_edges(
        "sending_summarised_chunks_checking_exceed_context",
        build_document_group_send("any_document_bigger_than_context_window"),
        path_map=["any_document_bigger_than_context_window"],
    )
    builder.add_conditional_edges(
        "any_document_bigger_than_context_window",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            True: "files_too_large_error",
            False: "sending_summarised_chunks",
        },
    )
    builder.add_conditional_edges(
        "sending_summarised_chunks",
        build_document_group_send("summarise_summarised_chunks"),
        path_map=["summarise_summarised_chunks"],
    )
    builder.add_edge("summarise_summarised_chunks", "any_summarised_docs_bigger_than_context_window")
    builder.add_conditional_edges(
        "any_summarised_docs_bigger_than_context_window",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocs),
        {
            True: "files_too_large_error",
            False: "summarise_document",
        },
    )
    builder.add_edge("summarise_document", "clear_documents")
    builder.add_edge("clear_documents", END)
    builder.add_edge("files_too_large_error", END)
    return builder.compile(debug=debug)


def get_chat_graph(
    debug: bool = False,
) -> CompiledGraph:
    """Creates a subgraph for standard chat."""
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node(
        "p_set_chat_route",
        build_set_route_pattern(route=ChatRoute.chat),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_chat",
        build_chat_pattern(prompt_set=PromptSet.Chat, final_response_chain=True),
        retry=RetryPolicy(max_attempts=3),
    )

    # Edges
    builder.add_edge(START, "p_set_chat_route")
    builder.add_edge("p_set_chat_route", "p_chat")
    builder.add_edge("p_chat", END)

    return builder.compile(debug=debug)


def get_retrieve_metadata_graph(metadata_retriever: VectorStoreRetriever, debug: bool = False):
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node(
        "p_retrieve_metadata",
        build_retrieve_pattern(
            retriever=metadata_retriever,
            structure_func=structure_documents_by_file_name,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_set_metadata",
        build_set_metadata_pattern(),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_clear_metadata_documents",
        clear_documents_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Edges
    builder.add_edge(START, "p_retrieve_metadata")
    builder.add_edge("p_retrieve_metadata", "p_set_metadata")
    builder.add_edge("p_set_metadata", "p_clear_metadata_documents")
    builder.add_edge("p_clear_metadata_documents", END)

    return builder.compile(debug=debug)


def strip_route(state: RedboxState):
    state.request.question = state.request.question.replace("@newroute ", "")
    return state


def build_new_route_graph(
    all_chunks_retriever: VectorStoreRetriever,
    tabular_retriever: VectorStoreRetriever,
    agent_configs: Dict[str, AgentConfig],
    debug: bool = False,
) -> CompiledGraph:
    def update_submission_eval(state: RedboxState):
        state.tasks_evaluator = EVAL_SUBMISSION
        return state

    def update_submission_qa(state: RedboxState):
        state.tasks_evaluator = EVAL_SUBMISSION_QA
        return state

    def add_agent(
        builder, agent_configs, agent_name, edge_nodes=["combine_question_evaluator"], with_loop=False, **kwargs
    ):
        try:
            config = agent_configs[agent_name]
            match config.name:
                case "Summarisation_Agent":
                    builder.add_node(
                        config.name,
                        invoke_custom_state(
                            custom_graph=get_summarise_graph,
                            agent_name=config.name,
                            all_chunks_retriever=all_chunks_retriever,
                            max_tokens=config.agents_max_tokens,
                            use_as_agent=True,
                            model=ChatLLMBackend(name=config.llm_backend.name, provider=config.llm_backend.provider)
                            if config.llm_backend is not None
                            else None,
                            debug=debug,
                        ),
                    )
                case "Tabular_Agent":
                    builder.add_node(
                        "Tabular_Agent",
                        empty_process,
                    )

                    builder.add_node(
                        "retrieve_tabular_documents",
                        build_retrieve_pattern(
                            retriever=tabular_retriever,
                            structure_func=structure_documents_by_file_name,
                            final_source_chain=False,
                        ),
                    )

                    builder.add_node("retrieve_tabular_schema", get_tabular_schema())

                    builder.add_node(
                        "call_tabular_agent",
                        get_tabular_agent(
                            tools=config.tools,
                            max_attempt=get_settings().max_attempts,
                            max_tokens=config.agents_max_tokens,
                            model=ChatLLMBackend(name=config.llm_backend.name, provider=config.llm_backend.provider)
                            if config.llm_backend is not None
                            else None,
                        ),
                    )
                    builder.add_edge("Tabular_Agent", "retrieve_tabular_documents")
                    builder.add_edge("retrieve_tabular_documents", "retrieve_tabular_schema")
                    builder.add_edge("retrieve_tabular_schema", "call_tabular_agent")
                    builder.add_edge("call_tabular_agent", "combine_question_evaluator")
                case _:
                    success = "fail"
                    is_intermediate_step = False
                    worker = WorkerAgent(config=config)
                    builder.add_node(
                        config.name,
                        worker.execute()
                        if not with_loop
                        else build_agent_with_loop(
                            agent_name=config.name,
                            system_prompt=config.prompt.system,
                            tools=config.tools,
                            max_tokens=config.agents_max_tokens,
                            loop_condition=lambda: success == "fail" or is_intermediate_step,
                            max_attempt=kwargs.get("max_attempt", 2),
                            use_metadata=kwargs.get("use_metadata", False),
                            using_chat_history=kwargs.get("using_chat_history", False),
                            model=ChatLLMBackend(name=config.llm_backend.name, provider=config.llm_backend.provider)
                            if config.llm_backend is not None
                            else None,
                        ),
                    )

            # add edge
            _edge_nodes = edge_nodes.copy()
            _edge_nodes.insert(0, config.name)
            for i in range(len(_edge_nodes) - 1):
                builder.add_edge(_edge_nodes[i], _edge_nodes[i + 1])
        except KeyError:
            # can't find agent
            log.error(f"Here is the list of agents: {agent_configs}")
            raise ValueError(f"No agent found with name {agent_name}")

    allow_plan_feedback = get_settings().allow_plan_feedback

    builder = StateGraph(RedboxState)

    # add nodes
    builder.add_node(
        "set_route_to_newroute",
        build_set_route_pattern(route=ChatRoute.newroute),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node("remove_keyword", strip_route)
    builder.add_node("stream_plan", stream_plan())
    builder.add_node(
        "planner",
        my_planner(
            allow_plan_feedback=allow_plan_feedback,
            node_after_streamed="stream_plan",
            node_afer_replan="sending_task",
            node_for_no_task="Evaluator_Agent",
        ),
    )
    builder.add_node("update_submission_eval", update_submission_eval)
    builder.add_node("update_submission_qa", update_submission_qa)
    builder.add_node("send", empty_process)
    builder.add_node("user_feedback_evaluation", empty_process)
    builder.add_node("Evaluator_Agent", create_evaluator())
    builder.add_node("combine_question_evaluator", combine_question_evaluator())
    builder.add_node(
        "report_citations",
        report_sources_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node("stream_suggestion", stream_suggestion())
    builder.add_node("sending_task", empty_process)

    # add all agents here
    add_agent(builder, agent_configs, "Internal_Retrieval_Agent")
    add_agent(builder, agent_configs, "External_Retrieval_Agent")
    add_agent(builder, agent_configs, "Summarisation_Agent", edge_nodes=[])  # streaming response directly
    add_agent(builder, agent_configs, "Tabular_Agent", edge_nodes=[])  # go to other nodes/subgraphs
    add_agent(builder, agent_configs, "Web_Search_Agent")
    add_agent(builder, agent_configs, "Legislation_Search_Agent")
    add_agent(
        builder,
        agent_configs,
        "Datahub_Agent",
        with_loop=True,
        using_chat_history=True,
        edge_nodes=["combine_question_evaluator"],
    )
    add_agent(builder, agent_configs, "Knowledge_Base_Retrieval_Agent")
    add_agent(
        builder,
        agent_configs,
        "Submission_Checker_Agent",
        with_loop=True,
        use_metadata=True,
        using_chat_history=True,
        edge_nodes=["update_submission_eval", "combine_question_evaluator"],
    )
    add_agent(
        builder,
        agent_configs,
        "Submission_Question_Answer_Agent",
        with_loop=True,
        use_metadata=True,
        using_chat_history=True,
        edge_nodes=["update_submission_qa", "combine_question_evaluator"],
    )

    builder.add_node("fake_node", empty_process)

    # add edges
    builder.add_edge(START, "set_route_to_newroute")
    builder.add_edge("set_route_to_newroute", "remove_keyword")
    builder.add_conditional_edges(
        "remove_keyword", lambda s: s.user_feedback == "", {True: "planner", False: "user_feedback_evaluation"}
    )
    builder.add_conditional_edges(
        "user_feedback_evaluation",
        build_user_feedback_evaluation(),
        {
            "approve": "sending_task",
            "modify": "planner",
            "reject": "stream_suggestion",
            "more_info": "stream_suggestion",
        },
    )
    builder.add_conditional_edges("sending_task", sending_task_to_agent)
    builder.add_edge("combine_question_evaluator", "Evaluator_Agent")
    builder.add_edge("Evaluator_Agent", "report_citations")
    builder.add_edge("report_citations", END)
    builder.add_edge("stream_plan", END)
    builder.add_edge("stream_suggestion", END)

    return builder.compile(debug=debug)

from typing import List

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel import RetryPolicy

from redbox.chains.components import get_structured_response_with_citations_parser
from redbox.chains.parser import ClaudeParser
from redbox.chains.runnables import build_self_route_output_parser
from redbox.graph.edges import (
    build_documents_bigger_than_context_conditional,
    build_keyword_detection_conditional,
    build_total_tokens_request_handler_conditional,
    documents_selected_conditional,
    is_using_search_keyword,
    multiple_docs_in_group_conditional,
    remove_gadget_keyword,
)
from redbox.graph.nodes.processes import (
    build_activity_log_node,
    build_agent,
    build_chat_pattern,
    build_error_pattern,
    build_merge_pattern,
    build_passthrough_pattern,
    build_retrieve_pattern,
    build_set_metadata_pattern,
    build_set_route_pattern,
    build_set_self_route_from_llm_answer,
    build_stuff_pattern,
    clear_documents_process,
    create_evaluator,
    create_planner,
    empty_process,
    lm_choose_route,
    report_sources_process,
)
from redbox.graph.nodes.sends import (
    build_document_chunk_send,
    build_document_group_send,
    build_tool_send,
    sending_task_to_agent,
)
from redbox.graph.nodes.tools import get_log_formatter_for_retrieval_tool
from redbox.models.chain import AgentDecision, PromptSet, RedboxState
from redbox.models.chat import ChatRoute, ErrorRoute
from redbox.models.graph import ROUTABLE_KEYWORDS, RedboxActivityEvent
from redbox.models.prompts import DOCUMENT_AGENT_PROMPT, EXTERNAL_DATA_AGENT
from redbox.transform import structure_documents_by_file_name, structure_documents_by_group_and_indices


def new_root_graph(all_chunks_retriever, parameterised_retriever, metadata_retriever, tools, multi_agent_tools, debug):
    agent_parser = ClaudeParser(pydantic_object=AgentDecision)

    def lm_choose_route_wrapper(state: RedboxState):
        return lm_choose_route(state, parser=agent_parser)

    builder = StateGraph(RedboxState)

    # nodes
    builder.add_node("chat_graph", get_chat_graph(debug=debug))
    builder.add_node("search_graph", get_search_graph(retriever=parameterised_retriever, debug=debug))
    builder.add_node("gadget_graph", get_agentic_search_graph(tools=tools, debug=debug))
    builder.add_node("summarise_graph", get_summarise_graph(all_chunks_retriever=all_chunks_retriever, debug=debug))
    builder.add_node(
        "new_route_graph",
        build_new_graph(multi_agent_tools, debug),
    )
    builder.add_node(
        "retrieve_metadata", get_retrieve_metadata_graph(metadata_retriever=metadata_retriever, debug=debug)
    )

    builder.add_node("is_summarise_route", empty_process)
    builder.add_node("has_keyword", empty_process)
    builder.add_node("is_self_route_enabled", empty_process)
    builder.add_node("any_documents_selected", empty_process)
    builder.add_node("llm_choose_route", empty_process)

    builder.add_node(
        "log_user_request",
        build_activity_log_node(
            lambda s: [
                RedboxActivityEvent(
                    message=f"You selected {len(s.request.s3_keys)} file{"s" if len(s.request.s3_keys)>1 else ""} - {",".join(s.request.s3_keys)}"
                )
                if len(s.request.s3_keys) > 0
                else "You selected no files",
            ]
        ),
    )

    # edges
    builder.add_edge(START, "log_user_request")
    builder.add_edge(START, "retrieve_metadata")
    builder.add_edge("retrieve_metadata", "has_keyword")
    builder.add_conditional_edges(
        "has_keyword",
        build_keyword_detection_conditional(*ROUTABLE_KEYWORDS.keys()),
        {
            ChatRoute.search: "search_graph",
            ChatRoute.gadget: "gadget_graph",
            ChatRoute.newroute: "new_route_graph",
            ChatRoute.summarise: "summarise_graph",
            "DEFAULT": "any_documents_selected",
        },
    )
    builder.add_conditional_edges(
        "any_documents_selected",
        documents_selected_conditional,
        {
            True: "is_self_route_enabled",
            False: "chat_graph",
        },
    )
    builder.add_conditional_edges(
        "is_self_route_enabled",
        lambda s: s.request.ai_settings.self_route_enabled,
        {True: "llm_choose_route", False: "summarise_graph"},
    )
    builder.add_conditional_edges(
        "llm_choose_route",
        lm_choose_route_wrapper,
        {"search": "search_graph", "summarise": "summarise_graph"},
    )

    builder.add_edge("search_graph", "is_summarise_route")
    builder.add_conditional_edges(
        "is_summarise_route", lambda s: s.route_name == ChatRoute.summarise, {True: "summarise_graph", False: END}
    )
    builder.add_edge("search_graph", END)
    builder.add_edge("gadget_graph", END)
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
            final_source_chain=final_sources,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "llm_answer_question",
        build_stuff_pattern(prompt_set=prompt_set, final_response_chain=True),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "set_route_to_summarise",
        build_set_route_pattern(route=ChatRoute.summarise),
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
            output_parser=build_self_route_output_parser(
                match_condition=rag_cannot_answer,
                max_tokens_to_check=4,
                final_response_chain=True,
            ),
            final_response_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "is_self_route_on",
        empty_process,
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

    # Edges
    builder.add_edge(START, "llm_generate_query")
    builder.add_edge("llm_generate_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "is_using_search_keyword")
    builder.add_conditional_edges(
        "is_using_search_keyword", is_using_search_keyword, {True: "llm_answer_question", False: "is_self_route_on"}
    )
    builder.add_conditional_edges(
        "is_self_route_on",
        lambda s: s.request.ai_settings.self_route_enabled,
        {True: "check_if_RAG_can_answer", False: "llm_answer_question"},
    )
    builder.add_edge("llm_answer_question", "set_route_to_search")
    builder.add_edge("check_if_RAG_can_answer", "RAG_cannot_answer")

    builder.add_conditional_edges(
        "RAG_cannot_answer",
        lambda s: rag_cannot_answer(s.last_message),
        {True: "clear_documents", False: "set_route_to_search"},
    )
    builder.add_edge("clear_documents", "set_route_to_summarise")
    builder.add_edge("set_route_to_summarise", END)
    builder.add_edge("set_route_to_search", END)

    return builder.compile(debug=debug)


def get_summarise_graph(all_chunks_retriever: VectorStoreRetriever, debug=True):
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
            final_response_chain=True,
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


def get_self_route_graph(retriever: VectorStoreRetriever, prompt_set: PromptSet, debug: bool = False):
    builder = StateGraph(RedboxState)

    def self_route_question_is_unanswerable(llm_response: str):
        return "unanswerable" in llm_response

    # Processes
    builder.add_node(
        "p_condense_question",
        build_chat_pattern(prompt_set=PromptSet.CondenseQuestion),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_retrieve_docs",
        build_retrieve_pattern(
            retriever=retriever,
            structure_func=structure_documents_by_file_name,
            final_source_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_answer_question_or_decide_unanswerable",
        build_stuff_pattern(
            prompt_set=prompt_set,
            output_parser=build_self_route_output_parser(
                match_condition=self_route_question_is_unanswerable,
                max_tokens_to_check=4,
                final_response_chain=True,
            ),
            final_response_chain=False,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_set_route_name_from_answer",
        build_set_self_route_from_llm_answer(
            self_route_question_is_unanswerable,
            true_condition_state_update={"route_name": ChatRoute.chat_with_docs_map_reduce},
            false_condition_state_update={"route_name": ChatRoute.search},
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_clear_documents",
        clear_documents_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Edges
    builder.add_edge(START, "p_condense_question")
    builder.add_edge("p_condense_question", "p_retrieve_docs")
    builder.add_edge("p_retrieve_docs", "p_answer_question_or_decide_unanswerable")
    builder.add_edge("p_answer_question_or_decide_unanswerable", "p_set_route_name_from_answer")
    builder.add_conditional_edges(
        "p_set_route_name_from_answer",
        lambda state: state.route_name,
        {
            ChatRoute.chat_with_docs_map_reduce: "p_clear_documents",
            ChatRoute.search: END,
        },
    )
    builder.add_edge("p_clear_documents", END)

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


def get_agentic_search_graph(tools: List[StructuredTool], debug: bool = False) -> CompiledGraph:
    """Creates a subgraph for agentic RAG."""

    citations_output_parser, format_instructions = get_structured_response_with_citations_parser()
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node("remove_keyword", remove_gadget_keyword)

    builder.add_node(
        "set_route_to_gadget",
        build_set_route_pattern(route=ChatRoute.gadget),
        retry=RetryPolicy(max_attempts=3),
    )

    def build_smart_agent(state: RedboxState):
        return build_stuff_pattern(
            prompt_set=PromptSet.SearchAgentic,
            tools=tools,
            output_parser=citations_output_parser,
            format_instructions=format_instructions,
            final_response_chain=False,  # Output parser handles streaming
            additional_variables={"has_selected_files": True if state.request.s3_keys else False},
        )

    builder.add_node(
        "smart_agent",
        build_smart_agent,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "invoke_tool_calls",
        ToolNode(tools=tools),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "give_up_agent",
        build_stuff_pattern(prompt_set=PromptSet.GiveUpAgentic, final_response_chain=True),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "report_citations",
        report_sources_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Log
    builder.add_node(
        "log_tool_call_activities",
        build_activity_log_node(
            lambda s: [
                RedboxActivityEvent(message=get_log_formatter_for_retrieval_tool(tool_state_entry).log_call())
                for tool_state_entry in s.last_message.tool_calls
            ]
        ),
        retry=RetryPolicy(max_attempts=3),
    )

    # Decisions
    builder.add_node(
        "should_continue",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Sends
    builder.add_node(
        "send_tool",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Edges
    builder.add_edge(START, "remove_keyword")
    builder.add_edge("remove_keyword", "set_route_to_gadget")
    builder.add_edge("set_route_to_gadget", "should_continue")
    builder.add_conditional_edges(
        "should_continue",
        lambda state: state.steps_left > 8,
        {
            True: "smart_agent",
            False: "give_up_agent",
        },
    )
    builder.add_edge("smart_agent", "send_tool")
    builder.add_edge("send_tool", "report_citations")
    builder.add_edge("smart_agent", "log_tool_call_activities")
    builder.add_conditional_edges("send_tool", build_tool_send("invoke_tool_calls"), path_map=["invoke_tool_calls"])
    builder.add_edge("invoke_tool_calls", "should_continue")
    builder.add_edge("report_citations", END)

    return builder.compile(debug=debug)


def get_chat_with_documents_graph(
    all_chunks_retriever: VectorStoreRetriever,
    parameterised_retriever: VectorStoreRetriever,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a subgraph for chatting with documents."""
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node(
        "p_pass_question_to_text",
        build_passthrough_pattern(),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_set_chat_docs_route",
        build_set_route_pattern(route=ChatRoute.chat_with_docs),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_set_chat_docs_map_reduce_route",
        build_set_route_pattern(route=ChatRoute.chat_with_docs_map_reduce),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_summarise_each_document",
        build_merge_pattern(prompt_set=PromptSet.ChatwithDocsMapReduce),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_summarise_document_by_document",
        build_merge_pattern(prompt_set=PromptSet.ChatwithDocsMapReduce),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_summarise",
        build_stuff_pattern(
            prompt_set=PromptSet.ChatwithDocs,
            final_response_chain=True,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_clear_documents",
        clear_documents_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_too_large_error",
        build_error_pattern(
            text="These documents are too large to work with.",
            route_name=ErrorRoute.files_too_large,
        ),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_answer_or_decide_route",
        get_self_route_graph(parameterised_retriever, PromptSet.SelfRoute),
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "p_retrieve_all_chunks",
        build_retrieve_pattern(
            retriever=all_chunks_retriever,
            structure_func=structure_documents_by_file_name,
            final_source_chain=True,
        ),
        retry=RetryPolicy(max_attempts=3),
    )

    builder.add_node(
        "p_activity_log_tool_decision",
        build_activity_log_node(lambda state: RedboxActivityEvent(message=f"Using _{state.route_name}_")),
        retry=RetryPolicy(max_attempts=3),
    )

    # Decisions
    builder.add_node(
        "d_request_handler_from_total_tokens",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "d_single_doc_summaries_bigger_than_context",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "d_doc_summaries_bigger_than_context",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "d_groups_have_multiple_docs",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "d_self_route_is_enabled",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Sends
    builder.add_node(
        "s_chunk",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "s_group_1",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "s_group_2",
        empty_process,
        retry=RetryPolicy(max_attempts=3),
    )

    # Edges
    builder.add_edge(START, "p_pass_question_to_text")
    builder.add_edge("p_pass_question_to_text", "d_request_handler_from_total_tokens")
    builder.add_conditional_edges(
        "d_request_handler_from_total_tokens",
        build_total_tokens_request_handler_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            "max_exceeded": "p_too_large_error",
            "context_exceeded": "d_self_route_is_enabled",
            "pass": "p_set_chat_docs_route",
        },
    )
    builder.add_conditional_edges(
        "d_self_route_is_enabled",
        lambda s: s.request.ai_settings.self_route_enabled,
        {True: "p_answer_or_decide_route", False: "p_set_chat_docs_map_reduce_route"},
        then="p_activity_log_tool_decision",
    )
    builder.add_conditional_edges(
        "p_answer_or_decide_route",
        lambda state: state.route_name,
        {
            ChatRoute.search: END,
            ChatRoute.chat_with_docs_map_reduce: "p_retrieve_all_chunks",
        },
    )
    builder.add_edge("p_set_chat_docs_route", "p_retrieve_all_chunks")
    builder.add_edge("p_set_chat_docs_map_reduce_route", "p_retrieve_all_chunks")
    builder.add_conditional_edges(
        "p_retrieve_all_chunks",
        lambda s: s.route_name,
        {
            ChatRoute.chat_with_docs: "p_summarise",
            ChatRoute.chat_with_docs_map_reduce: "s_chunk",
        },
    )
    builder.add_conditional_edges(
        "s_chunk",
        build_document_chunk_send("p_summarise_each_document"),
        path_map=["p_summarise_each_document"],
    )
    builder.add_edge("p_summarise_each_document", "d_groups_have_multiple_docs")
    builder.add_conditional_edges(
        "d_groups_have_multiple_docs",
        multiple_docs_in_group_conditional,
        {
            True: "s_group_1",
            False: "d_doc_summaries_bigger_than_context",
        },
    )
    builder.add_conditional_edges(
        "s_group_1",
        build_document_group_send("d_single_doc_summaries_bigger_than_context"),
        path_map=["d_single_doc_summaries_bigger_than_context"],
    )
    builder.add_conditional_edges(
        "d_single_doc_summaries_bigger_than_context",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            True: "p_too_large_error",
            False: "s_group_2",
        },
    )
    builder.add_conditional_edges(
        "s_group_2",
        build_document_group_send("p_summarise_document_by_document"),
        path_map=["p_summarise_document_by_document"],
    )
    builder.add_edge("p_summarise_document_by_document", "d_doc_summaries_bigger_than_context")
    builder.add_conditional_edges(
        "d_doc_summaries_bigger_than_context",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocs),
        {
            True: "p_too_large_error",
            False: "p_summarise",
        },
    )
    builder.add_edge("p_summarise", "p_clear_documents")
    builder.add_edge("p_clear_documents", END)
    builder.add_edge("p_too_large_error", END)

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


def build_new_graph(
    multi_agent_tools: dict,
    debug: bool = False,
) -> CompiledGraph:
    builder = StateGraph(RedboxState)
    builder.add_node("planner", create_planner())
    builder.add_node(
        "Document_Agent",
        build_agent(
            agent_name="Document Agent",
            system_prompt=DOCUMENT_AGENT_PROMPT,
            tools=multi_agent_tools["document_agent"],
            use_metadata=True,
        ),
    )
    builder.add_node("send", empty_process)
    builder.add_node(
        "External_Data_Agent",
        build_agent(
            agent_name="External Data Agent",
            system_prompt=EXTERNAL_DATA_AGENT,
            tools=multi_agent_tools["external_document_agent"],
            use_metadata=False,
        ),
    )
    builder.add_node("Evaluator_Agent", create_evaluator())
    builder.add_node(
        "report_citations",
        report_sources_process,
        retry=RetryPolicy(max_attempts=3),
    )

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", sending_task_to_agent)
    builder.add_edge("Document_Agent", "Evaluator_Agent")
    builder.add_edge("External_Data_Agent", "Evaluator_Agent")
    builder.add_edge("Evaluator_Agent", "report_citations")
    builder.add_edge("report_citations", END)

    return builder.compile(debug=debug)

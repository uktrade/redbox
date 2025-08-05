from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import reduce
from types import UnionType
from typing import Annotated, List, Literal, NotRequired, Required, TypedDict, get_args, get_origin
from uuid import UUID, uuid4

import environ
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingStepsManager
from pydantic import BaseModel, Field, field_validator

from redbox.models import prompts
from redbox.models.chat import DecisionEnum, ToolEnum
from redbox.models.settings import ChatLLMBackend

load_dotenv()
env = environ.Env()


class ChainChatMessage(TypedDict):
    role: Literal["user", "ai", "system"]
    text: str


class AISettings(BaseModel):
    """Prompts and other AI settings"""

    # LLM settings
    context_window_size: int = 128_000
    llm_max_tokens: int = env.int("LLM_MAX_TOKENS", default=1024)

    # Prompts and LangGraph settings
    max_document_tokens: int = 1_000_000
    new_route_enabled: bool = env.bool("NEW_ROUTE_ENABLED", default=False)
    map_max_concurrency: int = 128
    stuff_chunk_context_ratio: float = 0.75
    recursion_limit: int = 50

    # Common Prompt Fragments

    system_info_prompt: str = prompts.SYSTEM_INFO
    persona_info_prompt: str = prompts.PERSONA_INFO
    caller_info_prompt: str = prompts.CALLER_INFO

    # Task Prompt Fragments

    chat_system_prompt: str = prompts.CHAT_SYSTEM_PROMPT
    chat_question_prompt: str = prompts.CHAT_QUESTION_PROMPT
    chat_with_docs_system_prompt: str = prompts.CHAT_WITH_DOCS_SYSTEM_PROMPT
    chat_with_docs_question_prompt: str = prompts.CHAT_WITH_DOCS_QUESTION_PROMPT
    chat_with_docs_reduce_system_prompt: str = prompts.CHAT_WITH_DOCS_REDUCE_SYSTEM_PROMPT
    self_route_system_prompt: str = prompts.SELF_ROUTE_SYSTEM_PROMPT
    retrieval_system_prompt: str = prompts.RETRIEVAL_SYSTEM_PROMPT
    retrieval_question_prompt: str = prompts.RETRIEVAL_QUESTION_PROMPT
    agentic_retrieval_system_prompt: str = prompts.AGENTIC_RETRIEVAL_SYSTEM_PROMPT
    agentic_retrieval_question_prompt: str = prompts.AGENTIC_RETRIEVAL_QUESTION_PROMPT
    agentic_give_up_system_prompt: str = prompts.AGENTIC_GIVE_UP_SYSTEM_PROMPT
    agentic_give_up_question_prompt: str = prompts.AGENTIC_GIVE_UP_QUESTION_PROMPT
    condense_system_prompt: str = prompts.CONDENSE_SYSTEM_PROMPT
    condense_question_prompt: str = prompts.CONDENSE_QUESTION_PROMPT
    chat_map_system_prompt: str = prompts.CHAT_MAP_SYSTEM_PROMPT
    chat_map_question_prompt: str = prompts.CHAT_MAP_QUESTION_PROMPT
    reduce_system_prompt: str = prompts.REDUCE_SYSTEM_PROMPT
    new_route_retrieval_system_prompt: str = prompts.NEW_ROUTE_RETRIEVAL_SYSTEM_PROMPT
    new_route_retrieval_question_prompt: str = prompts.NEW_ROUTE_RETRIEVAL_QUESTION_PROMPT
    llm_decide_route_prompt: str = prompts.LLM_DECIDE_ROUTE
    citation_prompt: str = prompts.CITATION_PROMPT
    planner_system_prompt: str = prompts.PLANNER_PROMPT
    planner_question_prompt: str = prompts.PLANNER_QUESTION_PROMPT
    planner_format_prompt: str = prompts.PLANNER_FORMAT_PROMPT
    answer_instruction_prompt: str = prompts.ANSWER_INSTRUCTION_SYSTEM_PROMPT

    # Elasticsearch RAG and boost values
    rag_k: int = 30
    rag_num_candidates: int = 10
    rag_gauss_scale_size: int = 3
    rag_gauss_scale_decay: float = 0.5
    rag_gauss_scale_min: float = 1.1
    rag_gauss_scale_max: float = 2.0
    elbow_filter_enabled: bool = False
    match_boost: float = 1.0
    match_name_boost: float = 2.0
    match_description_boost: float = 0.5
    match_keywords_boost: float = 0.5
    knn_boost: float = 2.0
    similarity_threshold: float = 0.7

    # this is also the azure_openai_model
    chat_backend: ChatLLMBackend = ChatLLMBackend()

    # settings for tool call
    tool_govuk_retrieved_results: int = 100
    tool_govuk_returned_results: int = 5

    # agents reporting to planner agent
    agents: list = ["Internal_Retrieval_Agent", "External_Retrieval_Agent", "Summarisation_Agent"]
    agents_max_tokens: dict = {
        "Internal_Retrieval_Agent": 10000,
        "External_Retrieval_Agent": 5000,
        "Summarisation_Agent": 20000,
    }


class Source(BaseModel):
    source: str = Field(description="URL or reference to the source", default="")
    source_type: str = Field(description="creator_type of tool", default="Unknown")
    document_name: str = Field(description="Full title from document", default="Unknown")
    highlighted_text_in_source: str = Field(
        description="Direct quote from the provided document (20+ words)", default=""
    )
    page_numbers: list[int] = Field(description="Page Number in document the highlighted text is on", default=[1])
    ref_id: str = Field(
        description="The Reference ID in the format 'ref_N'. Number each quote sequentially starting from ref_1, then ref_2, ref_3, and so on.",
        default="",
    )

    @field_validator("document_name", mode="before")
    def validate_document_name(cls, value):
        if not value:
            return cls.model_fields["document_name"].default
        return value

    @field_validator("source_type", mode="before")
    def validate_source_type(cls, value):
        if not value:
            return cls.model_fields["source_type"].default
        return value

    @field_validator("page_numbers", mode="before")
    def validate_page_numbers(cls, value):
        if not value:
            return cls.model_fields["page_numbers"].default
        else:
            for i, val in enumerate(value):
                if isinstance(val, str):
                    try:
                        value[i] = int(val)
                    except ValueError:
                        value[i] = 1
            return value


class Citation(BaseModel):
    text_in_answer: str = Field(
        description="Part of text from `answer` that references sources and matches exactly with the `answer`, without rephrasing or altering the meaning. Partial matches are acceptable as long as they are exact excerpts from the `answer`",
        default="",
    )
    sources: list[Source] = Field(default_factory=list)


class StructuredResponseWithCitations(BaseModel):
    answer: str = Field(description="Markdown structured answer to the question", default="")
    citations: list[Citation] = Field(default_factory=list)


class StructuredResponseWithStepsTaken(BaseModel):
    output: str = Field(description="Markdown structured answer to the question", default="")
    # sql_query: str = Field(description="The SQL Query used to generate a response", default="")
    reasoning: str = Field(description="The Agent's reasoning", default="")


DocumentMapping = dict[UUID, Document | None]
DocumentGroup = dict[UUID, DocumentMapping | None]


class DocumentState(BaseModel):
    """A document state containing groups of documents."""

    groups: DocumentGroup = Field(default_factory=DocumentGroup)


def document_reducer(current: DocumentState | None, update: DocumentState | list[DocumentState]) -> DocumentState:
    """Merges two document states based on the following rules.

    * Groups are matched by the group key.
    * Documents are matched by the group key and document key.

    Then:

    * If key(s) are matched, the group or Document is replaced
    * If key(s) are matched and the key is None, the key is cleared
    * If key(s) aren't matched, group or Document is added
    """
    # If update is actually a list of state updates, run them one by one
    if isinstance(update, list):
        reduced = reduce(lambda current, update: document_reducer(current, update), update, current)
        return reduced

    # If state is empty, return update
    if current is None:
        return update

    reduced = {k: v.copy() for k, v in current.groups.items()}

    # Update with update
    for group_key, group in update.groups.items():
        # If group is None, remove from output if a group key is matched
        if group is None:
            reduced.pop(group_key, None)
            continue

        # If group key isn't matched, add it
        if group_key not in reduced:
            reduced[group_key] = group.copy()

        for document_key, document in group.items():
            if document is None:
                # If Document is None, remove from output if a group and document key is matched
                reduced[group_key].pop(document_key, None)
            else:
                # Otherwise, update or add the value
                reduced[group_key][document_key] = document

        # Remove group_key from output if it becomes empty after updates
        if not reduced[group_key]:
            del reduced[group_key]

    return DocumentState(groups=reduced)


class RedboxQuery(BaseModel):
    question: str = Field(description="The last user chat message")
    s3_keys: list[str] = Field(description="List of files to process", default_factory=list)
    user_uuid: UUID = Field(description="User the chain in executing for")
    chat_history: list[ChainChatMessage] = Field(description="All previous messages in chat (excluding question)")
    ai_settings: AISettings = Field(description="User request AI settings", default_factory=AISettings)
    permitted_s3_keys: list[str] = Field(description="List of permitted files for response", default_factory=list)


class LLMCallMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    llm_model_name: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}


class RequestMetadata(BaseModel):
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    selected_files_total_tokens: int = 0
    number_of_selected_files: int = 0

    @property
    def input_tokens(self) -> dict[str, int]:
        """
        Creates a dictionary of model names to number of input tokens used
        """
        tokens_by_model = dict()
        for call_metadata in self.llm_calls:
            tokens_by_model[call_metadata.llm_model_name] = (
                tokens_by_model.get(call_metadata.llm_model_name, 0) + call_metadata.input_tokens
            )
        return tokens_by_model

    @property
    def output_tokens(self):
        """
        Creates a dictionary of model names to number of output tokens used
        """
        tokens_by_model = dict()
        for call_metadata in self.llm_calls:
            tokens_by_model[call_metadata.llm_model_name] = (
                tokens_by_model.get(call_metadata.llm_model_name, 0) + call_metadata.output_tokens
            )
        return tokens_by_model


def metadata_reducer(
    current: RequestMetadata | None,
    update: RequestMetadata | list[RequestMetadata] | None,
):
    """Merges two metadata states."""
    # If update is actually a list of state updates, run them one by one
    if isinstance(update, list):
        reduced = reduce(lambda current, update: metadata_reducer(current, update), update, current)
        return reduced

    if current is None:
        return update
    if update is None:
        return current

    return RequestMetadata(
        llm_calls=sorted(set(current.llm_calls) | set(update.llm_calls), key=lambda c: c.timestamp),
        selected_files_total_tokens=update.selected_files_total_tokens or current.selected_files_total_tokens,
        number_of_selected_files=update.number_of_selected_files or current.number_of_selected_files,
    )


agent_options = {agent: agent for agent in AISettings().agents}
AgentEnum = Enum("AgentEnum", agent_options)


class AgentTask(BaseModel):
    task: str = Field(description="Task to be completed by the agent", default="")
    agent: AgentEnum = Field(
        description="Name of the agent to complete the task", default=AgentEnum.Internal_Retrieval_Agent
    )
    expected_output: str = Field(description="What this agent should produce", default="")


class MultiAgentPlan(BaseModel):
    tasks: List[AgentTask] = Field(description="A list of tasks to be carried out by agents", default=[])
    model_config = {"extra": "forbid"}


class RedboxState(BaseModel):
    request: RedboxQuery
    user_feedback: str = ""
    documents: Annotated[DocumentState, document_reducer] = DocumentState()
    route_name: str | None = None
    metadata: Annotated[RequestMetadata | None, metadata_reducer] = None
    citations: list[Citation] | None = None
    steps_left: Annotated[int | None, RemainingStepsManager] = None
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    agents_results: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    agent_plans: MultiAgentPlan | None = None
    tasks_evaluator: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    db_location: str | None = None

    @property
    def last_message(self) -> AnyMessage:
        if not self.messages:
            raise ValueError("No messages in the state")
        return self.messages[-1]


class PromptSet(StrEnum):
    Chat = "chat"
    ChatwithDocs = "chat_with_docs"
    ChatwithDocsMapReduce = "chat_with_docs_map_reduce"
    Search = "search"
    SearchAgentic = "search_agentic"
    GiveUpAgentic = "give_up_agentic"
    SelfRoute = "self_route"
    CondenseQuestion = "condense_question"
    NewRoute = "new_route"
    Planner = "planner"


def get_prompts(state: RedboxState, prompt_set: PromptSet) -> tuple[str, str, str]:
    format_prompt = ""
    if prompt_set == PromptSet.Chat:
        system_prompt = state.request.ai_settings.chat_system_prompt
        question_prompt = state.request.ai_settings.chat_question_prompt
    elif prompt_set == PromptSet.ChatwithDocs:
        system_prompt = state.request.ai_settings.chat_with_docs_system_prompt
        question_prompt = state.request.ai_settings.chat_with_docs_question_prompt
    elif prompt_set == PromptSet.ChatwithDocsMapReduce:
        system_prompt = state.request.ai_settings.chat_map_system_prompt
        question_prompt = state.request.ai_settings.chat_map_question_prompt
    elif prompt_set == PromptSet.Search:
        system_prompt = state.request.ai_settings.retrieval_system_prompt
        question_prompt = state.request.ai_settings.retrieval_question_prompt
        format_prompt = state.request.ai_settings.citation_prompt
    elif prompt_set == PromptSet.SearchAgentic:
        system_prompt = state.request.ai_settings.agentic_retrieval_system_prompt
        question_prompt = state.request.ai_settings.agentic_retrieval_question_prompt
        format_prompt = state.request.ai_settings.citation_prompt
    elif prompt_set == PromptSet.GiveUpAgentic:
        system_prompt = state.request.ai_settings.agentic_give_up_system_prompt
        question_prompt = state.request.ai_settings.agentic_give_up_question_prompt
    elif prompt_set == PromptSet.SelfRoute:
        system_prompt = state.request.ai_settings.self_route_system_prompt
        question_prompt = state.request.ai_settings.retrieval_question_prompt
        format_prompt = state.request.ai_settings.citation_prompt
    elif prompt_set == PromptSet.CondenseQuestion:
        system_prompt = state.request.ai_settings.condense_system_prompt
        question_prompt = state.request.ai_settings.condense_question_prompt
    elif prompt_set == PromptSet.NewRoute:
        system_prompt = state.request.ai_settings.new_route_retrieval_system_prompt
        question_prompt = state.request.ai_settings.new_route_retrieval_question_prompt
        format_prompt = state.request.ai_settings.citation_prompt
    elif prompt_set == PromptSet.Planner:
        system_prompt = state.request.ai_settings.planner_system_prompt
        question_prompt = state.request.ai_settings.planner_question_prompt
        format_prompt = state.request.ai_settings.planner_format_prompt
    return (system_prompt, question_prompt, format_prompt)


def is_dict_type[T](annotated_type: T) -> bool:
    """Unwraps an annotated type to work out if it's a subclass of dict."""
    if get_origin(annotated_type) is Annotated:
        base_type = get_args(annotated_type)[0]
    else:
        base_type = annotated_type

    origin = get_origin(base_type)
    if origin in {Required, NotRequired}:
        base_type = get_args(base_type)[0]

    if origin is UnionType:
        return any(map(is_dict_type, get_args(base_type)))

    return origin is dict or issubclass(base_type, dict)


def dict_reducer(current: dict, update: dict) -> dict:
    """
    Recursively merge two dictionaries:

    * If update has None for a key, current's key will be replaced with None.
    * If both values are dictionaries, they will be merged recursively.
    * Otherwise, the value in update will replace the value in current.
    """
    merged = current.copy()

    for key, new_value in update.items():
        if new_value is None:
            merged[key] = None
        elif isinstance(new_value, dict) and isinstance(merged.get(key), dict):
            merged[key] = dict_reducer(merged[key], new_value)
        else:
            merged[key] = new_value

    return merged


def merge_redbox_state_updates(current: RedboxState, update: RedboxState) -> RedboxState:
    """
    Merge RedboxStates to the following rules, intended for use on state updates.

    * Unannotated items are overwritten but never with None
    * Annotated items apply their reducer function
    * UNLESS they're a dictionary, in which case we use dict_reducer to preserve Nones
    """
    merged_state = current.copy()

    all_keys = set(current.keys()).union(set(update.keys()))

    for update_key in all_keys:
        current_value = current.get(update_key, None)
        update_value = update.get(update_key, None)

        annotation = RedboxState.__annotations__.get(update_key, None)

        if get_origin(annotation) is Annotated:
            if is_dict_type(annotation):
                # If it's annotated and a subclass of dict, apply a custom reducer function
                merged_state[update_key] = dict_reducer(current=current_value or {}, update=update_value or {})
            elif current_value is None:
                merged_state[update_key] = update_value
            elif update_value is None:
                merged_state[update_key] = current_value
            else:
                # If it's annotated and not a dict, apply its reducer function
                _, reducer_func = get_args(annotation)
                merged_state[update_key] = reducer_func(current_value, update_value)
        else:
            # If not annotated, replace but don't overwrite an existing value with None
            if update_value is not None:
                merged_state[update_key] = update_value
            else:
                merged_state[update_key] = current_value

    return merged_state


class GeneratedMetadata(BaseModel):
    """Document Metadata generated by the LLM"""

    name: str = Field(description="document name", default="")
    description: str | None = Field(description="document description", default="")
    keywords: list[str] = Field(description="document keywords", default_factory=list)


class AgentDecision(BaseModel):
    next: ToolEnum = ToolEnum.search


class FeedbackEvalDecision(BaseModel):
    next: DecisionEnum = DecisionEnum.approve


def get_plan_fix_prompts():
    suffix_texts = [
        "Please let me know if you want me to go ahead with the plan, or make any changes.",
        "Let me know if you would like to proceed, or you can also ask me to make changes.",
        "If you're happy with this approach let me know, or you can change the approach also.",
        "Let me know if you'd like me to proceed, or if you want to amend or change the plan.",
    ]
    prefix_texts = [
        "Here is the plan I will execute:",
        "Here is my proposed plan:",
        "I can look into this for you, here's my current plan:",
        "Sure, here's my current plan:",
    ]
    return (prefix_texts, suffix_texts)


def get_plan_fix_suggestion_prompts():
    return [
        "It looks like you do not want to go ahead with the plan. Please let me know how I can help.",
        "Okay, no problem. The plan has been cancelled.",
        "I've stopped that task as requested. Let me know if you need anything else.",
        "Cancellation confirmed. What would you like to do next?",
    ]

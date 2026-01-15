import logging
import re
from typing import Any, Callable, Iterable, Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun, dispatch_custom_event
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough, chain

from redbox.api.format import format_documents
from redbox.chains.activity import log_activity
from redbox.chains.components import (
    get_basic_metadata_retriever,
    get_chat_llm,
    get_knowledge_base_metadata_retriever,
    get_tokeniser,
)
from redbox.models.chain import ChainChatMessage, PromptSet, RedboxState, get_prompts
from redbox.models.errors import QuestionLengthError
from redbox.models.graph import RedboxEventType
from redbox.models.settings import ChatLLMBackend, get_settings
from redbox.transform import bedrock_tokeniser, flatten_document_state, get_all_metadata

log = logging.getLogger()
re_string_pattern = re.compile(r"(\S+)")


def prompt_budget_calculation(state: RedboxState, prompts_budget: int):
    ai_settings = state.request.ai_settings
    chat_history_budget = ai_settings.context_window_size - ai_settings.llm_max_tokens - prompts_budget
    if chat_history_budget <= 0:
        raise QuestionLengthError
    return chat_history_budget


def truncate_chat_history(state: RedboxState, prompts_budget: int, tokeniser: callable = bedrock_tokeniser):
    _tokeniser = tokeniser or get_tokeniser()
    chat_history_budget = prompt_budget_calculation(state, prompts_budget)
    truncated_history: list[ChainChatMessage] = []
    for msg in state.request.chat_history[::-1]:
        chat_history_budget -= _tokeniser(msg["text"])
        if chat_history_budget <= 0:
            break
        else:
            truncated_history.insert(0, msg)

    return truncated_history


def build_chat_prompt_from_messages_runnable(
    prompt_set: PromptSet,
    tokeniser: callable = bedrock_tokeniser,
    format_instructions: str = "",
    additional_variables: dict | None = None,
) -> Runnable:
    @chain
    def _chat_prompt_from_messages(state: RedboxState) -> Runnable:
        """
        Create a ChatPromptTemplate as part of a chain using 'chat_history'.
        Returns the PromptValue using values in the input_dict
        """
        ai_settings = state.request.ai_settings
        _tokeniser = tokeniser or get_tokeniser()
        _additional_variables = additional_variables or dict()
        task_system_prompt, task_question_prompt, format_prompt = get_prompts(state, prompt_set)

        system_info_prompt = ai_settings.system_info_prompt.replace(
            "{built_in_tools}",
            "\n".join(
                [f"- {agent.name.removesuffix('_Agent')}: {agent.description}" for agent in ai_settings.worker_agents]
            ),
        )

        in_default_mode = all(agent.default_agent for agent in ai_settings.worker_agents)
        if not in_default_mode:
            # -- In Tool Mode --
            agent_names = ",".join([f"'{agent.name.removesuffix('_Agent')}'" for agent in ai_settings.worker_agents])
            system_info_prompt = system_info_prompt.replace(
                "{knowledge_mode}",
                f"You are in tool mode using: {agent_names}. Make sure to tell the user when stating your capabilities or responding to a greeting, do not mention capabilities outside the scope of this tool. If a user requests you to perform a capability outside of this tool advise them to use normal chat. ",
            )
        else:
            # -- Default --
            system_info_prompt = system_info_prompt.replace(
                "{knowledge_mode}",
                "Your default capability is to act as a multi-agent planner to detect the user's intent and find relevant information in parallel using agents for search, summarisation, and searching Welcome to GOV.UK  and/or wikipedia, and you will provide a plan to the user when you invoke 2 or more tools to generate a response to a query. ",
            )

        log.debug("Setting chat prompt")
        # Set the system prompt to be our composed structure
        # We preserve the format instructions
        system_prompt_message = f"""
            {system_info_prompt}
            {task_system_prompt}
            {ai_settings.persona_info_prompt if in_default_mode else ""}
            {ai_settings.caller_info_prompt}
            {ai_settings.answer_instruction_prompt}
            """
        prompts_budget = bedrock_tokeniser(task_system_prompt) + bedrock_tokeniser(task_question_prompt)

        truncated_history = truncate_chat_history(state=state, prompts_budget=prompts_budget, tokeniser=_tokeniser)

        prompt_template_context = (
            state.request.model_dump()
            | {
                "messages": state.messages,
                "formatted_documents": format_documents(flatten_document_state(state.documents)),
            }
            | _additional_variables
        )

        chatprompt = ChatPromptTemplate(
            messages=(
                [("system", system_prompt_message)]
                + [(msg["role"], msg["text"]) for msg in truncated_history]
                + [MessagesPlaceholder("messages")]
                + [("human", task_question_prompt)]
                + ([("human", format_prompt)] if len(format_instructions) > 0 else [])
            ),
            partial_variables={"format_instructions": format_instructions},
        ).invoke(prompt_template_context)

        return chatprompt

    return _chat_prompt_from_messages


@chain
def final_response_if_needed(input_: dict) -> Runnable:
    model_name = input_.get("metadata").llm_calls[0].llm_model_name
    need_log = None
    if not input_["final_chain"]:
        need_log = False
    else:
        if input_.get("messages")[-1].tool_calls:
            need_log = False
        else:
            need_log = True

    return RunnablePassthrough.assign(
        _log=RunnableLambda(lambda _: (log_activity(f"Generating response with {model_name}...") if need_log else None))
    )


def build_llm_chain(
    prompt_set: PromptSet,
    llm: BaseChatModel,
    output_parser: Runnable | Callable = None,
    format_instructions: str = "",
    final_response_chain: bool = False,
    additional_variables: dict = {},
    summary_multiagent_flag: bool = False,
) -> Runnable:
    """Builds a chain that correctly forms a text and metadata state update.

    Permits both invoke and astream_events.
    """
    model_name = llm._default_config.get("model", "unknown")
    _llm = llm.with_config(tags=["response_flag"]) if final_response_chain else llm
    _llm = (
        _llm.with_config(tags=["summary_multiagent_tag"]) if summary_multiagent_flag else _llm
    )  # used for summarisation in multi-agent route
    _output_parser = output_parser if output_parser else StrOutputParser()

    _llm_text_and_tools = _llm | {
        "raw_response": RunnablePassthrough(),
        "parsed_response": _output_parser,
    }

    text_and_tools = {
        "text_and_tools": _llm_text_and_tools,
        "prompt": RunnableLambda(lambda prompt: prompt.to_string()),
        "model": lambda _: model_name,
        "final_chain": lambda _: final_response_chain,
    }

    return (
        build_chat_prompt_from_messages_runnable(
            prompt_set, format_instructions=format_instructions, additional_variables=additional_variables
        )
        | text_and_tools
        | get_all_metadata
        | final_response_if_needed
    )


def build_self_route_output_parser(
    match_condition: Callable[[str], bool],
    max_tokens_to_check: int,
    parser=None,
) -> Runnable[Iterable[AIMessageChunk], Iterable[str]]:
    """
    This Runnable reads the streamed responses from an LLM until the match
    condition is true for the response so far it has read a number of tokens.
    If the match condition is true it breaks off and returns nothing to the
    client, if not then it streams the response to the client as normal.

    Used to handle responses from prompts like 'If this question can be
    answered answer it, else return False'
    """

    def _self_route_output_parser(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
        current_content = ""
        token_count = 0
        for chunk in chunks:
            current_content += chunk.content
            token_count += 1
            if match_condition(current_content):
                yield current_content
                return
            elif token_count > max_tokens_to_check:
                break
        yield current_content
        for chunk in chunks:
            yield chunk.content

    return _self_route_output_parser | parser


@RunnableLambda
def send_token_events(tokens: str):
    dispatch_custom_event(RedboxEventType.response_tokens, data=tokens)


class CannedChatLLM(BaseChatModel):
    """A custom chat model that returns its text as if an LLM returned it.

    Based on https://python.langchain.com/v0.2/docs/how_to/custom_chat_model/
    """

    messages: list[AIMessage]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run the LLM on the given input.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        message = AIMessage(content=self.messages[-1].content)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the LLM on the given prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        for token in re_string_pattern.split(self.messages[-1].content):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Final token should be empty
        chunk = ChatGenerationChunk(message=AIMessageChunk(content=""))
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)

        yield chunk

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CannedChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "canned"


def basic_chat_chain(
    system_prompt,
    tools=None,
    _additional_variables: dict = {},
    parser=None,
    using_only_structure: bool = False,
    using_chat_history: bool = False,
    model: ChatLLMBackend | None = None,
):
    @chain
    def _basic_chat_chain(state: RedboxState):
        nonlocal parser
        if model is not None:
            llm = get_chat_llm(model, tools=tools)
        else:  # use default
            if tools:
                llm = get_chat_llm(state.request.ai_settings.chat_backend, tools=tools)
            else:
                llm = get_chat_llm(state.request.ai_settings.chat_backend)

        # chat history
        if using_chat_history:
            _tokeniser = get_tokeniser()
            prompts_budget = _tokeniser(system_prompt)
            truncated_history = truncate_chat_history(state=state, prompts_budget=prompts_budget, tokeniser=_tokeniser)

        context = {
            "question": state.request.question,
            "chat_history": truncated_history if using_chat_history else "",
        } | _additional_variables
        if parser:
            if isinstance(parser, StrOutputParser):
                prompt = ChatPromptTemplate([(system_prompt)])
            else:
                format_instructions = parser.get_format_instructions()
                prompt = ChatPromptTemplate(
                    [(system_prompt)], partial_variables={"format_instructions": format_instructions}
                )
            if using_only_structure:
                chain = prompt | llm
            else:
                chain = prompt | llm | parser
        else:
            prompt = ChatPromptTemplate([(system_prompt)])
            chain = prompt | llm
        return chain.invoke(context)

    return _basic_chat_chain


def chain_use_metadata(
    system_prompt: str,
    parser=None,
    tools=None,
    _additional_variables: dict = {},
    using_only_structure=False,
    using_chat_history=False,
    model: ChatLLMBackend | None = None,
    use_knowledge_base=False,
):
    @chain
    def get_metadata(state: RedboxState):
        metadata = get_basic_metadata_retriever(get_settings()).invoke(state)
        return metadata

    @chain
    def get_knowledge_base_metadata(state: RedboxState):
        knowledge_base_metadata = get_knowledge_base_metadata_retriever(get_settings()).invoke(state)
        return knowledge_base_metadata

    @chain
    def use_result(input):
        if input["metadata"] is not None:
            additional_variables = {"metadata": input["metadata"]}
        if input["knowledge_base_metadata"] is not None:
            additional_variables["knowledge_base_metadata"] = input["knowledge_base_metadata"]
        if _additional_variables:
            additional_variables = dict(additional_variables, **_additional_variables)
        chain = basic_chat_chain(
            system_prompt=system_prompt,
            tools=tools,
            parser=parser,
            _additional_variables=additional_variables,
            using_only_structure=using_only_structure,
            using_chat_history=using_chat_history,
            model=model,
        )
        return chain.invoke(input["state"])

    return (
        get_metadata | use_result
        if not use_knowledge_base
        else RunnableParallel(
            state=RunnablePassthrough(), metadata=get_metadata, knowledge_base_metadata=get_knowledge_base_metadata
        )
        | use_result
    )


def create_chain_agent(
    system_prompt,
    use_metadata=False,
    tools=None,
    parser=None,
    _additional_variables: dict = {},
    using_only_structure=False,
    using_chat_history=False,
    model: ChatLLMBackend | None = None,
    use_knowledge_base=False,
):
    if use_metadata or use_knowledge_base:
        return chain_use_metadata(
            system_prompt=system_prompt,
            tools=tools,
            parser=parser,
            _additional_variables=_additional_variables,
            using_only_structure=using_only_structure,
            using_chat_history=using_chat_history,
            model=model,
            use_knowledge_base=use_knowledge_base,
        )
    else:
        return basic_chat_chain(
            system_prompt=system_prompt,
            tools=tools,
            parser=parser,
            _additional_variables=_additional_variables,
            using_only_structure=using_only_structure,
            using_chat_history=using_chat_history,
            model=model,
        )

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import List

from langchain.schema import StrOutputParser
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from pydantic import BaseModel, Field

from redbox.chains.components import (
    get_basic_metadata_retriever,
    get_chat_llm,
    get_embeddings,
    get_structured_response_with_citations_parser,
)
from redbox.chains.parser import ClaudeParser
from redbox.graph.nodes.processes import (
    PromptSet,
    build_activity_log_node,
    build_set_route_pattern,
    build_stuff_pattern,
    empty_process,
    report_sources_process,
)
from redbox.graph.nodes.sends import _copy_state
from redbox.graph.nodes.tools import build_govuk_search_tool, build_search_documents_tool, build_search_wikipedia_tool
from redbox.models.chain import PromptSet, RedboxState
from redbox.models.chat import ChatRoute
from redbox.models.file import ChunkResolution
from redbox.models.graph import RedboxActivityEvent
from redbox.models.prompts import DOCUMENT_AGENT_PROMPT, EXTERNAL_DATA_AGENT, PLANNER_PROMPT
from redbox.models.settings import Settings, get_settings


def test_graph():
    def basic_chat_chain(
        system_prompt, tools=None, _additional_variables: dict = {}, parser=None, using_only_structure=False
    ):
        @as_runnable
        def _basic_chat_chain(state: RedboxState):
            nonlocal parser
            if tools:
                llm = get_chat_llm(state.request.ai_settings.chat_backend, tools=tools)
            else:
                llm = get_chat_llm(state.request.ai_settings.chat_backend)
            context = {
                "question": state.request.question,
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
        system_prompt: str, parser=None, tools=None, _additional_variables: dict = {}, using_only_structure=False
    ):
        metadata = None

        @as_runnable
        def get_metadata(state: RedboxState):
            nonlocal metadata
            env = get_settings()
            retriever = get_basic_metadata_retriever(env)
            metadata = retriever.invoke(state)
            return state

        @as_runnable
        def use_result(state: RedboxState):
            additional_variables = {"metadata": metadata}
            if _additional_variables:
                additional_variables = dict(additional_variables, **_additional_variables)
                print(additional_variables)
            chain = basic_chat_chain(
                system_prompt=system_prompt,
                tools=tools,
                parser=parser,
                _additional_variables=additional_variables,
                using_only_structure=using_only_structure,
            )
            return chain.invoke(state)

        return get_metadata | use_result

    def create_agent(
        system_prompt,
        use_metadata=False,
        tools=None,
        parser=None,
        _additional_variables: dict = {},
        using_only_structure=False,
    ):
        if use_metadata:
            return chain_use_metadata(
                system_prompt=system_prompt,
                tools=tools,
                parser=parser,
                _additional_variables=_additional_variables,
                using_only_structure=using_only_structure,
            )
        else:
            return basic_chat_chain(
                system_prompt=system_prompt,
                tools=tools,
                parser=parser,
                _additional_variables=_additional_variables,
                using_only_structure=using_only_structure,
            )

    agents = ["Document_Agent", "External_Data_Agent"]

    # create options map for the supervisor output parser.
    agent_options = {agent: agent for agent in agents}

    # create Enum object
    AgentEnum = Enum("AgentEnum", agent_options)

    class AgentTask(BaseModel):
        task: str = Field(description="Task to be completed by the agent", default="")
        agent: AgentEnum = Field(description="Name of the agent to complete the task", default=AgentEnum.Document_Agent)
        # input: str = Field(description = "What information will be provided to this agent", default = "")
        expected_output: str = Field(description="What this agent should produce", default="")
        # Purpose: str = Field(description = "How this output contributes to the overall goal", default = "")

    class MultiAgentPlan(BaseModel):
        tasks: List[AgentTask] = Field(description="A list of tasks to be carried out by agents", default=[])

    # tools
    env = Settings()
    search_documents = build_search_documents_tool(
        es_client=env.elasticsearch_client(),
        index_name=env.elastic_chunk_alias,
        embedding_model=get_embeddings(env),
        embedding_field_name=env.embedding_document_field_name,
        chunk_resolution=ChunkResolution.normal,
    )

    search_wikipedia = build_search_wikipedia_tool()
    search_govuk = build_govuk_search_tool()

    def run_tools_parallel(ai_msg, tools, state):
        # Create a list to store futures
        futures = []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            # Submit tool invocations to the executor
            for tool_call in ai_msg.tool_calls:
                # Find the matching tool by name
                selected_tool = next((tool for tool in tools if tool.name == tool_call.get("name")), None)

                if selected_tool is None:
                    print(f"Warning: No tool found for {tool_call.get('name')}")
                    continue

                # Get arguments and submit the tool invocation
                args = tool_call.get("args", {})
                args["state"] = state
                future = executor.submit(selected_tool.invoke, args)
                futures.append(future)

            # Collect responses as tools complete
            responses = []
            for future in as_completed(futures):
                try:
                    response = future.result()
                    responses.append(AIMessage(response))
                except Exception as e:
                    print(f"Tool invocation error: {e}")

            return responses

    def sending_task(state: RedboxState):
        parser = ClaudeParser(pydantic_object=MultiAgentPlan)
        plan = parser.parse(state.last_message.content)
        task_send_states: list[RedboxState] = [
            (task.agent.value, _copy_state(state, messages=[AIMessage(content=task.model_dump_json())]))
            for task in plan.tasks
        ]
        return [Send(node=target, arg=state) for target, state in task_send_states]

    def create_planner(state: RedboxState):
        orchestration_agent = create_agent(
            system_prompt=PLANNER_PROMPT,
            use_metadata=True,
            tools=None,
            parser=ClaudeParser(pydantic_object=MultiAgentPlan),
            using_only_structure=True,
        )
        res = orchestration_agent.invoke(state)
        state.messages.append(AIMessage(res.content))
        return state

    def create_evaluator(state: RedboxState):
        citation_parser, format_instructions = get_structured_response_with_citations_parser()
        evaluator_agent = build_stuff_pattern(
            prompt_set=PromptSet.NewRoute,
            tools=None,
            output_parser=citation_parser,
            format_instructions=format_instructions,
            final_response_chain=False,
        )
        return evaluator_agent

    def build_document_agent(state: RedboxState):
        tools = [search_documents]
        parser = ClaudeParser(pydantic_object=AgentTask)
        try:
            task = parser.parse(state.last_message.content)
        except Exception as e:
            print(f"Cannot parse in document agent: {e}")
        activity_node = build_activity_log_node(
            RedboxActivityEvent(message=f"Document Agent is completing task: {task.task}")
        )
        activity_node.invoke(state)

        doc_agent = create_agent(
            system_prompt=DOCUMENT_AGENT_PROMPT,
            use_metadata=True,
            parser=None,
            tools=tools,
            _additional_variables={"task": task.task, "expected_output": task.expected_output},
        )
        ai_msg = doc_agent.invoke(state)
        result = run_tools_parallel(ai_msg, tools, state)
        return {"messages": result}

    def build_external_data_agent(state: RedboxState):
        tools = [search_govuk, search_wikipedia]
        parser = ClaudeParser(pydantic_object=AgentTask)
        try:
            task = parser.parse(state.last_message.content)
        except Exception as e:
            print(f"Cannot parse in document agent: {e}")
        activity_node = build_activity_log_node(
            RedboxActivityEvent(message=f"External Data Agent is completing task: {task.task}")
        )
        activity_node.invoke(state)
        ext_agent = create_agent(
            system_prompt=EXTERNAL_DATA_AGENT,
            use_metadata=False,
            parser=None,
            tools=tools,
            _additional_variables={"task": task.task, "expected_output": task.expected_output},
        )
        ai_msg = ext_agent.invoke(state)
        result = run_tools_parallel(ai_msg, tools, state)
        return {"messages": result}

    builder = StateGraph(RedboxState)
    builder.add_node("planner", create_planner)
    builder.add_node("Document_Agent", build_document_agent)
    builder.add_node("send", empty_process)
    builder.add_node("External_Data_Agent", build_external_data_agent)
    builder.add_node("Evaluator_Agent", create_evaluator)
    builder.add_node(
        "report_citations",
        report_sources_process,
        retry=RetryPolicy(max_attempts=3),
    )
    builder.add_node(
        "set_route_to_new_route",
        build_set_route_pattern(route=ChatRoute.newroute),
        retry=RetryPolicy(max_attempts=3),
    )

    builder.add_edge(START, "set_route_to_new_route")
    builder.add_edge("set_route_to_new_route", "planner")
    builder.add_conditional_edges("planner", sending_task)
    builder.add_edge("Document_Agent", "Evaluator_Agent")
    builder.add_edge("External_Data_Agent", "Evaluator_Agent")
    builder.add_edge("Evaluator_Agent", "report_citations")
    builder.add_edge("report_citations", END)

    return builder.compile()

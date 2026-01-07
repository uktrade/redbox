from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from redbox import Redbox
from redbox.graph.nodes.sends import run_tools_parallel
from redbox.models.chain import (
    ChainChatMessage,
    Citation,
    MultiAgentPlanBase,
    RedboxQuery,
    RedboxState,
    RequestMetadata,
    Source,
    StructuredResponseWithCitations,
    configure_agent_task_plan,
    metadata_reducer,
)
from redbox.models.chat import ChatRoute
from redbox.models.graph import RedboxActivityEvent
from redbox.models.settings import Settings
from redbox.test.data import (
    GenericFakeChatModelWithTools,
    RedboxChatTestCase,
    RedboxTestData,
    generate_test_cases,
    mock_all_chunks_retriever,
    mock_metadata_retriever,
    mock_parameterised_retriever,
    mock_tabular_retriever,
)


def run_assertion(
    test_case,
    final_state: RedboxState,
    route_name: str | None,
    token_events: list,
    metadata_events: list,
    activity_events: list,
    document_events: list,
):
    # Assertions
    assert route_name is not None, f"No Route Name event fired! - Final State: {final_state}"

    # Bit of a bodge to retain the ability to check that the LLM streaming is working in most cases
    if not route_name.startswith("error"):
        metadata_events = metadata_events[-1:]  # Hack for tabular agent: check final response
        assert len(token_events) > 1, (
            f"Expected tokens as a stream. Received: {token_events}"
        )  # Temporarily turning off streaming check
        assert len(metadata_events) == len(test_case.test_data.llm_responses), (
            f"Expected {len(test_case.test_data.llm_responses)} metadata events. Received {len(metadata_events)}"
        )

    assert test_case.test_data.expected_activity_events(activity_events), (
        f"Activity events not as expected. Received: {activity_events}"
    )

    llm_response = "".join(token_events)
    number_of_selected_files = len(test_case.query.s3_keys)
    metadata_response = metadata_reducer(
        RequestMetadata(
            selected_files_total_tokens=0
            if number_of_selected_files == 0
            else (int(test_case.test_data.tokens_in_all_docs / number_of_selected_files) * number_of_selected_files),
            number_of_selected_files=number_of_selected_files,
        ),
        metadata_events,
    )

    expected_text = (
        test_case.test_data.expected_text
        if test_case.test_data.expected_text is not None
        else test_case.test_data.llm_responses[-1]
    )
    expected_text = expected_text.content if isinstance(expected_text, AIMessage) else expected_text

    assert final_state.last_message.content == llm_response, (
        f"Text response from streaming: '{llm_response}' did not match final state text '{final_state.last_message.content}'"
    )
    assert final_state.last_message.content == expected_text, (
        f"Expected text: '{expected_text}' did not match received text '{final_state.last_message.content}'"
    )

    assert final_state.route_name == test_case.test_data.expected_route, (
        f"Expected Route: '{test_case.test_data.expected_route}'. Received '{final_state.route_name}'"
    )
    if metadata := final_state.metadata:
        assert metadata == metadata_response, f"Expected metadata: '{metadata_response}'. Received '{metadata}'"
    for document_list in document_events:
        for document in document_list:
            assert document in test_case.docs, f"Document not in test case docs: {document}"


async def run_app(
    test_case: RedboxChatTestCase,
    user_feedback: str = "",
    agent_plans: MultiAgentPlanBase | None = None,
):
    env = Settings()
    # Instantiate app
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever(test_case.docs),
        parameterised_retriever=mock_parameterised_retriever(test_case.docs),
        tabular_retriever=mock_tabular_retriever(test_case.docs),
        metadata_retriever=mock_metadata_retriever(
            [d for d in test_case.docs if d.metadata["uri"] in test_case.query.s3_keys]
        ),
        env=env,
        debug=True,
    )

    # Define callback functions
    token_events = []
    metadata_events = []
    activity_events = []
    document_events = []
    route_name = None

    async def streaming_response_handler(tokens: str):
        token_events.append(tokens)

    async def metadata_response_handler(metadata: dict):
        metadata_events.append(metadata)

    async def streaming_route_name_handler(route: str):
        nonlocal route_name
        route_name = route

    async def streaming_activity_handler(activity_event: RedboxActivityEvent):
        activity_events.append(activity_event)

    async def documents_response_handler(documents: list[Document]):
        document_events.append(documents)

    # Run the app
    final_state = await app.run(
        input=RedboxState(
            request=test_case.query,
            agent_plans=agent_plans,
            user_feedback=user_feedback,
        ),
        response_tokens_callback=streaming_response_handler,
        metadata_tokens_callback=metadata_response_handler,
        route_name_callback=streaming_route_name_handler,
        activity_event_callback=streaming_activity_handler,
        documents_callback=documents_response_handler,
    )

    # assertion
    run_assertion(test_case, final_state, route_name, token_events, metadata_events, activity_events, document_events)


ANSWER_NO_CITATION = AIMessage(
    content=StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json()
)
ANSWER_WITH_CITATION = AIMessage(
    content=StructuredResponseWithCitations(
        answer="AI is a lie",
        citations=[
            Citation(
                text_in_answer="AI is a lie I made up",
                sources=[
                    Source(
                        source="SomeAIGuy",
                        document_name="http://localhost/someaiguy.html",
                        highlighted_text_in_source="I lied about AI",
                        page_numbers=[1],
                    )
                ],
            )
        ],
    ).model_dump_json()
)

WORKER_RESPONSE = AIMessage(
    content="I will be calling a tool",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_e4003b",
                "function": {"arguments": '{\n  "query": "ai"\n}', "name": "_some_tool"},
                "type": "function",
            }
        ]
    },
)

WORKER_TOOL_RESPONSE = AIMessage(content="this work is done by worker")

TABULAR_TOOL_RESPONSE = AIMessage(content=["this work is done by worker", "pass", "False"])


class TestNewRoutes:
    def create_new_route_test(
        self, question: str, number_of_docs: int = 0, tokens_in_all_docs: int = 0, chat_history: list = []
    ):
        return generate_test_cases(
            query=RedboxQuery(
                question=question, s3_keys=[], user_uuid=uuid4(), chat_history=chat_history, permitted_s3_keys=[]
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=number_of_docs,
                    tokens_in_all_docs=tokens_in_all_docs,
                    llm_responses=["AI is a lie"],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.newroute,
                )
            ],
            test_id="fake case",
        )[0]

    @pytest.mark.parametrize(
        "test_name, user_prompt, documents, has_task, agent, evaluator",
        [
            ("no document no task", "What is AI", [0, 0], False, None, ANSWER_NO_CITATION),
            (
                "no document one task",
                "What is AI",
                [0, 0],
                True,
                "Web_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with single doc",
                "What is AI",
                [1, 1000],
                True,
                "Web_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "no document one task",
                "tell me about AI legislation",
                [0, 0],
                True,
                "Legislation_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with single doc",
                "tell me about AI legislation",
                [1, 1000],
                True,
                "Legislation_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with multiple doc one task",
                "What is AI",
                [3, 10_000],
                True,
                "Internal_Retrieval_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with tabular doc one task",
                "What is AI",
                [1, 1000],
                True,
                "Tabular_Agent",
                ANSWER_NO_CITATION,
            ),
            (
                "no such keyword no doc",
                "@nosuschkeyword what is 2+2?",
                [0, 0],
                False,
                None,
                ANSWER_NO_CITATION,
            ),
            (
                "no such keyword no doc",
                "@nosuschkeyword What is AI",
                [0, 0],
                True,
                "Web_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "no such keyword with doc",
                "@nosuschkeyword What is AI",
                [3, 10_000],
                True,
                "Internal_Retrieval_Agent",
                ANSWER_WITH_CITATION,
            ),
            (
                "no such keyword no doc",
                "@nosuschkeyword tell me about AI legislation",
                [0, 0],
                True,
                "Legislation_Search_Agent",
                ANSWER_WITH_CITATION,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_newroute_zero_or_one_task(
        self, test_name, user_prompt, documents, has_task, agent, evaluator, mocker: MockerFixture
    ):
        test_case = self.create_new_route_test(
            question=user_prompt, number_of_docs=documents[0], tokens_in_all_docs=documents[1]
        )
        # mocking planner agent
        agent_task, multi_agent_plan = configure_agent_task_plan({agent: agent})
        tasks = [agent_task()] if has_task else []
        configured_multi_agent_plan = multi_agent_plan().model_copy(update={"tasks": tasks})
        planner = configured_multi_agent_plan.model_dump_json()
        planner_response = GenericFakeChatModelWithTools(messages=iter([planner]))
        planner_response._default_config = {"model": "bedrock"}

        evaluator_response = GenericFakeChatModelWithTools(messages=iter([evaluator]))
        evaluator_response._default_config = {"model": "bedrock"}

        # mock response from worker agent
        worker_response = GenericFakeChatModelWithTools(messages=iter([WORKER_RESPONSE]))
        worker_response._default_config = {"model": "bedrock"}

        mock_chat_chain = mocker.patch("redbox.chains.runnables.get_chat_llm")
        mock_chat_chain.side_effect = [planner_response, worker_response]

        # mock tool call
        if agent == "Internal_Retrieval_Agent":
            # This is a mocker for the new agent refactor. You will need to remove other mocking once all agents have been refactored.
            mocker.patch(
                "redbox.graph.agents.workers.run_tools_parallel",
                return_value=[WORKER_TOOL_RESPONSE],
            )
            mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=evaluator_response)
        else:
            if agent == "Tabular_Agent":
                tool_response = TABULAR_TOOL_RESPONSE
                mocker.patch(
                    "redbox.graph.nodes.processes.get_chat_llm", side_effect=[worker_response, evaluator_response]
                )

            else:
                tool_response = WORKER_TOOL_RESPONSE
                mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=evaluator_response)

            mocker.patch("redbox.graph.nodes.processes.run_tools_parallel", return_value=[tool_response])

        await run_app(test_case)

        if has_task and agent != "Tabular_Agent":
            assert mock_chat_chain.call_count == 2

    @pytest.mark.parametrize(
        "test_name, user_feedback, agents, evaluator",
        [
            (
                "approve plan",
                '{"next": "approve"}',
                ["External_Retrieval_Agent", "Web_Search_Agent"],
                ANSWER_WITH_CITATION,
            ),
            (
                "modify plan",
                '{"next": "modify"}',
                ["External_Retrieval_Agent", "Web_Search_Agent"],
                ANSWER_WITH_CITATION,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_newroute_more_than_one_tasks(
        self, test_name, user_feedback, agents, evaluator, mocker: MockerFixture
    ):
        # mocking user feedback classification
        feedback_class_response = GenericFakeChatModelWithTools(messages=iter([AIMessage(content=user_feedback)]))
        feedback_class_response._default_config = {"model": "bedrock"}
        side_effect = [feedback_class_response]

        # mocking planner agent with tasks
        tasks = []
        for i in range(len(agents)):
            agent = agents[i]
            agent_task, multi_agent_plan = configure_agent_task_plan({agent: agent})
            tasks += [agent_task()]
        configured_multi_agent_plan = multi_agent_plan().model_copy(update={"tasks": tasks})
        old_plan = []
        if "modify" in user_feedback:
            # old plan here
            old_plan = [
                ChainChatMessage(
                    role="ai",
                    text=(configured_multi_agent_plan.model_dump_json()).replace("{", "{{").replace("}", "}}"),
                )
            ]

            planner_response = GenericFakeChatModelWithTools(
                messages=iter([configured_multi_agent_plan.model_dump_json()])
            )
            planner_response._default_config = {"model": "bedrock"}
            side_effect += [planner_response]

        # mock response from worker agents
        tool_call_side_effect = []
        for i in range(len(agents)):
            worker_response = GenericFakeChatModelWithTools(messages=iter([WORKER_RESPONSE]))
            worker_response._default_config = {"model": "bedrock"}
            side_effect += [worker_response]
            tool_call_side_effect += [[WORKER_TOOL_RESPONSE]]

        mock_chat_chain = mocker.patch("redbox.chains.runnables.get_chat_llm")
        mock_chat_chain.side_effect = side_effect

        # mock tool call
        mock_tool_calls = mocker.patch("redbox.graph.nodes.processes.run_tools_parallel")
        mock_tool_calls.side_effect = tool_call_side_effect

        # mock evaluator
        evaluator_response = GenericFakeChatModelWithTools(messages=iter([evaluator]))
        evaluator_response._default_config = {"model": "bedrock"}
        mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=evaluator_response)

        test_case = self.create_new_route_test(
            question="What is AI?", number_of_docs=3, tokens_in_all_docs=30_000, chat_history=old_plan
        )

        await run_app(test_case, agent_plans=configured_multi_agent_plan, user_feedback=test_name)

        assert mock_chat_chain.call_count == len(side_effect)
        assert mock_tool_calls.call_count == len(tool_call_side_effect)

    def test_run_tools_parallel_no_tool_calls(self):
        ai_msg = AIMessage(content="test content")
        tools = []
        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, tools, state)
        assert result == "test content"

    def test_run_tools_parallel_with_valid_tool(self):
        tool = Mock()
        tool.name = "test_tool"
        tool.invoke.return_value = "tool result"

        ai_msg = AIMessage(content="test", additional_kwargs={"tool_calls": [{"name": "test_tool", "args": {}}]})

        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, [tool], state)
        assert len(result) == 4

    def test_run_tools_parallel_tool_not_found(self):
        tool = Mock()
        tool.name = "other_tool"

        ai_msg = AIMessage(content="test", additional_kwargs={"tool_calls": [{"name": "test_tool", "args": {}}]})

        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, [tool], state)
        assert len(result) == 4

    def test_run_tools_parallel_tool_timeout(self):
        tool = Mock()
        tool.name = "test_tool"
        tool.invoke.side_effect = TimeoutError()

        ai_msg = AIMessage(content="test", additional_kwargs={"tool_calls": [{"name": "test_tool", "args": {}}]})

        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, [tool], state, timeout=1)
        assert len(result) == 4

    def test_run_tools_parallel_tool_exception(self):
        tool = Mock()
        tool.name = "test_tool"
        tool.invoke.side_effect = Exception("Tool error")

        ai_msg = AIMessage(content="test", additional_kwargs={"tool_calls": [{"name": "test_tool", "args": {}}]})

        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, [tool], state)
        assert len(result) == 4

    def test_run_tools_parallel_multiple_tools(self):
        tool1 = Mock()
        tool1.name = "tool1"
        tool1.invoke.return_value = "result1"

        tool2 = Mock()
        tool2.name = "tool2"
        tool2.invoke.return_value = "result2"

        ai_msg = AIMessage(
            content="test",
            additional_kwargs={"tool_calls": [{"name": "tool1", "args": {}}, {"name": "tool2", "args": {}}]},
        )

        dummy_query = RedboxQuery(
            question="dummy", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        )
        state = RedboxState(request=dummy_query)

        result = run_tools_parallel(ai_msg, [tool1, tool2], state)
        assert len(result) == 4
        assert result == "test"

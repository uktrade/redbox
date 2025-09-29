from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from redbox import Redbox
from redbox.models.chain import (
    AgentEnum,
    AgentTask,
    ChainChatMessage,
    Citation,
    MultiAgentPlan,
    RedboxQuery,
    RedboxState,
    RequestMetadata,
    Source,
    StructuredResponseWithCitations,
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
        assert (
            len(token_events) > 1
        ), f"Expected tokens as a stream. Received: {token_events}"  # Temporarily turning off streaming check
        assert len(metadata_events) == len(
            test_case.test_data.llm_responses
        ), f"Expected {len(test_case.test_data.llm_responses)} metadata events. Received {len(metadata_events)}"

    assert test_case.test_data.expected_activity_events(
        activity_events
    ), f"Activity events not as expected. Received: {activity_events}"

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

    assert (
        final_state.last_message.content == llm_response
    ), f"Text response from streaming: '{llm_response}' did not match final state text '{final_state.last_message.content}'"
    assert (
        final_state.last_message.content == expected_text
    ), f"Expected text: '{expected_text}' did not match received text '{final_state.last_message.content}'"

    assert (
        final_state.route_name == test_case.test_data.expected_route
    ), f"Expected Route: '{ test_case.test_data.expected_route}'. Received '{final_state.route_name}'"
    if metadata := final_state.metadata:
        assert metadata == metadata_response, f"Expected metadata: '{metadata_response}'. Received '{metadata}'"
    for document_list in document_events:
        for document in document_list:
            assert document in test_case.docs, f"Document not in test case docs: {document}"


async def run_app(
    test_case: RedboxChatTestCase,
    user_feedback: str = "",
    agent_plans: MultiAgentPlan | None = None,
):
    env = Settings()
    # Instantiate app
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever(test_case.docs),
        parameterised_retriever=mock_parameterised_retriever(test_case.docs),
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
                AgentEnum.Web_Search_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with single doc",
                "What is AI",
                [1, 1000],
                True,
                AgentEnum.Web_Search_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "no document one task",
                "What is AI",
                [0, 0],
                True,
                AgentEnum.Legislation_Search_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with single doc",
                "What is AI",
                [1, 1000],
                True,
                AgentEnum.Legislation_Search_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "chat with multiple doc one task",
                "What is AI",
                [3, 10_000],
                True,
                AgentEnum.Internal_Retrieval_Agent,
                ANSWER_WITH_CITATION,
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
                AgentEnum.Web_Search_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "no such keyword with doc",
                "@nosuschkeyword What is AI",
                [3, 10_000],
                True,
                AgentEnum.Internal_Retrieval_Agent,
                ANSWER_WITH_CITATION,
            ),
            (
                "no such keyword no doc",
                "@nosuschkeyword What is AI",
                [0, 0],
                True,
                AgentEnum.Legislation_Search_Agent,
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
        tasks = (
            [AgentTask(task="This is a fake task", agent=agent, expected_output="This is a fake output")]
            if has_task
            else []
        )
        planner = (MultiAgentPlan(tasks=tasks)).model_dump_json()
        planner_response = GenericFakeChatModelWithTools(messages=iter([planner]))
        planner_response._default_config = {"model": "bedrock"}
        if has_task:
            # mock response from worker agent
            worker_response = GenericFakeChatModelWithTools(messages=iter([WORKER_RESPONSE]))
            worker_response._default_config = {"model": "bedrock"}
            mock_chat_chain = mocker.patch("redbox.chains.runnables.get_chat_llm")
            mock_chat_chain.side_effect = [planner_response, worker_response]
            # mock tool call
            mocker.patch("redbox.graph.nodes.processes.run_tools_parallel", return_value=[WORKER_TOOL_RESPONSE])
        else:
            mock_chat_chain = mocker.patch("redbox.chains.runnables.get_chat_llm", return_value=planner_response)

        # mock evaluator
        evaluator_response = GenericFakeChatModelWithTools(messages=iter([evaluator]))
        evaluator_response._default_config = {"model": "bedrock"}
        mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=evaluator_response)
        await run_app(test_case)

        if has_task:
            assert mock_chat_chain.call_count == 2

    @pytest.mark.parametrize(
        "test_name, user_feedback, agents, evaluator",
        [
            (
                "approve plan",
                '{"next": "approve"}',
                [AgentEnum.Internal_Retrieval_Agent, AgentEnum.Web_Search_Agent],
                ANSWER_WITH_CITATION,
            ),
            (
                "modify plan",
                '{"next": "modify"}',
                [AgentEnum.Internal_Retrieval_Agent, AgentEnum.Web_Search_Agent],
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
            tasks += [AgentTask(task="This is a fake task", agent=agents[i], expected_output="This is a fake output")]

        planner = MultiAgentPlan(tasks=tasks)
        old_plan = []
        if "modify" in user_feedback:
            # old plan here
            old_plan = [
                ChainChatMessage(
                    role="ai",
                    text=(MultiAgentPlan(tasks=tasks).model_dump_json()).replace("{", "{{").replace("}", "}}"),
                )
            ]

            planner_response = GenericFakeChatModelWithTools(messages=iter([planner.model_dump_json()]))
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

        await run_app(test_case, agent_plans=planner, user_feedback=test_name)

        assert mock_chat_chain.call_count == len(side_effect)
        assert mock_tool_calls.call_count == len(tool_call_side_effect)

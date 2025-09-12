import copy
import logging

# from enum import Enum
from typing import Any
from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from pytest_mock import MockerFixture

from redbox import Redbox
from redbox.models.chain import (
    AgentTask,
    AISettings,
    Citation,
    MultiAgentPlan,
    RedboxQuery,
    RedboxState,
    RequestMetadata,
    Source,
    StructuredResponseWithCitations,
    metadata_reducer,
)
from redbox.models.chat import ChatRoute, ErrorRoute
from redbox.models.file import ChunkResolution
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
from redbox.transform import structure_documents_by_group_and_indices

# create logger
logger = logging.getLogger("simple_example")
logger.setLevel(logging.INFO)

LANGGRAPH_DEBUG = True


SELF_ROUTE_TO_CHAT = ["Condense self route question", "unanswerable"]
# OUTPUT_WITH_CITATIONS = AIMessage(
# content=StructuredResponseWithCitations(answer="AI is a lie", citations=[]) #.model_dump_json()
# )

OUTPUT_WITH_CITATIONS = AIMessage(
    content=StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json()
)

SELF_ROUTE_TO_SEARCH = ["Condense self route question", OUTPUT_WITH_CITATIONS]


NEW_ROUTE_NO_FEEDBACK = [OUTPUT_WITH_CITATIONS]  # only streaming tokens through evaluator


class MockedAgent:
    def __init__(self, return_value):
        self.return_value = return_value

    def invoke(self, state):
        return self.return_value


def mock_planner_agent(mocker, OUTPUT_WITH_PLANNER):
    mocked_agent = MockedAgent(OUTPUT_WITH_PLANNER)
    mocker.patch("redbox.graph.nodes.processes.create_planner", return_value=mocked_agent)
    return mocked_agent


def mock_worker_agent(mocker, OUTPUT_WITH_WORKER):
    mocked_agent = MockedAgent("")
    mocker.patch("redbox.graph.nodes.processes.create_chain_agent", return_value=mocked_agent)
    mocker.patch("redbox.graph.nodes.processes.run_tools_parallel", return_value=[OUTPUT_WITH_WORKER])


def mock_evaluator_agent(mocker, test_case):
    llm_evaluator = GenericFakeChatModelWithTools(messages=iter(test_case.test_data.llm_responses))
    llm_evaluator._default_config = {"model": "bedrock"}
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm_evaluator)


def assert_number_of_events(num_of_events: int):
    return lambda events_list: len(events_list) == num_of_events


TEST_CASES = [
    test_case
    for generated_cases in [
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=0,
                    tokens_in_all_docs=0,
                    llm_responses=["Testing Response 1"],
                    expected_route=ChatRoute.chat,
                ),
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=100,
                    llm_responses=["Testing Response 1"],
                    expected_route=ChatRoute.chat,
                ),
                RedboxTestData(
                    number_of_docs=10,
                    tokens_in_all_docs=1200,
                    llm_responses=["Testing Response 1"],
                    expected_route=ChatRoute.chat,
                ),
            ],
            test_id="Basic Chat",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
                ai_settings=AISettings(new_route_enabled=True),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=1_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=50_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=80_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
            ],
            test_id="Chat with single doc - new route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
                ai_settings=AISettings(new_route_enabled=False),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=1_000,
                    llm_responses=SELF_ROUTE_TO_SEARCH,
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                    # expected_activity_events=assert_number_of_events(3),
                ),
            ],
            test_id="Chat with single doc - self_route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["s3_key_1", "s3_key_2"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key_1", "s3_key_2"],
                ai_settings=AISettings(new_route_enabled=False),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=40_000,
                    llm_responses=SELF_ROUTE_TO_SEARCH,
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                ),
            ],
            test_id="Chat with multiple docs - self route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["s3_key_1", "s3_key_2"],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings=AISettings(new_route_enabled=True),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=40_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=80_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=140_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=4,
                    tokens_in_all_docs=140_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
            ],
            test_id="Chat with multiple docs - new route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@summarise What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=200_000,
                    llm_responses=["Map Step Response"] * 2 + ["Merge Per Document Response"] + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs_map_reduce,
                ),
            ],
            test_id="Chat with large doc",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
                ai_settings=AISettings(new_route_enabled=False),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=200_000,
                    chunk_resolution=ChunkResolution.normal,
                    llm_responses=SELF_ROUTE_TO_SEARCH,  # + ["Condense Question", "Testing Response 1"],
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                    # expected_activity_events=assert_number_of_events(2),
                ),
            ],
            test_id="Self Route Search large doc",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question=" What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
                ai_settings=AISettings(new_route_enabled=True),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=10,
                    tokens_in_all_docs=2_000_000,
                    llm_responses=["These documents are too large to work with."],
                    expected_route=ErrorRoute.files_too_large,
                    expected_text="These documents are too large to work with.",
                ),
            ],
            test_id="Document too big for system",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@search What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
                ai_settings=AISettings(new_route_enabled=False),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=["Condense response", OUTPUT_WITH_CITATIONS],
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                ),
                RedboxTestData(
                    number_of_docs=5,
                    tokens_in_all_docs=10000,
                    llm_responses=["Condense response", OUTPUT_WITH_CITATIONS],
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                ),
            ],
            test_id="Search keyword - self route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@search What is AI?",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=["Condense response", OUTPUT_WITH_CITATIONS],
                    expected_route=ChatRoute.search,
                    expected_text="AI is a lie",
                    s3_keys=[],
                ),
            ],
            test_id="Search, nothing selected",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@gadget What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=[
                        AIMessage(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "id": "call_e4003b",
                                        "function": {"arguments": '{\n  "query": "ai"\n}', "name": "_search_documents"},
                                        "type": "function",
                                    }
                                ]
                            },
                        ),
                        StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json(),
                    ],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.gadget,
                ),
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=[
                        AIMessage(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "id": "call_e4003b",
                                        "function": {"arguments": '{\n  "query": "ai"\n}', "name": "_search_documents"},
                                        "type": "function",
                                    }
                                ]
                            },
                        ),
                        StructuredResponseWithCitations(
                            answer="AI is a lie, here is some more blurb about why. It's hard to believe but we're mostly making this up",
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
                        ).model_dump_json(),
                    ],
                    expected_text="AI is a lie, here is some more blurb about why. It's hard to believe but we're mostly making this up",
                    expected_citations=[],
                    expected_route=ChatRoute.gadget,
                ),
            ],
            test_id="Agentic search",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@gadget What is AI?",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=[
                        AIMessage(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "id": "call_e4003b",
                                        "function": {"arguments": '{\n  "query": "ai"\n}', "name": "_search_documents"},
                                        "type": "function",
                                    }
                                ]
                            },
                        ),
                        StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json(),
                    ],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.gadget,
                    s3_keys=["s3_key"],
                ),
            ],
            test_id="Agentic search, nothing selected",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@nosuchkeyword What is AI?",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=10,
                    tokens_in_all_docs=1000,
                    llm_responses=["Testing Response 1"],
                    expected_route=ChatRoute.chat,
                ),
            ],
            test_id="No Such Keyword",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@nosuchkeyword What is AI?",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
                ai_settings=AISettings(new_route_enabled=True),
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=50_000,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
                    expected_route=ChatRoute.newroute,
                    expected_text="AI is a lie",
                ),
            ],
            test_id="No Such Keyword with docs - new route ON",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="@gadget Tell me about travel advice to cuba",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=[
                        AIMessage(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "id": "call_e4003b",
                                        "function": {
                                            "arguments": '{\n  "query": "travel advice to cuba"\n}',
                                            "name": "_search_govuk",
                                        },
                                        "type": "function",
                                    }
                                ]
                            },
                        ),
                        StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json(),
                    ],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.gadget,
                ),
            ],
            test_id="Agentic govuk search",
        ),
    ]
    for test_case in generated_cases
]

NEW_ROUTE_internal_retrieval_test_ids = [
    "Chat with single doc - new route ON-0",
    "Chat with single doc - new route ON-1",
    "Chat with single doc - new route ON-2",
    "Chat with multiple docs - new route ON-0",
    "Chat with multiple docs - new route ON-1",
    "Chat with multiple docs - new route ON-2",
    "Chat with multiple docs - new route ON-3",
    "No Such Keyword with docs - new route ON-0",
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("test"), TEST_CASES, ids=[t.test_id for t in TEST_CASES])
async def test_streaming(test: RedboxChatTestCase, env: Settings, mocker: MockerFixture):
    # Current setup modifies test data as it's not a fixture. This is a hack
    test_case = copy.deepcopy(test)
    mocker.patch("redbox.graph.root.lm_choose_route", return_value="search")

    @tool
    def _search_documents(query: str) -> dict[str, Any]:
        """Tool to search documents."""
        return {"documents": structure_documents_by_group_and_indices(test_case.docs)}

    @tool
    def _search_govuk(query: str) -> dict[str, Any]:
        """Tool to search gov.uk for travel advice and other government information."""
        return {"documents": structure_documents_by_group_and_indices(test_case.docs)}

    mocker.patch("redbox.app.build_search_documents_tool", return_value=_search_documents)
    mocker.patch("redbox.app.build_govuk_search_tool", return_value=_search_govuk)

    if test_case.test_id in NEW_ROUTE_internal_retrieval_test_ids:
        OUTPUT_WITH_PLANNER = MultiAgentPlan(
            tasks=[
                AgentTask(
                    task="Task to be completed by the agent",
                    agent="Internal_Retrieval_Agent",
                    expected_output="What this agent should produce",
                )
            ]
        )

        OUTPUT_WITH_WORKER = AIMessage(content="document returned")

        mock_planner_agent(mocker, OUTPUT_WITH_PLANNER)
        mock_worker_agent(mocker, OUTPUT_WITH_WORKER)
        mock_evaluator_agent(mocker, test_case)

    elif test_case.test_id == "Document too big for system-0":
        OUTPUT_WITH_PLANNER = MultiAgentPlan(
            tasks=[
                AgentTask(
                    task="Task to be completed by the agent",
                    agent="Summarisation_Agent",
                    expected_output="What this agent should produce",
                )
            ]
        )

        mock_planner_agent(mocker, OUTPUT_WITH_PLANNER)
        llm = GenericFakeChatModelWithTools(messages=iter(test_case.test_data.llm_responses))
        llm._default_config = {"model": "bedrock"}
        mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)

    else:
        # Mock the LLM and relevant tools
        llm = GenericFakeChatModelWithTools(messages=iter(test_case.test_data.llm_responses))
        llm._default_config = {"model": "bedrock"}
        mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)

    # Instantiate app
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever(test_case.docs),
        parameterised_retriever=mock_parameterised_retriever(test_case.docs),
        metadata_retriever=mock_metadata_retriever(
            [d for d in test_case.docs if d.metadata["uri"] in test_case.query.s3_keys]
        ),
        env=env,
        debug=LANGGRAPH_DEBUG,
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
        input=RedboxState(request=test_case.query),
        response_tokens_callback=streaming_response_handler,
        metadata_tokens_callback=metadata_response_handler,
        route_name_callback=streaming_route_name_handler,
        activity_event_callback=streaming_activity_handler,
        documents_callback=documents_response_handler,
    )

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
        test.test_data.expected_text if test.test_data.expected_text is not None else test.test_data.llm_responses[-1]
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


def test_get_available_keywords(env: Settings):
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever([]),
        parameterised_retriever=mock_parameterised_retriever([]),
        metadata_retriever=mock_metadata_retriever([]),
        env=env,
        debug=LANGGRAPH_DEBUG,
    )
    keywords = {
        ChatRoute.search,
        ChatRoute.newroute,
        ChatRoute.gadget,
        ChatRoute.summarise,
        ChatRoute.tabular,
        ChatRoute.legislation,
    }

    assert keywords == set(app.get_available_keywords().keys())


def test_draw_method(env: Settings, mocker: MockerFixture):
    # Initialize the app with mocked retrievers
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever([]),
        parameterised_retriever=mock_parameterised_retriever([]),
        metadata_retriever=mock_metadata_retriever([]),
        env=env,
    )

    # Create a mock graph instance
    mock_graph_instance = MagicMock()
    mock_graph_instance.draw_mermaid_png.return_value = "mermaid_png_output"

    # Mock the graph retrieval methods
    mocker.patch.object(app.graph, "get_graph", return_value=mock_graph_instance)
    mocker.patch("redbox.graph.root.get_agentic_search_graph", return_value=MagicMock())
    mocker.patch("redbox.graph.root.get_summarise_graph", return_value=MagicMock())

    # Act
    result_root = app.draw(graph_to_draw="root")

    # Assert
    mock_graph_instance.draw_mermaid_png.assert_called_with(
        draw_method=MermaidDrawMethod.PYPPETEER, output_file_path=None
    )
    assert result_root == "mermaid_png_output"

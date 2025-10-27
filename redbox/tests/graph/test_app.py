import copy
import logging

# from enum import Enum
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from pytest_mock import MockerFixture

from redbox import Redbox
from redbox.models.chain import (
    AgentEnum,
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


OUTPUT_WITH_CITATIONS = AIMessage(
    content=StructuredResponseWithCitations(answer="AI is a lie", citations=[]).model_dump_json()
)

NEW_ROUTE_NO_FEEDBACK = [OUTPUT_WITH_CITATIONS]  # only streaming tokens through evaluator
TASK_INTERNAL_AGENT = MultiAgentPlan(
    tasks=[
        AgentTask(
            id="task1",
            task="Task to be completed by the agent",
            agent="Internal_Retrieval_Agent",
            expected_output="What this agent should produce",
        )
    ]
)
TASK_SUMMARISE_AGENT = MultiAgentPlan(
    tasks=[
        AgentTask(
            id="task1",
            task="Task to be completed by the agent",
            agent="Summarisation_Agent",
            expected_output="What this agent should produce",
        ),
        AgentTask(
            id="task2",
            task="Produce final response",
            agent=AgentEnum.Evaluator_Agent,
            expected_output="Final response",
            dependencies=[],
        ),
    ]
)


class MockedAgent:
    def __init__(self, return_value):
        self.return_value = return_value

    def invoke(self, state):
        return self.return_value


def mock_planner_agent(mocker, planner_output):
    mocked_agent = MockedAgent(planner_output)
    mocker.patch("redbox.graph.nodes.processes.create_planner", return_value=mocked_agent)
    return mocked_agent


TEST_CASES = [
    test_case
    for generated_cases in [
        generate_test_cases(
            query=RedboxQuery(
                question="@chat What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=0,
                    tokens_in_all_docs=0,
                    llm_responses=NEW_ROUTE_NO_FEEDBACK,
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
            test_id="Using keyword @Chat",
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
            test_id="Using keyword @summarise",
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
            test_id="Using keyword @search",
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
            test_id="Using keyword @search, nothing selected",
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
                        OUTPUT_WITH_CITATIONS,
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
            test_id="Using keyword @gadget",
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
                        OUTPUT_WITH_CITATIONS,
                    ],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.gadget,
                    s3_keys=["s3_key"],
                ),
            ],
            test_id="Using keyword @gadget, nothing selected",
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
                        OUTPUT_WITH_CITATIONS,
                    ],
                    expected_text="AI is a lie",
                    expected_route=ChatRoute.gadget,
                ),
            ],
            test_id="Using keyword @gadget gov search",
        ),
    ]
    for test_case in generated_cases
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

    if test_case.test_id == "Document too big for system-0":
        mock_planner_agent(mocker, planner_output=TASK_SUMMARISE_AGENT)

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
        ChatRoute.chat,
    }

    assert keywords == set(app.get_available_keywords().keys())


def test_draw_method(env: Settings, mocker: MockerFixture):
    # Initialise the app with mocked retrievers
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


def test_handle_db_file_operations(env: Settings, mocker: MockerFixture):
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever([]),
        parameterised_retriever=mock_parameterised_retriever([]),
        metadata_retriever=mock_metadata_retriever([]),
        env=env,
    )

    mock_remove = mocker.patch("os.remove")
    mock_exists = mocker.patch("os.path.exists", return_value=True)

    app.previous_db_location = "/path/to/old_db.sqlite"
    app.previous_s3_keys = ["key1", "key2"]
    app.handle_db_file(None)

    mock_exists.assert_called_once_with("/path/to/old_db.sqlite")
    mock_remove.assert_called_once_with("/path/to/old_db.sqlite")
    assert app.previous_db_location is None
    assert app.previous_s3_keys is None

    mock_remove.reset_mock()
    mock_exists.reset_mock()

    app.previous_db_location = "/path/to/old_db.sqlite"

    final_state = RedboxState(
        request=RedboxQuery(
            question="What is the meaning of life?",
            s3_keys=[],
            user_uuid=uuid4(),
            chat_history=[],
            permitted_s3_keys=[],
            db_location="/path/to/new_db.sqlite",
        ),
        db_location=None,
    )
    app.handle_db_file(final_state)

    mock_exists.assert_called_once_with("/path/to/old_db.sqlite")
    mock_remove.assert_called_once_with("/path/to/old_db.sqlite")
    assert app.previous_db_location == "/path/to/new_db.sqlite"

    mock_remove.reset_mock()
    mock_exists.reset_mock()

    app.previous_db_location = "/path/to/same_db.sqlite"
    final_state = RedboxState(
        request=RedboxQuery(
            question="Why do dogs bark?",
            s3_keys=["key1"],
            user_uuid=uuid4(),
            chat_history=[],
            permitted_s3_keys=["key1"],
            db_location="/path/to/same_db.sqlite",
        )
    )
    app.handle_db_file(final_state)

    mock_exists.assert_not_called()
    mock_remove.assert_not_called()
    assert app.previous_db_location == "/path/to/same_db.sqlite"
    assert app.previous_s3_keys == ["key1"]


def test_remove_db_file_if_exists_error_handling(env: Settings, mocker: MockerFixture):
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever([]),
        parameterised_retriever=mock_parameterised_retriever([]),
        metadata_retriever=mock_metadata_retriever([]),
        env=env,
    )

    mock_logger = mocker.patch("redbox.app.logger")
    app.remove_db_file_if_exists(None)
    mock_logger.error.assert_not_called()

    mock_exists = mocker.patch("os.path.exists", return_value=False)
    app.remove_db_file_if_exists("/path/to/nonexistent.db")
    mock_exists.assert_called_once_with("/path/to/nonexistent.db")
    mock_logger.error.assert_not_called()

    mock_exists = mocker.patch("os.path.exists", return_value=True)
    app.remove_db_file_if_exists("/path/to/protected.db")
    mock_logger.error.assert_called_once()
    assert "Error encountered when deleting the db file" in mock_logger.error.call_args[0][0]


def test_add_docs_and_db_to_input_state(env: Settings, mocker: MockerFixture):
    app = Redbox(
        all_chunks_retriever=mock_all_chunks_retriever([]),
        parameterised_retriever=mock_parameterised_retriever([]),
        metadata_retriever=mock_metadata_retriever([]),
        env=env,
    )

    app.previous_db_location = "/path/to/db.sqlite"
    app.previous_s3_keys = ["key1", "key2"]

    input_state = RedboxState(
        request=RedboxQuery(
            question="@tabular How many rows in this table?",
            s3_keys=["new_key"],
            user_uuid=uuid4(),
            chat_history=[],
            permitted_s3_keys=["new_key"],
        )
    )

    result = app.add_docs_and_db_to_input_state(input_state)

    assert result.request.previous_s3_keys == ["key1", "key2"]
    assert result.request.db_location == "/path/to/db.sqlite"

    input_state = RedboxState(
        request=RedboxQuery(
            question="What is the purpose of AI long term?",
            s3_keys=["new_key"],
            user_uuid=uuid4(),
            chat_history=[],
        )
    )

    result = app.add_docs_and_db_to_input_state(input_state)

    assert not hasattr(result.request, "previous_s3_keys") or result.request.previous_s3_keys is None
    assert not hasattr(result.request, "db_location") or result.request.db_location is None

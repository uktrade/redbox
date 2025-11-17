import copy
import logging
import os
import sqlite3
from pathlib import Path

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
from redbox.graph.nodes.processes import create_or_update_db_from_tabulars
from redbox.models.chain import (
    AgentTask,
    AISettings,
    Citation,
    DocumentState,
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

TASK_SUMMARISE_AGENT = MultiAgentPlan(
    tasks=[
        AgentTask(
            task="Task to be completed by the agent",
            agent="Summarisation_Agent",
            expected_output="What this agent should produce",
        )
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
        test.test_data.expected_text if test.test_data.expected_text is not None else test.test_data.llm_responses[-1]
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


TABULAR_TEST_CASES = [
    test_case
    for generated_cases in [
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["example.csv"],
                user_uuid="22345678-1234-5678-1234-567812345678",
                chat_history=[],
                permitted_s3_keys=["example.csv"],
                previous_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=["AI is a lie"],
                    expected_route=ChatRoute.newroute,
                )
            ],
            test_id="asking first question to tabular with a new selected file",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["example.csv"],
                user_uuid="22345678-1234-5678-1234-567812345678",
                chat_history=[],
                permitted_s3_keys=["example.csv"],
                previous_s3_keys=["example.csv"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=["AI is a lie"],
                    expected_route=ChatRoute.newroute,
                )
            ],
            test_id="asking follow-up question to tabular with same file selected",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="What is AI?",
                s3_keys=["account.csv"],
                user_uuid="22345678-1234-5678-1234-567812345678",
                chat_history=[],
                permitted_s3_keys=["account.csv", "example.csv"],
                previous_s3_keys=["example.csv"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=1,
                    tokens_in_all_docs=10000,
                    llm_responses=["AI is a lie"],
                    expected_route=ChatRoute.newroute,
                )
            ],
            test_id="de-selecting old file, selecting a new file and asking another question",
        ),
    ]
    for test_case in generated_cases
]

TASK_TABULAR_AGENT = MultiAgentPlan(
    tasks=[
        AgentTask(
            task="Task to be completed by the agent",
            agent="Tabular_Agent",
            expected_output="What this agent should produce",
        )
    ]
)


@pytest.mark.parametrize(("test"), TABULAR_TEST_CASES, ids=[t.test_id for t in TABULAR_TEST_CASES])
@pytest.mark.parametrize("simulate_interrupt", [False, True])
def test_tabular_file_handling(test, tmp_path: Path, mocker: MockerFixture, simulate_interrupt: bool):
    """
    This unit test is testing the database handling inside the tabular schema retrieval. It invokes the relevant part of the graph.
    - Test case 1: File selected and no previous files selected: check that the database is created
    - Test case 2: Same file still selected (same as test case 1), asking a follow-up question: check that the same database still exist, and was not deleted
    - Test case 3: Previous File de-selected, a new file is selected: check that the existing database is deleted and a new database is created
    """
    test_case = copy.deepcopy(test)

    request: RedboxQuery = test_case.query
    request.previous_s3_keys = []
    request.db_location = None

    if test_case.test_id.startswith("asking follow-up question to tabular with same file selected"):
        request.previous_s3_keys = sorted(request.s3_keys)
    elif test_case.test_id.startswith("de-selecting old file, selecting a new file and asking another question"):
        request.previous_s3_keys = ["old_file.csv"]

    group_uuid = str(uuid4())
    doc_uuids = {str(uuid4()): doc for doc in test_case.docs}
    mock_documents = DocumentState(groups={group_uuid: doc_uuids})

    state = RedboxState(
        request=request,
        documents=mock_documents,
    )

    spy_remove_call = mocker.spy(os, "remove")

    if simulate_interrupt:
        original_func = create_or_update_db_from_tabulars

        def interrupting_func(state_arg):
            original_func(state_arg)
            msg = "Simulated interruption"
            raise RuntimeError(msg)

        mocker.patch(
            "redbox.graph.nodes.processes.create_or_update_db_from_tabulars",
            side_effect=interrupting_func,
        )

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        prior_db_exists = False
        if test_case.test_id.startswith("asking follow-up") or test_case.test_id.startswith("de-selecting"):
            prior_request = copy.deepcopy(request)
            prior_request.s3_keys = sorted(request.previous_s3_keys)
            prior_request.previous_s3_keys = []
            prior_request.db_location = None

            if test_case.test_id.startswith("asking follow-up"):
                prior_doc_uuids = doc_uuids
                prior_mock_documents = mock_documents
            else:
                old_doc = Document(page_content="col1,col2\nval1,val2", metadata={"uri": "old_file.csv"})
                prior_doc_uuids = {str(uuid4()): old_doc}
                prior_mock_documents = DocumentState(groups={group_uuid: prior_doc_uuids})

            prior_state = RedboxState(
                request=prior_request,
                documents=prior_mock_documents,
            )
            _ = create_or_update_db_from_tabulars(prior_state)
            prior_db_exists = True

        initial_changed = state.documents_changed()

        db_created = False
        try:
            create_or_update_db_from_tabulars(state)
            db_created = True
        except RuntimeError as e:
            if simulate_interrupt:
                assert str(e) == "Simulated interruption"
                db_path = state.request.db_location
                if db_path:
                    db_created = os.path.exists(db_path)

        # check if database file exists
        if db_created:
            db_path = state.request.db_location
            assert os.path.exists(db_path)

            # check that the database path follow expected format
            assert db_path == f"generated_db_{request.user_uuid}.db"

            # Additional checks
            with sqlite3.connect(db_path) as conn:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                    ).fetchall()
                ]
            if not simulate_interrupt:
                expected_table_count = len(test_case.docs)
                assert len(tables) == expected_table_count
                sample_key = request.s3_keys[0].split("/")[-1].split(".")[0]
                assert any(sample_key in table for table in tables)

        if test_case.test_id.startswith("asking follow-up question to tabular with same file selected"):
            assert prior_db_exists
            # check if database file was not deleted before creation
            spy_remove_call.assert_not_called()
            if not simulate_interrupt:
                assert not initial_changed

        elif test_case.test_id.startswith("de-selecting old file, selecting a new file and asking another question"):
            assert prior_db_exists
            # check if database file was not deleted before creation
            spy_remove_call.assert_called_once_with(state.request.db_location)
            if not simulate_interrupt:
                assert initial_changed
                assert all("old_file" not in table for table in tables)

        else:
            spy_remove_call.assert_not_called()
            if not simulate_interrupt:
                assert initial_changed

        # State updates
        if not simulate_interrupt:
            assert sorted(request.s3_keys) == sorted(state.request.previous_s3_keys)

    finally:
        os.chdir(original_cwd)

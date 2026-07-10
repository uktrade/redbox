from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from redbox import Redbox
from redbox.models.chain import (
    Citation,
    RedboxQuery,
    RedboxState,
    Source,
    StructuredResponseWithCitations,
    configure_agent_task_plan,
)
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.test.data import (
    GenericFakeChatModelWithTools,
    RedboxTestData,
    generate_test_cases,
    mock_all_chunks_retriever,
    mock_metadata_retriever,
    mock_parameterised_retriever,
)

EVALUATOR_ANSWER = AIMessage(
    content=StructuredResponseWithCitations(answer="Here is the answer.", citations=[]).model_dump_json()
)

EVALUATOR_WITH_CITATION = AIMessage(
    content=StructuredResponseWithCitations(
        answer="Here is the answer along with citations.",
        citations=[
            Citation(
                text_in_answer="Here is the answer.",
                sources=[
                    Source(
                        source="TestSource",
                        document_name="test-doc.pdf",
                        highlighted_text_in_source="supporting evidence",
                        page_numbers=[1],
                    )
                ],
            )
        ],
    ).model_dump_json()
)

# worker response where there  is a tool call
WORKER_TOOL_CALL = AIMessage(
    content="Searching for information.",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_abc123",
                "function": {"arguments": {"query": "test query"}, "name": "search_tool"},
                "type": "function",
            }
        ]
    },
)

# worker response with no tool call
WORKER_DIRECT_RESPONSE = AIMessage(content="Here is some information.")

WORKER_TOOL_RESULT = AIMessage(content="Here is the result from the requested tool.")

TABULAR_TOOL_RESULT = AIMessage(content=["Tabular analysis result", "pass", "False"])


def _make_plan(agent_name: str) -> str:
    agent_task, multi_agent_plan = configure_agent_task_plan({agent_name: agent_name})
    plan = multi_agent_plan().model_copy(update={"tasks": [agent_task()]})
    return plan.model_dump.json()


def _fake_llm(response) -> GenericFakeChatModelWithTools:
    llm = GenericFakeChatModelWithTools(messages=iter([response]))
    llm._default_config = {"model": "bedrock"}
    return llm


def _make_test_case(question: str, number_of_docs: int = 0, tokens: int = 0):
    return generate_test_cases(
        query=RedboxQuery(
            question=question,
            s3_keys=[],
            user_uuid=uuid4(),
            chat_history=[],
            permitted_s3_keys=[],
        ),
        test_data=[
            RedboxTestData(
                number_of_docs=number_of_docs,
                tokens_in_all_docs=tokens,
                llm_responses=["Here is the answer."],
                expected_route=ChatRoute.newroute,
            )
        ],
        test_id="agent-integraqtion-test",
    )[0]


def _make_app(test_case) -> Redbox:
    return Redbox(
        all_chunks_retriever=mock_all_chunks_retriever(test_case.docs),
        parameterised_retriever=mock_parameterised_retriever(test_case.docs),
        metadata_retriever=mock_metadata_retriever(test_case.docs),
        env=Settings(),
        debug=True,
    )


async def _run(test_case, app) -> tuple[list[str], RedboxState]:
    token_events = []

    async def collect_tokens(t: str):
        token_events.append(t)

    final_state = await app.run(
        input=RedboxState(request=test_case.query),
        response_tokens_callback=collect_tokens,
    )
    return token_events, final_state


def _assert_non_blank_response(token_events: list[str], final_state: RedboxState, agent_name: str):
    assert "".join(token_events), f"{agent_name} produced a blank response"
    assert final_state.last_message.content, f"{agent_name} produced a blank final state"


# SUMMARISATION AGENT
@pytest.mark.asyncio
async def test_summarisation_agent_returns_response(mocker):
    """
    Integration test to check Summarisation_Agent produces a non-blank reponse when asked to summarise a file that has been input
    """
    test_case = _make_test_case("Summarise this given document.", number_of_docs=1, tokens=5000)

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("Summarisation_Agent")),
            _fake_llm(WORKER_DIRECT_RESPONSE),
        ],
    )

    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_WITH_CITATION))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "Summarisation_Agent")


# INTERNAL RETRIEVAL AGENT
@pytest.mark.asyncio
async def test_internal_retrieval_agent_returns_response(mocker):
    """
    Integration test to check Internal_Retrieval_Agent produces a non-blank reponse when asked to find information from within internal documents
    """
    test_case = _make_test_case("Find information on this in my documents", number_of_docs=3, tokens=10000)

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("Internal_Retrieval_Agent")),
            _fake_llm(WORKER_TOOL_CALL),
        ],
    )

    mocker.patch("redbox.graph.agents.workers.run_tools_parallel", return_value=_fake_llm(WORKER_TOOL_RESULT))
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_WITH_CITATION))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "Internal_Retrieval_Agent")


# INTERNAL RETRIEVAL AGENT
@pytest.mark.asyncio
async def test_external_retrieval_agent_returns_response(mocker):
    """
    Integration test to check External_Retrieval_Agent produces a non-blank reponse when asked to find extermal information associated with a question
    """
    test_case = _make_test_case("Find external information about this topic")

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("External_Retrieval_Agent")),
            _fake_llm(WORKER_TOOL_CALL),
        ],
    )

    mocker.patch("redbox.graph.agents.workers.run_tools_parallel", return_value=_fake_llm(WORKER_TOOL_RESULT))
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_WITH_CITATION))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "External_Retrieval_Agent")


# WEB SEARCH AGENT
@pytest.mark.asyncio
async def test_web_search_agent_returns_response(mocker):
    """
    Integration test to check Web_Search_Agent produces a non-blank reponse when asked to search the web for some information
    """
    test_case = _make_test_case("Search the web for this information.")

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("Web_Search_Agent")),
            _fake_llm(WORKER_TOOL_CALL),
        ],
    )

    mocker.patch("redbox.graph.agents.workers.run_tools_parallel", return_value=_fake_llm(WORKER_TOOL_RESULT))
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_WITH_CITATION))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "Web_Search_Agent")


# TABULAR AGENT
@pytest.mark.asyncio
async def test_tabular_agent_returns_response(mocker):
    """
    Integration test to check Tabular_Agent produces a non-blank reponse when asked to analyse a file which is in a tabular format
    """
    test_case = _make_test_case("Tell me about information in this spreadsheet.", number_of_docs=1, tokens=5000)

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("Tabular_Agent")),
            _fake_llm(WORKER_TOOL_CALL),
        ],
    )

    mocker.patch("redbox.graph.agents.workers.run_tools_parallel", return_value=_fake_llm(TABULAR_TOOL_RESULT))
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_ANSWER))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "Tabular_Agent")


# DATAHUB AGENT
@pytest.mark.asyncio
async def test_datahub_agent_returns_response(mocker):
    """
    Integration test to check Datahub_Agent produces a non-blank reponse when asked to retrieve data from datahub.
    As Datahub Agent uses an MCP server, the call to this is mocked here
    """
    test_case = _make_test_case("Retieve information about DBT from datahub.")

    mocker.patch(
        "redbox.chains.runnables.get_chat_llm",
        side_effect=[
            _fake_llm(_make_plan("Datahub_Agent")),
            _fake_llm(WORKER_TOOL_CALL),
        ],
    )

    mocker.patch(
        "redbox.graph.nodes.runner.wrap_async.get_datahub_mcp_tools",
        new_callable=AsyncMock,
        return_value=[],
    )

    mocker.patch("redbox.graph.agents.workers.run_tools_parallel", return_value=_fake_llm(WORKER_TOOL_RESULT))
    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=_fake_llm(EVALUATOR_WITH_CITATION))

    token_events, final_state = await _run(test_case, _make_app(test_case))
    _assert_non_blank_response(token_events, final_state, "Datahub_Agent")

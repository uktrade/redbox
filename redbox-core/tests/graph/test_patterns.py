from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from pytest_mock import MockerFixture

from redbox.chains.components import get_structured_response_with_citations_parser
from redbox.chains.parser import ClaudeParser
from redbox.chains.runnables import CannedChatLLM, build_chat_prompt_from_messages_runnable, build_llm_chain
from redbox.graph.nodes.processes import (
    build_chat_pattern,
    build_merge_pattern,
    build_passthrough_pattern,
    build_retrieve_pattern,
    build_set_route_pattern,
    build_set_text_pattern,
    build_stuff_pattern,
    clear_documents_process,
    empty_process,
    lm_choose_route,
)
from redbox.models.chain import (
    AgentDecision,
    AISettings,
    Citation,
    DocumentState,
    PromptSet,
    RedboxQuery,
    RedboxState,
    Source,
)
from redbox.models.chat import ChatRoute
from redbox.test.data import (
    RedboxChatTestCase,
    RedboxTestData,
    generate_docs,
    generate_test_cases,
    mock_all_chunks_retriever,
    mock_basic_metadata_retriever,
    mock_parameterised_retriever,
)
from redbox.transform import flatten_document_state, structure_documents_by_file_name

LANGGRAPH_DEBUG = True

CHAT_PROMPT_TEST_CASES = generate_test_cases(
    query=RedboxQuery(question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]),
    test_data=[
        RedboxTestData(
            number_of_docs=0,
            tokens_in_all_docs=0,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat,
        ),
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Chat prompt runnable",
)


@pytest.mark.parametrize(("test_case"), CHAT_PROMPT_TEST_CASES, ids=[t.test_id for t in CHAT_PROMPT_TEST_CASES])
def test_build_chat_prompt_from_messages_runnable(test_case: RedboxChatTestCase, tokeniser: callable):
    """Tests a given state can be turned into a chat prompt."""
    chat_prompt = build_chat_prompt_from_messages_runnable(prompt_set=PromptSet.Chat, tokeniser=tokeniser)
    state = RedboxState(request=test_case.query)

    response = chat_prompt.invoke(state)
    messages = response.to_messages()
    assert isinstance(messages, list)
    assert len(messages) > 0


BUILD_LLM_TEST_CASES = generate_test_cases(
    query=RedboxQuery(question="What is AI?", file_uuids=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=[AIMessage(content="Testing Response 1")],
            expected_route=ChatRoute.chat_with_docs,
        ),
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=[
                AIMessage(
                    content="Tool Response 1",
                    tool_calls=[ToolCall(name="foo", args={"query": "bar"}, id="tool_1")],
                )
            ],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Build LLM runnable",
)


@pytest.mark.parametrize(("test_case"), BUILD_LLM_TEST_CASES, ids=[t.test_id for t in BUILD_LLM_TEST_CASES])
def test_build_llm_chain(test_case: RedboxChatTestCase):
    """Tests a given state can update the data and metadata correctly."""
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}
    llm_chain = build_llm_chain(PromptSet.Chat, llm)
    state = RedboxState(request=test_case.query)

    final_state = llm_chain.invoke(state)

    test_case_content = test_case.test_data.llm_responses[-1].content
    test_case_tool_calls = test_case.test_data.llm_responses[-1].tool_calls

    assert (
        final_state["messages"][-1].content == test_case_content
    ), f"Expected LLM response: '{test_case_content}'. Received '{final_state["messages"][-1].content}'"
    assert final_state["messages"][-1].tool_calls == test_case_tool_calls
    assert sum(final_state["metadata"].input_tokens.values())
    assert sum(final_state["metadata"].output_tokens.values())


CHAT_TEST_CASES = generate_test_cases(
    query=RedboxQuery(question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]),
    test_data=[
        RedboxTestData(
            number_of_docs=0,
            tokens_in_all_docs=0,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat,
        )
    ],
    test_id="Chat pattern",
)


@pytest.mark.parametrize(("test_case"), CHAT_TEST_CASES, ids=[t.test_id for t in CHAT_TEST_CASES])
def test_build_chat_pattern(test_case: RedboxChatTestCase, mocker: MockerFixture):
    """Tests a given state["request"] correctly changes state["text"]."""
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}
    state = RedboxState(request=test_case.query)

    chat = build_chat_pattern(prompt_set=PromptSet.Chat, final_response_chain=True)

    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)
    final_state = chat(state)

    test_case_content = test_case.test_data.llm_responses[-1].content

    assert (
        final_state["messages"][-1].content == test_case_content
    ), f"Expected LLM response: '{test_case_content}'. Received '{final_state['messages'][-1].content}'"


SET_ROUTE_TEST_CASES = generate_test_cases(
    query=RedboxQuery(question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]),
    test_data=[
        RedboxTestData(
            number_of_docs=0,
            tokens_in_all_docs=0,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat,
        ),
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Set route pattern",
)


@pytest.mark.parametrize(("test_case"), SET_ROUTE_TEST_CASES, ids=[t.test_id for t in SET_ROUTE_TEST_CASES])
def test_build_set_route_pattern(test_case: RedboxChatTestCase):
    """Tests a given value correctly changes state["route"]."""
    set_route = build_set_route_pattern(route=test_case.test_data.expected_route)
    state = RedboxState(request=test_case.query)

    response = set_route.invoke(state)
    final_state = RedboxState(**response, request=test_case.query)

    assert (
        final_state.route_name == test_case.test_data.expected_route.value
    ), f"Expected Route: '{ test_case.test_data.expected_route.value}'. Received '{final_state.route_name}'"


RETRIEVER_TEST_CASES = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1", "s3_key_2"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1", "s3_key_2"],
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=80_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
        RedboxTestData(
            number_of_docs=4,
            tokens_in_all_docs=140_000,
            llm_responses=["Map Step Response"] * 4 + ["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Retriever pattern",
)


@pytest.mark.parametrize(
    ("test_case", "mock_retriever"),
    [(test_case, mock_all_chunks_retriever) for test_case in RETRIEVER_TEST_CASES]
    + [(test_case, mock_parameterised_retriever) for test_case in RETRIEVER_TEST_CASES],
    ids=[f"All chunks, {t.test_id}" for t in RETRIEVER_TEST_CASES]
    + [f"Parameterised, {t.test_id}" for t in RETRIEVER_TEST_CASES],
)
def test_build_retrieve_pattern(test_case: RedboxChatTestCase, mock_retriever: BaseRetriever):
    """Tests a given state["request"] correctly changes state["documents"]."""
    retriever = mock_retriever(test_case.docs)
    retriever_function = build_retrieve_pattern(retriever=retriever, structure_func=structure_documents_by_file_name)
    state = RedboxState(request=test_case.query)

    response = retriever_function.invoke(state)
    final_state = RedboxState(**response, request=test_case.query)

    assert final_state.documents == structure_documents_by_file_name(test_case.docs)


MERGE_TEST_CASES = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1", "s3_key_2"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1", "s3_key_2"],
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
        RedboxTestData(
            number_of_docs=4,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 2"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Merge pattern",
)


@pytest.mark.parametrize(("test_case"), MERGE_TEST_CASES, ids=[t.test_id for t in MERGE_TEST_CASES])
def test_build_merge_pattern(test_case: RedboxChatTestCase, mocker: MockerFixture):
    """Tests a given state["request"] and state["documents"] correctly changes state["documents"]."""
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}
    state = RedboxState(request=test_case.query, documents=structure_documents_by_file_name(test_case.docs))

    merge = build_merge_pattern(prompt_set=PromptSet.ChatwithDocsMapReduce, final_response_chain=True)

    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)
    response = merge.invoke(state)
    final_state = RedboxState(**response, request=test_case.query)

    response_documents = [doc for doc in flatten_document_state(final_state.documents) if doc is not None]
    noned_documents = sum(1 for doc in final_state.documents.groups.values() for v in doc.values() if v is None)

    test_case_content = test_case.test_data.llm_responses[-1].content

    assert len(response_documents) == 1
    assert noned_documents == len(test_case.docs) - 1
    assert (
        response_documents[0].page_content == test_case_content
    ), f"Expected document content: '{test_case_content}'. Received '{response_documents[0].page_content}'"


STUFF_TEST_CASES = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1", "s3_key_2"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1", "s3_key_2"],
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
        RedboxTestData(
            number_of_docs=4,
            tokens_in_all_docs=40_000,
            llm_responses=["Testing Response 2"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Stuff pattern",
)


@pytest.mark.parametrize(("test_case"), STUFF_TEST_CASES, ids=[t.test_id for t in STUFF_TEST_CASES])
def test_build_stuff_pattern(test_case: RedboxChatTestCase, mocker: MockerFixture):
    """Tests a given state["request"] and state["documents"] correctly changes state["text"]."""
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}
    state = RedboxState(request=test_case.query, documents=structure_documents_by_file_name(test_case.docs))

    stuff = build_stuff_pattern(prompt_set=PromptSet.ChatwithDocs, final_response_chain=True)

    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)
    response = stuff.invoke(state)
    final_state = RedboxState(**response, request=test_case.query)

    test_case_content = test_case.test_data.llm_responses[-1].content

    assert (
        final_state.last_message.content == test_case_content
    ), f"Expected LLM response: '{test_case_content}'. Received '{final_state.last_message.content}'"


TOOL_TEST_CASES = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1", "s3_key_2"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1", "s3_key_2"],
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=2_000,
            llm_responses=["Testing Response 1"],
            expected_route=ChatRoute.chat_with_docs,
        ),
    ],
    test_id="Stuff pattern",
)


def test_build_passthrough_pattern():
    """Tests a given state["request"] correctly changes state["text"]."""
    passthrough = build_passthrough_pattern()
    state = RedboxState(
        request=RedboxQuery(
            question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        ),
    )

    response = passthrough.invoke(state)
    final_state = RedboxState(**response, request=state.request)

    assert final_state.last_message.content == "What is AI?"


def test_build_set_text_pattern():
    """Tests a given value correctly changes the state["text"]."""
    set_text = build_set_text_pattern(text="An hendy hap ychabbe ychent.")
    state = RedboxState(
        request=RedboxQuery(
            question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        ),
    )

    response = set_text.invoke(state)
    final_state = RedboxState(**response, request=state.request)

    assert final_state.last_message.content == "An hendy hap ychabbe ychent."


def test_empty_process():
    """Tests the empty process doesn't touch the state whatsoever."""
    state = RedboxState(
        request=RedboxQuery(
            question="What is AI?", s3_keys=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        ),
        documents=structure_documents_by_file_name([doc for doc in generate_docs(s3_key="s3_key")]),
        messages=[HumanMessage(content="Foo")],
        route_name=ChatRoute.chat_with_docs_map_reduce,
    )

    builder = StateGraph(RedboxState)
    builder.add_node("null", empty_process)
    builder.add_edge(START, "null")
    builder.add_edge("null", END)
    graph = builder.compile()

    response = graph.invoke(state)
    final_state = RedboxState(**response)

    assert final_state == state


CLEAR_DOC_TEST_CASES = [
    RedboxState(
        request=RedboxQuery(
            question="What is AI?", file_uuids=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        ),
        documents=structure_documents_by_file_name([doc for doc in generate_docs(s3_key="s3_key")]),
        messages=[HumanMessage(content="Foo")],
        route_name=ChatRoute.chat_with_docs_map_reduce,
    ),
    RedboxState(
        request=RedboxQuery(
            question="What is AI?", file_uuids=[], user_uuid=uuid4(), chat_history=[], permitted_s3_keys=[]
        ),
        documents={},
        messages=[HumanMessage(content="Foo")],
        route_name=ChatRoute.chat_with_docs_map_reduce,
    ),
]


@pytest.mark.parametrize(("test_case"), CLEAR_DOC_TEST_CASES)
def test_clear_documents(test_case: list[RedboxState]):
    """Tests that clear documents does what it says."""
    builder = StateGraph(RedboxState)
    builder.add_node("clear", clear_documents_process)
    builder.add_edge(START, "clear")
    builder.add_edge("clear", END)
    graph = builder.compile()

    response = graph.invoke(test_case)
    final_state = RedboxState(**response)

    assert final_state.documents == DocumentState(groups={})


def test_canned_llm():
    """Tests that the CannedLLM works in a normal call."""
    text = "Lorem ipsum dolor sit amet."
    canned = CannedChatLLM(messages=[AIMessage(content=text)])
    response = canned.invoke("Foo")
    assert text == response.content


@pytest.mark.asyncio
async def test_canned_llm_async():
    """Tests that the CannedLLM works asynchronously."""
    text = "Lorem ipsum dolor sit amet."
    canned = CannedChatLLM(messages=[AIMessage(content=text)])

    events: list[dict] = []
    async for e in canned.astream_events("Foo", version="v2"):
        events.append(e)

    response = "".join([d["data"]["chunk"].content for d in events if d.get("event") == "on_chat_model_stream"])

    assert text == response


LLM_ROUTE_TEST_CASE = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1"],
        ai_settings=AISettings(
            self_route_enabled=True,
        ),
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=2,
            tokens_in_all_docs=40_000,
            llm_responses=['{\n  "next": "search"\n}'],
            expected_route=ChatRoute.search,
        ),
    ],
    test_id="LLM choose route",
)


@pytest.mark.parametrize(
    "test_case, basic_metadata",
    [
        (LLM_ROUTE_TEST_CASE[0], mock_basic_metadata_retriever),
    ],
    ids=[t.test_id for t in LLM_ROUTE_TEST_CASE],
)
def test_llm_choose_route(test_case: RedboxChatTestCase, basic_metadata: BaseRetriever, mocker: MockerFixture):
    mocker.patch("redbox.chains.runnables.get_basic_metadata_retriever", return_value=basic_metadata(test_case.docs))
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}

    mocker.patch("redbox.chains.runnables.get_chat_llm", return_value=llm)
    agent_parser = ClaudeParser(pydantic_object=AgentDecision)
    state = RedboxState(request=test_case.query, documents=structure_documents_by_file_name(test_case.docs))

    llm_choose_search = lm_choose_route(state, parser=agent_parser)

    assert llm_choose_search == "search"


STRUCTURED_OUTPUT_TEST_CASE = generate_test_cases(
    query=RedboxQuery(
        question="What is AI?",
        s3_keys=["s3_key_1"],
        user_uuid=uuid4(),
        chat_history=[],
        permitted_s3_keys=["s3_key_1"],
        ai_settings=AISettings(
            self_route_enabled=True,
        ),
    ),
    test_data=[
        RedboxTestData(
            number_of_docs=1,
            tokens_in_all_docs=20_000,
            llm_responses=[
                '{\n  "answer": "blah ref_1 foo ref_2",\n  "citations": [\n    {\n      "text_in_answer": "blah",\n      "sources": [\n        {\n          "source": "https://www.gov.uk/test/",\n          "source_type": "",\n          "document_name": "",\n          "highlighted_text_in_source": "blah",\n          "page_numbers": None,\n          "ref_id": "ref_1"\n        }\n      ]\n    },\n    {\n      "text_in_answer": "foo",\n      "sources": [\n        {\n          "source": "https://www.gov.uk/test",\n          "source_type": "GOV.UK", \n          "document_name": "test",\n          "highlighted_text_in_source": "foo foo",\n          "page_numbers": ["i", 2],\n          "ref_id": "ref_2"\n        }\n      ]\n    }  ]\n    }'
            ],
            expected_route=ChatRoute.search,
        ),
    ],
    test_id="Structured output",
)


@pytest.mark.parametrize(
    ("test_case"), STRUCTURED_OUTPUT_TEST_CASE, ids=[t.test_id for t in STRUCTURED_OUTPUT_TEST_CASE]
)
def test_citation_structured_output(test_case: RedboxChatTestCase, mocker: MockerFixture):
    llm = GenericFakeChatModel(messages=iter(test_case.test_data.llm_responses))
    llm._default_config = {"model": "bedrock"}

    mocker.patch("redbox.graph.nodes.processes.get_chat_llm", return_value=llm)
    state = RedboxState(request=test_case.query, documents=structure_documents_by_file_name(test_case.docs))

    citations_output_parser, format_instructions = get_structured_response_with_citations_parser()
    stuff = build_stuff_pattern(
        prompt_set=PromptSet.Search,
        output_parser=citations_output_parser,
        format_instructions=format_instructions,
        final_response_chain=True,
    )
    response = stuff.invoke(state)

    final_state = RedboxState(**response, request=state.request)
    assert final_state.citations == [
        Citation(
            text_in_answer="blah",
            sources=[
                Source(
                    source="https://www.gov.uk/test/",
                    source_type="Unknown",
                    document_name="Unknown",
                    highlighted_text_in_source="blah",
                    page_numbers=[1],
                    ref_id="ref_1",
                )
            ],
        ),
        Citation(
            text_in_answer="foo",
            sources=[
                Source(
                    source="https://www.gov.uk/test",
                    source_type="GOV.UK",
                    document_name="test",
                    highlighted_text_in_source="foo foo",
                    page_numbers=[1, 2],
                    ref_id="ref_2",
                )
            ],
        ),
    ]

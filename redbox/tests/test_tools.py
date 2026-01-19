import re
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import requests_mock
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from opensearchpy import OpenSearch
from pytest_mock import MockerFixture
from requests import Response

from redbox.api.format import reduce_chunks_by_tokens
from redbox.graph.nodes.tools import (
    brave_response_to_documents,
    build_document_from_prompt_tool,
    build_govuk_search_tool,
    build_legislation_search_tool,
    build_retrieve_document_full_text,
    build_retrieve_knowledge_base,
    build_search_documents_tool,
    build_search_wikipedia_tool,
    build_web_search_tool,
    format_result,
    kagi_response_to_documents,
    web_search_call,
    web_search_with_retry,
)
from redbox.models.chain import AISettings, RedboxQuery, RedboxState
from redbox.models.file import ChunkCreatorType, ChunkMetadata, ChunkResolution
from redbox.models.settings import Settings
from redbox.test.data import RedboxChatTestCase
from redbox.transform import bedrock_tokeniser, combine_documents, flatten_document_state
from tests.retriever.test_retriever import TEST_CHAIN_PARAMETERS


@pytest.mark.parametrize(
    "loop, content, artifact, status, is_intermediate_step",
    [(True, "test", "test", "pass", True), (False, "test", "test", None, None)],
)
def test_format_result(loop, content, artifact, status, is_intermediate_step):
    formatted_result = format_result(loop, content, artifact, status, is_intermediate_step)

    assert type(formatted_result) is tuple
    if loop:
        assert type(formatted_result[0]) is tuple
        assert len(formatted_result[0]) == 3
    else:
        assert type(formatted_result[0]) is str


def test_document_from_prompt_tool():
    doc_to_prompt_tool = build_document_from_prompt_tool()
    tool_node = ToolNode(tools=[doc_to_prompt_tool])
    result_state = tool_node.invoke(
        RedboxState(
            request=RedboxQuery(
                question="Can you test this doc: some texts",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings=AISettings(),
                permitted_s3_keys=[],
            ),
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_retrieve_document_from_prompt",
                            "args": {},
                            "id": "1",
                        }
                    ],
                )
            ],
        )
    )

    assert (
        result_state["messages"][0].content
        == "<context>This is user prompt that containing documents.</context>Can you test this doc: some texts"
    )


def test_retrieve_knowledge_base(es_client: OpenSearch, es_index: str, stored_file_knowledge_base: RedboxChatTestCase):
    kb_tool = build_retrieve_knowledge_base(es_client, es_index)
    tool_node = ToolNode(tools=[kb_tool])
    result_state = tool_node.invoke(
        RedboxState(
            request=stored_file_knowledge_base.query,
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_retrieve_knowledge_base",
                            "args": {},
                            "id": "1",
                        }
                    ],
                )
            ],
        )
    )
    if stored_file_knowledge_base.test_id == "Successful Path-0":
        assert "<context>This is your knowledgebase result.</context>" in result_state["messages"][0].content
    elif stored_file_knowledge_base.test_id == "Empty knowledge base-0":
        assert result_state["messages"][0].content == "Tool returns empty result set."


def test_retrieve_document_full_text_tool(
    es_client: OpenSearch, es_index: str, stored_file_all_chunks: RedboxChatTestCase
):
    """
    Test that the tool is able to return a document's full text given file name
    """
    # build the tool
    ft_tool = build_retrieve_document_full_text(es_client, es_index)

    tool_node = ToolNode(tools=[ft_tool])
    result_state = tool_node.invoke(
        RedboxState(
            request=stored_file_all_chunks.query,
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_retrieve_document_full_text",
                            "args": {},
                            "id": "1",
                        }
                    ],
                )
            ],
        )
    )

    if stored_file_all_chunks.test_id == "Successful Path":
        assert len(result_state["messages"][0].content) > 0
    elif stored_file_all_chunks.test_id == "No permitted S3 keys":
        assert len(result_state["messages"][0].content) == 0
    elif stored_file_all_chunks.test_id == "Empty keys but permitted":
        assert len(result_state["messages"][0].content) > 0


@pytest.mark.parametrize("chain_params", TEST_CHAIN_PARAMETERS)
def test_knowledge_base_search_documents_tool(
    chain_params: dict,
    es_client: OpenSearch,
    es_index: str,
    embedding_model: FakeEmbeddings,
    env: Settings,
    stored_file_knowledge_base: RedboxChatTestCase,
):
    # Build and run
    kb_tool = build_search_documents_tool(
        es_client=es_client,
        index_name=es_index,
        embedding_model=embedding_model,
        embedding_field_name=env.embedding_document_field_name,
        chunk_resolution=ChunkResolution.normal,
        repository="knowledge_base",
    )

    tool_node = ToolNode(tools=[kb_tool])
    result_state = tool_node.invoke(
        RedboxState(
            request=stored_file_knowledge_base.query,
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_search_knowledge_base",
                            "args": {"query": "example"},
                            "id": "1",
                        }
                    ],
                )
            ],
        )
    )

    print(result_state["messages"][0])


@pytest.mark.parametrize("chain_params", TEST_CHAIN_PARAMETERS)
def test_search_documents_tool(
    chain_params: dict,
    stored_file_parameterised: RedboxChatTestCase,
    es_client: OpenSearch,
    es_index: str,
    embedding_model: FakeEmbeddings,
    env: Settings,
):
    """
    Tests the search documents tool.

    As this is a slight reworking of the parameterised retriever to
    work more as a tool, we partly just adapt the same unit test.

    Part of the rework is to emit a state, so some of our tests echo
    the structure_documents_* unit tests, which turn document
    lists into a DocumentState.

    Asserts:

    * If documents are selected and there's permission to get them
        * The length of the result is equal to the rag_k parameter
        * The result page content is a subset of all possible correct
        page content
        * The result contains only file_names the user selected
        * The result contains only file_names from permitted S3 keys
    * If documents are selected and there's no permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's no permission to get them
        * The length of the result is zero

    And that:

    * The result is an appropriate update to RedboxState
    * The DocumentState is the right shape
    """
    for k, v in chain_params.items():
        setattr(stored_file_parameterised.query.ai_settings, k, v)

    selected_docs = stored_file_parameterised.get_docs_matching_query()
    permitted_docs = stored_file_parameterised.get_all_permitted_docs()

    selected = bool(stored_file_parameterised.query.s3_keys)
    permission = bool(stored_file_parameterised.query.permitted_s3_keys)

    # Build and run
    search = build_search_documents_tool(
        es_client=es_client,
        index_name=es_index,
        embedding_model=embedding_model,
        embedding_field_name=env.embedding_document_field_name,
        chunk_resolution=ChunkResolution.normal,
    )

    tool_node = ToolNode(tools=[search])
    result_state = tool_node.invoke(
        RedboxState(
            request=stored_file_parameterised.query,
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_search_documents",
                            "args": {"query": stored_file_parameterised.query.question},
                            "id": "1",
                        }
                    ],
                )
            ],
        )
    )

    if not permission:
        assert result_state["messages"][0].content == ""
        assert result_state["messages"][0].artifact == []
    elif not selected:
        assert result_state["messages"][0].content == ""
        assert result_state["messages"][0].artifact == []
    else:
        print(result_state["messages"][0])
        print("goodbye")
        result_flat = result_state["messages"][0].artifact
        print(f"DEBUG: result_flat = {result_flat}")  # Debugging

        assert result_flat is not None, "Error: result_flat is None"
        assert isinstance(result_state, dict)
        assert len(result_state) == 1
        assert len(result_flat) == chain_params["rag_k"]

        assert {c.page_content for c in result_flat} <= {c.page_content for c in permitted_docs}
        assert {c.metadata["uri"] for c in result_flat} <= set(stored_file_parameterised.query.permitted_s3_keys)

        if selected:
            assert {c.page_content for c in result_flat} <= {c.page_content for c in selected_docs}
            assert {c.metadata["uri"] for c in result_flat} <= set(stored_file_parameterised.query.s3_keys)


@pytest.mark.xfail(reason="calls api")
def test_govuk_search_tool():
    tool = build_govuk_search_tool()

    tool_node = ToolNode(tools=[tool])
    response = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_search_govuk",
                            "args": {"query": "Cuba Travel Advice"},
                            "id": "1",
                        }
                    ],
                )
            ]
        }
    )
    assert response["messages"][0].content != ""

    # assert at least one document is travel advice
    assert any(
        "/foreign-travel-advice/cuba" in document.metadata["uri"] for document in response["messages"][0].artifact
    )

    for document in response["messages"][0].artifact:
        assert document.page_content != ""
        metadata = ChunkMetadata.model_validate(document.metadata)
        assert urlparse(metadata.uri).hostname == "www.gov.uk"
        assert metadata.creator_type == ChunkCreatorType.gov_uk


@pytest.mark.xfail(reason="calls api")
def test_wikipedia_tool():
    tool = build_search_wikipedia_tool()
    tool_node = ToolNode(tools=[tool])
    response = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "_search_wikipedia",
                            "args": {"query": "What was the highest office held by Gordon Brown"},
                            "id": "1",
                        }
                    ],
                )
            ]
        }
    )
    assert response["messages"][0].content != ""

    for document in response["messages"][0].artifact:
        assert document.page_content != ""
        metadata = ChunkMetadata.model_validate(document.metadata)
        assert urlparse(metadata.uri).hostname == "en.wikipedia.org"
        assert metadata.creator_type == ChunkCreatorType.wikipedia


@pytest.mark.parametrize(
    "is_filter, relevant_return, query, keyword",
    [
        (False, False, "UK government use of AI", "artificial intelligence"),
        (True, True, "UK government use of AI", "artificial intelligence"),
    ],
)
@pytest.mark.vcr
@pytest.mark.xfail(reason="calls api")
def test_gov_filter_AI(is_filter, relevant_return, query, keyword):
    def run_tool(is_filter):
        tool = build_govuk_search_tool(filter=is_filter)
        state_update = tool.invoke(
            {
                "query": query,
                "state": RedboxState(
                    request=RedboxQuery(
                        question=query,
                        s3_keys=[],
                        user_uuid=uuid4(),
                        chat_history=[],
                        ai_settings=AISettings(),
                        permitted_s3_keys=[],
                    )
                ),
            }
        )

        return flatten_document_state(state_update["documents"])

    # call gov tool without additional filter
    documents = run_tool(is_filter)
    assert any(keyword in document.page_content for document in documents) == relevant_return


@pytest.mark.parametrize(
    "provider, web_results",
    [
        (
            "Brave",
            [
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "1"}},
                {
                    "status_code": 200,
                    "json": {"web": {"results": [{"extra_snippets": ["fake_doc"], "url": "http://fake.com"}]}},
                },
            ],
        ),
        (
            "Brave",
            [
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "1"}},
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "2"}},
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "3"}},
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "4"}},
            ],
        ),
        (
            "Kagi",
            [
                {"status_code": 429, "text": "Too many requests", "headers": {"Retry-After": "1"}},
                {"status_code": 200, "json": {"data": [{"t": 0, "snippet": "fake doc", "url": "http://fake.com"}]}},
            ],
        ),
    ],
)
@requests_mock.Mocker(kw="mock")
def test_web_search_rate_limit(provider, web_results, mocker, **kwargs):
    # mock setting
    env = Settings(web_search=provider)
    mocker.patch("redbox.graph.nodes.tools.get_settings", return_value=env)

    kwargs["mock"].get(
        env.web_search_settings().end_point,
        web_results,
    )

    response = web_search_with_retry(query="hello")

    assert kwargs["mock"].call_count <= 3
    assert kwargs["mock"].called
    assert isinstance(response, Response)


class TestWebSearchCall:
    def test_web_search_fail(self, mocker: MockerFixture):
        mock_response = Response()
        mock_response.status_code = 429
        mock_response._content = "Rate limit".encode("utf-8")
        mocker.patch("redbox.graph.nodes.tools.web_search_with_retry", return_value=mock_response)
        response = web_search_call(query="hello")
        assert response[0] == ""

    def test_brave_success(self, mocker: MockerFixture):
        mock_response = Response()
        mock_response.status_code = 200
        mock_response._content = (
            '{"web": {"results": [{"extra_snippets": ["fake_doc"], "url": "http://fake.com"}]}}'.encode("utf-8")
        )
        mocker.patch("redbox.graph.nodes.tools.web_search_with_retry", return_value=mock_response)
        response = web_search_call(query="hello")
        assert len(response[0]) > 0


class TestGovTool:
    @pytest.mark.parametrize(
        "test_name, query, web_response, no_of_artifact",
        [
            ("empty query", "", [], 0),
            (
                "success path",
                "test query",
                [{"description": "fake", "indexable_content": "AI", "link": "test", "format": "html", "title": "fake"}],
                1,
            ),
            (
                "empty description",
                "test query",
                [{"description": "", "indexable_content": "AI", "link": "test", "format": "html", "title": "fake"}],
                1,
            ),
            (
                "missing field",
                "test query",
                [
                    {"description": "", "indexable_content": "AI", "link": "test", "format": "html", "title": "fake"},
                    {"description": "foo", "title": "fake", "indexable_content": "foo"},
                ],
                1,
            ),
        ],
    )
    @requests_mock.Mocker(kw="mock")
    def test_gov_uk_tool(self, test_name, query, web_response, no_of_artifact, mocker: MockerFixture, **kwargs):
        tool = build_govuk_search_tool(filter=True)
        ai_setting = AISettings(tool_govuk_returned_results=2)

        tool_node = ToolNode(tools=[tool])

        # mock embedding
        mock_embedding = mocker.patch("redbox.graph.nodes.tools.get_embeddings")
        mock_embedding.return_value = FakeEmbeddings(size=1024)

        kwargs["mock"].get(
            re.compile(r".*gov\.uk.*"),
            [
                {
                    "status_code": 200,
                    "json": {"results": web_response},
                },
            ],
        )

        response = tool_node.invoke(
            {
                "request": RedboxQuery(
                    question=query,
                    s3_keys=[],
                    user_uuid=uuid4(),
                    chat_history=[],
                    ai_settings=ai_setting,
                    permitted_s3_keys=[],
                ),
                "messages": [
                    AIMessage(
                        content=test_name,
                        tool_calls=[
                            {
                                "name": "_search_govuk",
                                "args": {"query": query},
                                "id": "1",
                            }
                        ],
                    )
                ],
            }
        )

        if no_of_artifact != 0:
            documents = response["messages"][-1].artifact
            assert documents[0].page_content == "AI"
            assert len(documents) == no_of_artifact
        else:
            assert response["messages"][-1].artifact == []


@requests_mock.Mocker(kw="mock")
def test_kagi_response_to_document(mocker, **kwargs):
    env = Settings(web_search="Kagi")
    mocker.patch("redbox.graph.nodes.tools.get_settings", return_value=env)
    kwargs["mock"].get(
        env.web_search_settings().end_point,
        [
            {
                "status_code": 200,
                "json": {
                    "data": [
                        {"t": 0, "snippet": "fake doc number 1", "url": "http://fake.com/page=1"},
                        {"t": 0, "snippet": "fake doc number 2", "url": "http://fake.com/page=2"},
                    ]
                },
            },
        ],
    )

    response = web_search_with_retry(
        query="hello",
        no_search_result=2,
        country_code="All",
        ui_lang="en-GB",
    )
    tokeniser = bedrock_tokeniser
    mapped_documents = []
    docs = kagi_response_to_documents(tokeniser, response, mapped_documents)

    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Document)


@pytest.mark.parametrize(
    "json_data, expected_docs_len",
    [
        (
            {"web": {"results": [{"extra_snippets": ["snippet1"], "url": "http://example.com/1"}]}},
            1,
        ),
        (
            {},
            0,
        ),
        (
            {"web": []},
            0,
        ),
        (
            {"web": ["not a dict"]},
            0,
        ),
        (
            {"web": {"results": []}},
            0,
        ),
    ],
)
@requests_mock.Mocker(kw="mock")
def test_brave_results_to_documents(json_data, expected_docs_len, mocker, **kwargs):
    env = Settings(web_search="Brave")
    mocker.patch("redbox.graph.nodes.tools.get_settings", return_value=env)
    kwargs["mock"].get(
        env.web_search_settings().end_point,
        [{"status_code": 200, "json": json_data}],
    )

    response = web_search_with_retry(
        query="test query",
        no_search_result=expected_docs_len or 1,
        country_code="All",
        ui_lang="en-GB",
    )
    mapped_documents = []
    docs = brave_response_to_documents(bedrock_tokeniser, response, mapped_documents)

    assert len(docs) == expected_docs_len
    for doc in docs:
        assert isinstance(doc, Document)


@pytest.mark.parametrize(
    "query, site, no_search_results",
    [
        ("What is AI?", "", 20),
        ("What is Dict in Python?", "stackoverflow.com", 20),
    ],
)
@pytest.mark.vcr
@pytest.mark.xfail(reason="calls api")
def test_web_search_tool(query, site, no_search_results):
    tool = build_web_search_tool()
    tool_node = ToolNode(tools=[tool])
    response = tool_node.invoke(
        [{"name": "_search_web", "args": {"query": query, "site": site}, "id": "1", "type": "tool_call"}]
    )
    assert response["messages"][0].content != ""
    assert len(response["messages"][0].artifact) <= no_search_results


@pytest.mark.parametrize(
    "query, no_search_results",
    [
        ("tell me about AI legislation", 20),
        ("What is the new upcoming temporary piece of legislation regarding road traffic in Scotland", 20),
    ],
)
@pytest.mark.vcr
@pytest.mark.xfail(reason="calls api")
def test_legislation_search_tool(query, no_search_results):
    tool = build_legislation_search_tool()
    tool_node = ToolNode(tools=[tool])
    response = tool_node.invoke(
        [
            {
                "name": "_search_legislation",
                "args": {"query": query},
                "id": "1",
                "type": "tool_call",
            }
        ]
    )
    assert response["messages"][0].content != ""
    assert len(response["messages"][0].artifact) <= no_search_results
    for res in response["messages"][0].artifact:
        netloc = urlparse(res.metadata["uri"]).netloc
        assert "www.legislation.gov.uk" == netloc


def test_reduce_chunks_by_tokens_empty_chunks():
    """
    Test when chunks is None or empty.

    Asserts:
    * When chunks is None, the function returns a list containing only the provided chunk
    * When chunks is an empty list, the function returns a list containing only the provided chunk
    """
    chunk = Document(page_content="Test content", metadata={"token_count": 10})

    result = reduce_chunks_by_tokens(None, chunk, 100)
    assert len(result) == 1
    assert result[0] == chunk

    result = reduce_chunks_by_tokens([], chunk, 100)
    assert len(result) == 1
    assert result[0] == chunk


def test_reduce_chunks_by_tokens_combine():
    """
    Test when the new chunk can be combined with the last chunk.

    Asserts:
    * The function combines the last chunk and the new chunk
    * The returned list has the same length as the input list
    * The combination is done correctly using combine_documents
    """
    last_chunk = Document(page_content="Last chunk", metadata={"token_count": 30})
    new_chunk = Document(page_content="New chunk", metadata={"token_count": 20})
    chunks = [Document(page_content="First chunk", metadata={"token_count": 25}), last_chunk]

    result = reduce_chunks_by_tokens(chunks, new_chunk, 100)

    assert len(result) == 2
    assert result[0] == chunks[0]

    expected_combined = combine_documents(last_chunk, new_chunk)
    assert result[1] == expected_combined


def test_reduce_chunks_by_tokens_append():
    """
    Test when the new chunk cannot be combined with the last chunk.

    Asserts:
    * The function appends the new chunk to the list
    * The returned list is one element longer than the input list
    * The original chunks are unchanged
    """
    last_chunk = Document(page_content="Last chunk", metadata={"token_count": 80})
    new_chunk = Document(page_content="New chunk", metadata={"token_count": 30})
    chunks = [Document(page_content="First chunk", metadata={"token_count": 25}), last_chunk]

    result = reduce_chunks_by_tokens(chunks, new_chunk, 100)

    assert len(result) == 3
    assert result[0] == chunks[0]
    assert result[1] == chunks[1]
    assert result[2] == new_chunk


def test_reduce_chunks_by_tokens_exact_boundary():
    """
    Test when the token counts sum exactly to the maximum.

    Asserts:
    * The function combines chunks when the sum equals the max_tokens
    """
    last_chunk = Document(page_content="Last chunk", metadata={"token_count": 50})
    new_chunk = Document(page_content="New chunk", metadata={"token_count": 50})
    chunks = [last_chunk]

    # Should combine since 50 + 50 = 100
    result = reduce_chunks_by_tokens(chunks, new_chunk, 100)

    assert len(result) == 1
    expected_combined = combine_documents(last_chunk, new_chunk)
    assert result[0] == expected_combined


def test_reduce_chunks_by_tokens_multiple_operations(monkeypatch):
    """
    Test multiple operations in sequence to ensure state is maintained correctly.

    Asserts:
    * The function correctly handles a sequence of operations
    """

    def mock_combine(doc1, doc2):
        combined_content = f"{doc1.page_content} + {doc2.page_content}"
        combined_tokens = doc1.metadata["token_count"] + doc2.metadata["token_count"]
        return Document(page_content=combined_content, metadata={"token_count": combined_tokens})

    monkeypatch.setattr("redbox.transform.combine_documents", mock_combine)

    chunks = []
    max_tokens = 100

    chunk1 = Document(page_content="Chunk 1", metadata={"token_count": 40})
    chunks = reduce_chunks_by_tokens(chunks, chunk1, max_tokens)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Chunk 1"

    chunk2 = Document(page_content="Chunk 2", metadata={"token_count": 30})
    chunks = reduce_chunks_by_tokens(chunks, chunk2, max_tokens)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Chunk 1Chunk 2"
    assert chunks[0].metadata["token_count"] == 70

    chunk3 = Document(page_content="Chunk 3", metadata={"token_count": 50})
    chunks = reduce_chunks_by_tokens(chunks, chunk3, max_tokens)
    assert len(chunks) == 2
    assert chunks[0].page_content == "Chunk 1Chunk 2"
    assert chunks[1].page_content == "Chunk 3"

    chunk4 = Document(page_content="Chunk 4", metadata={"token_count": 40})
    chunks = reduce_chunks_by_tokens(chunks, chunk4, max_tokens)
    assert len(chunks) == 2
    assert chunks[0].page_content == "Chunk 1Chunk 2"
    assert chunks[1].page_content == "Chunk 3Chunk 4"
    assert chunks[1].metadata["token_count"] == 90

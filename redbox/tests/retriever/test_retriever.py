import pytest
from langchain_core.messages import HumanMessage

from redbox.models.chain import RedboxState
from redbox.retriever import (
    AllElasticsearchRetriever,
    MetadataRetriever,
    ParameterisedElasticsearchRetriever,
)
from redbox.retriever.retrievers import KnowledgeBaseTabularMetadataRetriever, TabularReconstructionMixin
from redbox.test.data import RedboxChatTestCase

TEST_CHAIN_PARAMETERS = (
    {
        "rag_k": 0,
        "rag_num_candidates": 100,
        "match_boost": 1,
        "knn_boost": 2,
        "similarity_threshold": 0,
        "elbow_filter_enabled": True,
        "rag_gauss_scale_size": 3,
        "rag_gauss_scale_decay": 0.5,
        "rag_gauss_scale_min": 1.1,
        "rag_gauss_scale_max": 2.0,
    },
    {
        "rag_k": 0,
        "rag_num_candidates": 100,
        "match_boost": 1,
        "knn_boost": 2,
        "similarity_threshold": 0,
        "elbow_filter_enabled": False,
        "rag_gauss_scale_size": 1,
        "rag_gauss_scale_decay": 0.1,
        "rag_gauss_scale_min": 1.0,
        "rag_gauss_scale_max": 1.0,
    },
)


@pytest.mark.parametrize("chain_params", TEST_CHAIN_PARAMETERS)
def test_parameterised_retriever(
    chain_params: dict,
    parameterised_retriever: ParameterisedElasticsearchRetriever,
    stored_file_parameterised: RedboxChatTestCase,
):
    """
    Given a RedboxState, asserts:

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

    Recall that build_retriever_process pays attention to state["text"],
    NOT to state["question"] when choosing what to search with.
    """
    for k, v in chain_params.items():
        setattr(stored_file_parameterised.query.ai_settings, k, v)

    result = parameterised_retriever.invoke(
        RedboxState(
            request=stored_file_parameterised.query,
            messages=[HumanMessage(content=stored_file_parameterised.query.question)],
        )
    )
    selected_docs = stored_file_parameterised.get_docs_matching_query()
    permitted_docs = stored_file_parameterised.get_all_permitted_docs()

    selected = bool(stored_file_parameterised.query.s3_keys)
    permission = bool(stored_file_parameterised.query.permitted_s3_keys)

    if not permission:
        assert len(result) == 0
    elif not selected:
        assert len(result) == 0
    else:
        assert len(result) == chain_params["rag_k"]
        assert {c.page_content for c in result} <= {c.page_content for c in permitted_docs}
        assert {c.metadata["uri"] for c in result} <= set(stored_file_parameterised.query.permitted_s3_keys)

        if selected:
            assert {c.page_content for c in result} <= {c.page_content for c in selected_docs}
            assert {c.metadata["uri"] for c in result} <= set(stored_file_parameterised.query.s3_keys)


def test_all_chunks_retriever(
    all_chunks_retriever: AllElasticsearchRetriever, stored_file_all_chunks: RedboxChatTestCase
):
    """
    Given a RedboxState, asserts:

    * If documents are selected and there's permission to get them
        * The length of the result is equal to the total stored chunks
        * The result page content is identical to all possible correct
        page content
        * The result contains exactly file_names the user selected
        * The result contains a subset of file_names from permitted S3 keys
    * If documents are selected and there's no permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's no permission to get them
        * The length of the result is zero
    """
    result = all_chunks_retriever.invoke(RedboxState(request=stored_file_all_chunks.query))
    correct = stored_file_all_chunks.get_docs_matching_query()

    selected = bool(stored_file_all_chunks.query.s3_keys)
    permission = bool(stored_file_all_chunks.query.permitted_s3_keys)

    if selected and permission:
        assert len(result) == len(correct)
        assert {c.page_content for c in result} == {c.page_content for c in correct}
        assert {c.metadata["uri"] for c in result} == set(stored_file_all_chunks.query.s3_keys)
        assert {c.metadata["uri"] for c in result} <= set(stored_file_all_chunks.query.permitted_s3_keys)
    else:
        len(result) == 0


def test_metadata_retriever(metadata_retriever: MetadataRetriever, stored_file_metadata: RedboxChatTestCase):
    """
    Given a RedboxState, asserts:

    * If documents are selected and there's permission to get them
        * The length of the result is equal to the total stored chunks
        * The result contains exactly file_names the user selected
        * The result contains a subset of file_names from permitted S3 keys
    * If documents are selected and there's no permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's permission to get them
        * The length of the result is zero
    * If documents aren't selected and there's no permission to get them
        * The length of the result is zero
    """

    result = metadata_retriever.invoke(RedboxState(request=stored_file_metadata.query))
    correct = stored_file_metadata.get_docs_matching_query()

    selected = bool(stored_file_metadata.query.s3_keys)
    permission = bool(stored_file_metadata.query.permitted_s3_keys)

    if selected and permission:
        assert len(result) == len(correct)
        assert {c.metadata["uri"] for c in result} == set(stored_file_metadata.query.s3_keys)
        assert {c.metadata["uri"] for c in result} <= set(stored_file_metadata.query.permitted_s3_keys)
    else:
        len(result) == 0


def test_tabular_kb_retriever(
    kb_tabular_metadata_retriever: KnowledgeBaseTabularMetadataRetriever,
    stored_file_tabular_kb: RedboxChatTestCase,
):
    """
    Given a RedboxState, asserts that the tabular retriever:

    * Retrieves only selected and permitted files
    * Returns documents grouped in a DocumentState
    * Populates state.knowledge_tabular_files
    """

    # Invoke the retriever
    result = kb_tabular_metadata_retriever.invoke(RedboxState(request=stored_file_tabular_kb.query))
    correct = stored_file_tabular_kb.get_kb_docs_matching_query()

    # Determine if any files were selected and permitted
    selected = bool(stored_file_tabular_kb.query.knowledge_base_s3_keys)
    permission = bool(stored_file_tabular_kb.query.permitted_s3_keys)

    if selected and permission:
        assert len(result) == len(correct)
        assert {c["metadata"]["uri"] for c in result} == set(stored_file_tabular_kb.query.knowledge_base_s3_keys)
        assert {c["metadata"]["uri"] for c in result} <= set(stored_file_tabular_kb.query.permitted_s3_keys)
    else:
        len(result) == 0


class TestTabularReconstructionMixin:
    def make_doc(self, uri, name, index, text, columns=None):
        return {
            "text": text,
            "metadata": {
                "uri": uri,
                "index": index,
                "document_schema": {
                    "type": "tabular",
                    "name": name,
                    "columns": columns or {"col1": "TEXT"},
                },
            },
        }

    def test_reconstruct_orders_by_index(self):
        mixin = TabularReconstructionMixin()

        docs = [
            self.make_doc("file1", "sheet", 2, "h\nb2"),
            self.make_doc("file1", "sheet", 0, "h\nb0"),
            self.make_doc("file1", "sheet", 1, "h\nb1"),
        ]

        out = mixin._reconstruct_tables(docs)

        assert len(out) == 1
        assert out[0]["text"].startswith("h")
        assert "b0" in out[0]["text"]
        assert "b1" in out[0]["text"]
        assert "b2" in out[0]["text"]

        # ensure ordering applied
        assert out[0]["text"].find("b0") < out[0]["text"].find("b1")

    def test_header_is_stripped_after_first_chunk(self):
        mixin = TabularReconstructionMixin()

        docs = [
            self.make_doc("file1", "sheet", 0, "h\nrow0"),
            self.make_doc("file1", "sheet", 1, "h\nrow1"),
        ]

        out = mixin._reconstruct_tables(docs)[0]["text"]

        # header appears only once
        assert out.count("h") == 1
        assert "row0" in out
        assert "row1" in out

    def test_different_tables_are_not_merged(self):
        mixin = TabularReconstructionMixin()

        docs = [
            self.make_doc("file1", "sheetA", 0, "h\nA0"),
            self.make_doc("file1", "sheetB", 0, "h\nB0"),
        ]

        out = mixin._reconstruct_tables(docs)

        assert len(out) == 2

    def test_standalone_docs_pass_through(self):
        mixin = TabularReconstructionMixin()

        docs = [
            {
                "text": "not tabular",
                "metadata": {
                    "uri": "file1",
                    "index": 0,
                    "document_schema": {"type": "text"},
                },
            }
        ]

        out = mixin._reconstruct_tables(docs)

        assert out == docs

    def test_missing_index_falls_back_to_largest_chunk(self):
        mixin = TabularReconstructionMixin()

        docs = [
            {
                "text": "small",
                "metadata": {
                    "uri": "file1",
                    "document_schema": {
                        "type": "tabular",
                        "name": "sheet",
                        "columns": {"a": "TEXT"},
                    },
                },
            },
            {
                "text": "much larger chunk content",
                "metadata": {
                    "uri": "file1",
                    "document_schema": {
                        "type": "tabular",
                        "name": "sheet",
                        "columns": {"a": "TEXT"},
                    },
                },
            },
        ]

        out = mixin._reconstruct_tables(docs)

        assert len(out) == 1
        assert "larger" in out[0]["text"]

    def test_reconstructed_flag_set(self):
        mixin = TabularReconstructionMixin()

        docs = [
            self.make_doc("file1", "sheet", 0, "h\nr0"),
            self.make_doc("file1", "sheet", 1, "h\nr1"),
        ]

        out = mixin._reconstruct_tables(docs)[0]

        assert out["metadata"]["document_schema"]["reconstructed"] is True

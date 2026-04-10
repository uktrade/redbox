from uuid import uuid4

import json
import pytest
from langchain_core.documents import Document

from redbox.models.chain import RedboxQuery
from redbox.models.file import ChunkResolution, ChunkCreatorType
from redbox.test.data import RedboxChatTestCase, RedboxTestData, generate_test_cases
from redbox.api.format import MCPResponseMetadata, format_documents

KNOWLEDGE_BASE_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
                knowledge_base_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(number_of_docs=1, tokens_in_all_docs=1000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
                knowledge_base_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="Empty knowledge base",
        ),
    ]
    for test_case in generator
]

ALL_CHUNKS_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="No permitted S3 keys",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=8,
                    tokens_in_all_docs=8_000,
                    chunk_resolution=ChunkResolution.largest,
                    s3_keys=["s3_key"],
                )
            ],
            test_id="Empty keys but permitted",
        ),
    ]
    for test_case in generator
]

PARAMETERISED_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.normal)
            ],
            test_id="Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.normal)
            ],
            test_id="No permitted S3 keys",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=8,
                    tokens_in_all_docs=8_000,
                    chunk_resolution=ChunkResolution.normal,
                    s3_keys=["s3_key"],
                )
            ],
            test_id="Empty keys but permitted",
        ),
    ]
    for test_case in generator
]

METADATA_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=["s3_key"],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(number_of_docs=8, tokens_in_all_docs=8000, chunk_resolution=ChunkResolution.largest)
            ],
            test_id="No permitted S3 keys",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Irrelevant Question",
                s3_keys=[],
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["s3_key"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=8,
                    tokens_in_all_docs=8_000,
                    chunk_resolution=ChunkResolution.largest,
                    s3_keys=["s3_key"],
                )
            ],
            test_id="Empty keys but permitted",
        ),
    ]
    for test_case in generator
]

TABULAR_RETRIEVER_KB_CASES: list[RedboxChatTestCase] = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Retrieve tabular data",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["file1.csv", "file2.xlsx"],
                knowledge_base_s3_keys=["file1.csv", "file2.xlsx"],
                s3_keys=[],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="KB-Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="No permission case",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
                knowledge_base_s3_keys=[],
                s3_keys=[],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="KB-No Files",
        ),
    ]
    for test_case in generator
]

TABULAR_RETRIEVER_USER_CASES: list[RedboxChatTestCase] = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Retrieve tabular data",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["file1.csv", "file2.xlsx"],
                knowledge_base_s3_keys=[],
                s3_keys=["file1.csv", "file2.xlsx"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="User-Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="No permission case",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],
                knowledge_base_s3_keys=[],
                s3_keys=["file1.csv", "file2.xlsx"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=2,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="User-No Permission",
        ),
    ]
    for test_case in generator
]

TABULAR_RETRIEVER_CASES: list[tuple[RedboxChatTestCase, bool]] = [
    pytest.param((test_case, knowledge_base), id=f"{test_case.test_id}-{'kb' if knowledge_base else 'user'}")
    for knowledge_base, test_case in [(True, tc) for tc in TABULAR_RETRIEVER_KB_CASES]
    + [(False, tc) for tc in TABULAR_RETRIEVER_USER_CASES]
]

WRAP_ASYNC_TOOL_RESULTS: list[tuple[str, str | list[Document]]] = [
    (
        '{"status": "success"}',
        '{"status": "success"}',
    ),
    (
        json.dumps(
            {
                "result_type": "nullable",
                "result": {"name": "BMW", "founded": 1916, "url": "https://example.com"},
            }
        ),
        [
            Document(
                page_content='{"name": "BMW", "founded": 1916, "url": "https://example.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://example.com",
                    "page_number": "",
                },
            )
        ],
    ),
    (
        '{"total": 0, "records": []}',
        '{"total": 0, "records": []}',
    ),
    (
        json.dumps(
            {
                "result_type": "paged",
                "result": {
                    "items": [
                        {"name": "BMW", "url": "https://example.com"},
                        {"name": "BMW2", "url": "https://example2.com"},
                    ],
                    "total": 2,
                    "page": 0,
                    "page_size": 10,
                },
                "metadata": MCPResponseMetadata(
                    user_feedback=MCPResponseMetadata.UserFeedback(
                        required=True, reason="Multiple company records returned user must clarify which one they want."
                    )
                ).model_dump(),
            }
        ),
        [
            Document(
                page_content='{"name": "BMW", "url": "https://example.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://example.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "BMW2", "url": "https://example2.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://example2.com",
                    "page_number": "",
                },
            ),
        ],
    ),
    (
        json.dumps(
            {
                "result_type": "multipaged",
                "result": {
                    "companies": {
                        "result": {
                            "items": [
                                {"name": "BMW", "url": "https://bmw.com"},
                                {"name": "Audi", "url": "https://audi.com"},
                            ],
                            "total": 2,
                            "page": 0,
                            "page_size": 10,
                        }
                    },
                    "interactions": {
                        "result": {
                            "items": [{"name": "Meeting", "url": "https://interaction.com"}],
                            "total": 1,
                            "page": 0,
                            "page_size": 10,
                        }
                    },
                },
                "metadata": MCPResponseMetadata(
                    user_feedback=MCPResponseMetadata.UserFeedback(
                        required=True, reason="Multiple company records returned user must clarify which one they want."
                    )
                ).model_dump(),
            }
        ),
        [
            Document(
                page_content='{"name": "BMW", "url": "https://bmw.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://bmw.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "Audi", "url": "https://audi.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://audi.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "Meeting", "url": "https://interaction.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://interaction.com",
                    "page_number": "",
                },
            ),
        ],
    ),
    (
        json.dumps(
            {
                "result_type": "composite",
                "result": [
                    {"name": "BMW", "url": "https://bmw.com", "founded": 1916},
                    {
                        "interactions": {
                            "result": {
                                "items": [
                                    {"name": "Meeting", "url": "https://interaction.com"},
                                    {"name": "Call", "url": "https://interaction2.com"},
                                ],
                                "total": 2,
                                "page": 0,
                                "page_size": 10,
                            }
                        }
                    },
                ],
            }
        ),
        [
            Document(
                page_content='{"name": "BMW", "url": "https://bmw.com", "founded": 1916}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://bmw.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "Meeting", "url": "https://interaction.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://interaction.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "Call", "url": "https://interaction2.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://interaction2.com",
                    "page_number": "",
                },
            ),
        ],
    ),
]

MCP_TOOL_RESULTS: list[tuple[tuple[str, MCPResponseMetadata], str]] = [
    (
        (tool_result, MCPResponseMetadata.model_validate(json.loads(tool_result).get("metadata") or {})),
        format_documents(parsed_result) if isinstance(parsed_result, list) else parsed_result,
    )
    for tool_result, parsed_result in WRAP_ASYNC_TOOL_RESULTS
]

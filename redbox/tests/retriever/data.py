from uuid import uuid4
from urllib.parse import urlencode

from langchain_core.documents import Document

from redbox.models.chain import RedboxQuery
from redbox.models.file import ChunkResolution, ChunkCreatorType
from redbox.test.data import RedboxTestData, generate_test_cases
from redbox.api.format import format_documents

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

TABULAR_KB_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(
                question="Retrieve tabular data",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["file1.csv", "file2.xlsx"],
                knowledge_base_s3_keys=["file1.csv", "file2.xlsx"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=4,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="Successful Path",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="No permission case",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=[],  # no permission
                knowledge_base_s3_keys=["file1.csv", "file2.xlsx"],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=4,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="No permitted S3 keys",
        ),
        generate_test_cases(
            query=RedboxQuery(
                question="Empty selection",
                user_uuid=uuid4(),
                chat_history=[],
                permitted_s3_keys=["file1.csv", "file2.xlsx"],  # permission exists
                knowledge_base_s3_keys=[],
            ),
            test_data=[
                RedboxTestData(
                    number_of_docs=4,
                    tokens_in_all_docs=1000,
                    chunk_resolution=ChunkResolution.tabular,
                    s3_keys=["file1.csv", "file2.xlsx"],
                )
            ],
            test_id="Empty keys but permitted",
        ),
    ]
    for test_case in generator
]


WRAP_ASYNC_TOOL_RESULTS = [
    (
        '{"status": "success"}',
        [
            Document(
                page_content='{"status": "success"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": f"test-server@1.0/company_tool?{urlencode({'company_name': 'BMW'})}",
                    "page_number": "",
                },
            )
        ],
    ),
    (
        '{"name": "BMW", "founded": 1916, "datahub_link": "https://example.com"}',
        [
            Document(
                page_content='{"name": "BMW", "founded": 1916, "datahub_link": "https://example.com"}',
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
        [
            Document(
                page_content='{"total": 0, "records": []}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": f"test-server@1.0/company_tool?{urlencode({'company_name': 'BMW'})}",
                    "page_number": "",
                },
            )
        ],
    ),
    (
        '{"total": 2, "records": [{"name": "BMW", "company_link": "https://example.com"}, {"name": "BMW2", "company_link": "https://example2.com"}]}',
        [
            Document(
                page_content='{"name": "BMW", "company_link": "https://example.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://example.com",
                    "page_number": "",
                },
            ),
            Document(
                page_content='{"name": "BMW2", "company_link": "https://example2.com"}',
                metadata={
                    "creator_type": ChunkCreatorType.datahub,
                    "uri": "https://example2.com",
                    "page_number": "",
                },
            ),
        ],
    ),
]

MCP_TOOL_RESULTS = [
    (tool_result, format_documents(parsed_result)) for tool_result, parsed_result in WRAP_ASYNC_TOOL_RESULTS
]

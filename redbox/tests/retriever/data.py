from uuid import uuid4

from redbox.models.chain import RedboxQuery
from redbox.models.file import ChunkResolution
from redbox.test.data import RedboxTestData, generate_test_cases

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

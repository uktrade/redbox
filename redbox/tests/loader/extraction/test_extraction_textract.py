import pytest
from unittest.mock import patch, MagicMock, call
from botocore.exceptions import ClientError

from redbox.loader.extraction.textract import TextractService, PdfChunk, TextractJobFailed
from unstructured.documents.elements import NarrativeText, Title, Header, Footer, ListItem, Table, Text, ElementMetadata


BUCKET = "test-bucket"
KEY = "docs/file.pdf"
JOB_ID = "abc123"


def make_service() -> TextractService:
    with patch("boto3.client"):
        return TextractService(bucket=BUCKET, region="eu-west-2")


def make_client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "msg"}}, "operation")


def make_blocks(*pages):
    return [
        {
            "BlockType": "LINE",
            "Page": page,
            "Text": line,
        }
        for page, lines in pages
        for line in lines
    ]


def make_layout_blocks():
    return [
        {
            "Id": "layout",
            "BlockType": "LAYOUT_TEXT",
            "Page": 1,
            "Relationships": [{"Type": "CHILD", "Ids": ["line"]}],
        },
        {
            "Id": "line",
            "BlockType": "LINE",
            "Text": "Hello",
            "Page": 1,
        },
    ]


class TestIsRetryableTextractError:
    @pytest.mark.parametrize(
        "code, expected",
        [
            ("ProvisionedThroughputExceededException", True),
            ("ThrottlingException", True),
            ("Throttling", True),
            ("RequestLimitExceeded", True),
            ("ValidationException", False),
            ("AccessDeniedException", False),
            ("InvalidS3ObjectException", False),
        ],
    )
    def test_client_error_codes(self, code, expected):
        svc = make_service()
        assert svc._is_retryable_textract_error(make_client_error(code)) == expected

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("boom"),
            ValueError("bad"),
            Exception("generic"),
        ],
    )
    def test_non_client_errors_are_not_retryable(self, error):
        assert make_service()._is_retryable_textract_error(error) is False


class TestRetryTextractRequest:
    def test_returns_immediately_on_success(self):
        svc = make_service()
        func = MagicMock(return_value={"JobId": JOB_ID})
        result = svc._retry_textract_request(func, JobId=JOB_ID)
        assert result == {"JobId": JOB_ID}
        func.assert_called_once_with(JobId=JOB_ID)

    @patch("time.sleep")
    def test_retries_on_throttling_then_succeeds(self, mock_sleep):
        svc = make_service()
        func = MagicMock(
            side_effect=[
                make_client_error("ThrottlingException"),
                make_client_error("ThrottlingException"),
                {"result": "ok"},
            ]
        )
        result = svc._retry_textract_request(func, max_attempts=5, base_delay=0)
        assert result == {"result": "ok"}
        assert func.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    @pytest.mark.parametrize(
        "code",
        [
            "ProvisionedThroughputExceededException",
            "ThrottlingException",
            "Throttling",
            "RequestLimitExceeded",
        ],
    )
    def test_raises_after_max_attempts_exhausted(self, mock_sleep, code):
        svc = make_service()
        func = MagicMock(side_effect=make_client_error(code))
        with pytest.raises(ClientError):
            svc._retry_textract_request(func, max_attempts=3, base_delay=0)
        assert func.call_count == 3

    def test_non_retryable_error_raises_immediately(self):
        svc = make_service()
        func = MagicMock(side_effect=make_client_error("AccessDeniedException"))
        with pytest.raises(ClientError):
            svc._retry_textract_request(func, max_attempts=5, base_delay=0)
        func.assert_called_once()


class TestWaitForJob:
    @patch("time.sleep")
    @pytest.mark.parametrize(
        "status_sequence, expected_status",
        [
            (["SUCCEEDED"], "SUCCEEDED"),
            (["FAILED"], "FAILED"),
            (["IN_PROGRESS", "SUCCEEDED"], "SUCCEEDED"),
            (["IN_PROGRESS", "IN_PROGRESS", "SUCCEEDED"], "SUCCEEDED"),
            (["IN_PROGRESS", "FAILED"], "FAILED"),
        ],
    )
    def test_polls_until_terminal_status(self, mock_sleep, status_sequence, expected_status):
        svc = make_service()
        getter = MagicMock(side_effect=[{"JobStatus": s} for s in status_sequence])
        result, _ = svc._wait_for_job(JOB_ID, getter)
        assert result == expected_status
        assert getter.call_count == len(status_sequence)

    @patch("time.sleep")
    def test_sleeps_between_polls(self, mock_sleep):
        svc = make_service()
        getter = MagicMock(
            side_effect=[
                {"JobStatus": "IN_PROGRESS"},
                {"JobStatus": "IN_PROGRESS"},
                {"JobStatus": "SUCCEEDED"},
            ]
        )
        svc._wait_for_job(JOB_ID, getter)
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_propagates_getter_exception(self, _mock_sleep):
        svc = make_service()
        getter = MagicMock(side_effect=RuntimeError("poll failed"))
        with pytest.raises(RuntimeError, match="poll failed"):
            svc._wait_for_job(JOB_ID, getter)

    @patch("time.sleep")
    def test_retries_on_throttling(self, mock_sleep):
        svc = make_service()

        throttling_error = ClientError(
            {
                "Error": {
                    "Code": "ProvisionedThroughputExceededException",
                    "Message": "Rate exceeded",
                }
            },
            "GetDocumentTextDetection",
        )

        getter = MagicMock(
            side_effect=[
                throttling_error,
                {"JobStatus": "SUCCEEDED"},
            ]
        )

        result, _ = svc._wait_for_job(JOB_ID, getter)

        assert result == "SUCCEEDED"
        assert getter.call_count == 2


class TestGetTextractResults:
    @pytest.mark.parametrize(
        "responses, expected_pages",
        [
            # single page, single API call
            (
                [{"Blocks": make_blocks((1, ["Line A", "Line B"])), "NextToken": None}],
                ["Line A\nLine B"],
            ),
            # two pages in one response
            (
                [{"Blocks": make_blocks((1, ["P1L1"]), (2, ["P2L1", "P2L2"])), "NextToken": None}],
                ["P1L1", "P2L1\nP2L2"],
            ),
            # paginated: two API calls
            (
                [
                    {"Blocks": make_blocks((1, ["L1"])), "NextToken": "tok"},
                    {"Blocks": make_blocks((2, ["L2"])), "NextToken": None},
                ],
                ["L1", "L2"],
            ),
            # blocks with non-LINE types are ignored
            (
                [
                    {
                        "Blocks": [
                            {"BlockType": "PAGE", "Page": 1, "Text": "ignored"},
                            {"BlockType": "LINE", "Page": 1, "Text": "kept"},
                        ],
                        "NextToken": None,
                    }
                ],
                ["kept"],
            ),
            # empty blocks -> empty result
            (
                [{"Blocks": [], "NextToken": None}],
                [],
            ),
        ],
    )
    def test_assembles_pages(self, responses, expected_pages):
        svc = make_service()

        first_response = responses[0]
        getter = MagicMock(side_effect=responses[1:])

        result = svc._get_textract_results(
            JOB_ID,
            getter,
            first_response,
        )

        assert result == expected_pages

    def test_pages_sorted_by_page_number(self):
        svc = make_service()

        first_response = {
            "Blocks": make_blocks((3, ["C"]), (1, ["A"]), (2, ["B"])),
            "NextToken": None,
        }

        getter = MagicMock()

        assert svc._get_textract_results(
            JOB_ID,
            getter,
            first_response,
        ) == ["A", "B", "C"]

    def test_passes_next_token_in_subsequent_calls(self):
        svc = make_service()

        first_response = {
            "Blocks": [],
            "NextToken": "page2token",
        }

        getter = MagicMock(
            side_effect=[
                {"Blocks": [], "NextToken": None},
            ]
        )

        svc._get_textract_results(
            JOB_ID,
            getter,
            first_response,
        )

        assert getter.call_args_list == [
            call(JobId=JOB_ID, NextToken="page2token"),
        ]

    def test_multiple_next_tokens_are_chained(self):
        svc = make_service()

        first_response = {
            "Blocks": [],
            "NextToken": "tok1",
        }

        getter = MagicMock(
            side_effect=[
                {"Blocks": [], "NextToken": "tok2"},
                {"Blocks": [], "NextToken": None},
            ]
        )

        svc._get_textract_results(
            JOB_ID,
            getter,
            first_response,
        )

        assert getter.call_args_list == [
            call(JobId=JOB_ID, NextToken="tok1"),
            call(JobId=JOB_ID, NextToken="tok2"),
        ]

    def test_propagates_getter_exception(self):
        svc = make_service()

        first_response = {
            "Blocks": [],
            "NextToken": "page2token",
        }

        getter = MagicMock(side_effect=RuntimeError("fetch failed"))

        with pytest.raises(RuntimeError, match="fetch failed"):
            svc._get_textract_results(
                JOB_ID,
                getter,
                first_response,
            )


class TestDocumentTextDetection:
    @patch("time.sleep")
    def test_success_returns_pages(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})

        svc.textract.get_document_text_detection = MagicMock(
            return_value={
                "JobStatus": "SUCCEEDED",
                "Blocks": make_blocks((1, ["Hello"])),
                "NextToken": None,
            }
        )

        result = svc.document_text_detection(KEY)

        assert result == ["Hello"]

    @patch("time.sleep")
    def test_raises_on_failed_job(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})
        svc.textract.get_document_text_detection = MagicMock(return_value={"JobStatus": "FAILED"})

        with pytest.raises(RuntimeError, match="document_text_detection"):
            svc.document_text_detection(KEY)

    @patch("time.sleep")
    def test_passes_correct_s3_location(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(return_value=("SUCCEEDED", {"JobStatus": "SUCCEEDED"}))

        svc.fetch_document_text_detection_result = MagicMock(return_value=[])

        svc.document_text_detection(KEY)

        svc.textract.start_document_text_detection.assert_called_once_with(
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": KEY}}
        )

        svc._wait_for_job.assert_called_once_with(
            job_id=JOB_ID,
            getter=svc.textract.get_document_text_detection,
            timeout=None,
        )

        svc.fetch_document_text_detection_result.assert_called_once_with(JOB_ID)

    def test_propagates_start_exception(self):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(side_effect=make_client_error("AccessDeniedException"))

        with pytest.raises(ClientError):
            svc.document_text_detection(KEY)

    @patch("time.sleep")
    def test_orchestrates_wait_and_results_correctly(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(return_value=("SUCCEEDED", {"JobStatus": "SUCCEEDED"}))

        svc.fetch_document_text_detection_result = MagicMock(return_value=["Hello"])

        result = svc.document_text_detection(KEY)

        assert result == ["Hello"]

        svc._wait_for_job.assert_called_once_with(
            job_id=JOB_ID,
            getter=svc.textract.get_document_text_detection,
            timeout=None,
        )

        svc.fetch_document_text_detection_result.assert_called_once_with(JOB_ID)

    @patch("time.sleep")
    def test_does_not_fetch_results_if_job_failed(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(
            return_value=(
                "FAILED",
                {"JobStatus": "FAILED"},
            )
        )

        svc._get_textract_results = MagicMock()

        with pytest.raises(RuntimeError):
            svc.document_text_detection(KEY)

        svc._get_textract_results.assert_not_called()


class TestDocumentAnalysis:
    @patch("time.sleep")
    def test_success_returns_elements(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})

        svc.textract.get_document_analysis = MagicMock(
            return_value={
                "JobStatus": "SUCCEEDED",
                "Blocks": [
                    {
                        "Id": "layout-1",
                        "BlockType": "LAYOUT_TEXT",
                        "Page": 1,
                        "Relationships": [
                            {
                                "Type": "CHILD",
                                "Ids": ["line-1", "line-2"],
                            }
                        ],
                    },
                    {
                        "Id": "line-1",
                        "BlockType": "LINE",
                        "Page": 1,
                        "Text": "Line 1",
                    },
                    {
                        "Id": "line-2",
                        "BlockType": "LINE",
                        "Page": 1,
                        "Text": "Line 2",
                    },
                ],
                "NextToken": None,
            }
        )

        result = svc.document_analysis(KEY)

        assert len(result) == 1

        element = result[0]
        assert isinstance(element, NarrativeText)
        assert element.text == "Line 1\nLine 2"
        assert element.metadata.page_number == 1

        svc.textract.start_document_analysis.assert_called_once_with(
            DocumentLocation={
                "S3Object": {
                    "Bucket": BUCKET,
                    "Name": KEY,
                }
            },
            FeatureTypes=["LAYOUT"],
        )

        assert svc.textract.get_document_analysis.call_count == 2
        svc.textract.get_document_analysis.assert_called_with(JobId=JOB_ID)

    @patch("time.sleep")
    def test_raises_on_failed_job(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})
        svc.textract.get_document_analysis = MagicMock(return_value={"JobStatus": "FAILED"})

        with pytest.raises(RuntimeError, match="document_analysis"):
            svc.document_analysis(KEY)

    @patch("time.sleep")
    def test_passes_layout_feature_type(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(return_value=("SUCCEEDED", {"JobStatus": "SUCCEEDED"}))

        svc.fetch_document_analysis_result = MagicMock(return_value=[])

        svc.document_analysis(KEY)

        svc.textract.start_document_analysis.assert_called_once_with(
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": KEY}},
            FeatureTypes=["LAYOUT"],
        )

        svc.fetch_document_analysis_result.assert_called_once_with(JOB_ID)

    def test_propagates_start_exception(self):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(side_effect=make_client_error("AccessDeniedException"))

        with pytest.raises(ClientError):
            svc.document_analysis(KEY)

    @patch("time.sleep")
    def test_orchestrates_wait_and_results_correctly(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(
            return_value=(
                "SUCCEEDED",
                {"JobStatus": "SUCCEEDED", "Blocks": [], "NextToken": None},
            )
        )

        expected = [NarrativeText(text="Hello")]
        svc.fetch_document_analysis_result = MagicMock(return_value=expected)

        result = svc.document_analysis(KEY)

        assert result == expected

        svc._wait_for_job.assert_called_once_with(
            job_id=JOB_ID, getter=svc.textract.get_document_analysis, timeout=None
        )

        svc.fetch_document_analysis_result.assert_called_once_with(JOB_ID)

    @patch("time.sleep")
    def test_does_not_fetch_results_if_job_failed(self, _mock_sleep):
        svc = make_service()

        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})

        svc._wait_for_job = MagicMock(
            return_value=(
                "FAILED",
                {"JobStatus": "FAILED"},
            )
        )

        svc.fetch_document_analysis_result = MagicMock()

        with pytest.raises(RuntimeError):
            svc.document_analysis(KEY)

        svc.fetch_document_analysis_result.assert_not_called()


class TestLayoutBlocksToElementsAllTypes:
    def _block(self, block_id, block_type, page=1, child_ids=None):
        block = {
            "Id": block_id,
            "BlockType": block_type,
            "Page": page,
        }
        if child_ids:
            block["Relationships"] = [{"Type": "CHILD", "Ids": child_ids}]
        return block

    def _line(self, line_id, text, page=1):
        return {"Id": line_id, "BlockType": "LINE", "Text": text, "Page": page}

    @pytest.mark.parametrize(
        "layout_type, expected_cls",
        [
            ("LAYOUT_TITLE", Title),
            ("LAYOUT_SECTION_HEADER", Title),
            ("LAYOUT_HEADER", Header),
            ("LAYOUT_FOOTER", Footer),
            ("LAYOUT_LIST", ListItem),
            ("LAYOUT_TABLE", Table),
            ("LAYOUT_TEXT", NarrativeText),
            ("LAYOUT_FIGURE", Text),  # unmapped -> default Text
        ],
    )
    def test_maps_layout_type_to_element_class(self, layout_type, expected_cls):
        svc = make_service()
        blocks = [
            self._block("layout", layout_type, page=2, child_ids=["line"]),
            self._line("line", "Some text", page=2),
        ]
        elements = svc._layout_blocks_to_elements(blocks)
        assert len(elements) == 1
        assert type(elements[0]) is expected_cls
        assert elements[0].text == "Some text"
        assert elements[0].metadata.page_number == 2

    def test_non_layout_blocks_are_ignored(self):
        svc = make_service()
        blocks = [
            self._block("page", "PAGE"),
            self._block("layout", "LAYOUT_TEXT", child_ids=["line"]),
            self._line("line", "Kept"),
        ]
        elements = svc._layout_blocks_to_elements(blocks)
        assert len(elements) == 1
        assert elements[0].text == "Kept"

    def test_blocks_with_empty_text_are_skipped(self):
        svc = make_service()
        blocks = [
            self._block("layout", "LAYOUT_TEXT", child_ids=["line"]),
            self._line("line", "   ", page=1),  # whitespace-only -> stripped to empty
        ]
        assert svc._layout_blocks_to_elements(blocks) == []

    def test_multiple_child_lines_joined_with_newline(self):
        svc = make_service()
        blocks = [
            self._block("layout", "LAYOUT_TEXT", child_ids=["l1", "l2"]),
            self._line("l1", "Line one"),
            self._line("l2", "Line two"),
        ]
        elements = svc._layout_blocks_to_elements(blocks)
        assert elements[0].text == "Line one\nLine two"

    def test_non_line_children_are_excluded_from_text(self):
        svc = make_service()
        blocks = [
            self._block("layout", "LAYOUT_TEXT", child_ids=["l1", "kv"]),
            self._line("l1", "Real line"),
            {"Id": "kv", "BlockType": "KEY_VALUE_SET", "Text": "ignored"},
        ]
        elements = svc._layout_blocks_to_elements(blocks)
        assert elements[0].text == "Real line"

    def test_missing_page_results_in_none_page_number(self):
        svc = make_service()
        blocks = [
            {
                "Id": "layout",
                "BlockType": "LAYOUT_TEXT",
                "Relationships": [{"Type": "CHILD", "Ids": ["line"]}],
            },
            {"Id": "line", "BlockType": "LINE", "Text": "No page"},
        ]
        elements = svc._layout_blocks_to_elements(blocks)
        assert elements[0].metadata.page_number is None


class TestSplitPdfToS3Chunks:
    def _make_fitz_mock(self, total_pages: int):
        """Builds a mock for the `fitz` module used inside textract.py."""
        mock_fitz = MagicMock()

        source_doc = MagicMock()
        source_doc.page_count = total_pages

        chunk_doc = MagicMock()
        chunk_doc.tobytes = MagicMock(return_value=b"chunk-bytes")

        # fitz.open(stream=..., filetype="pdf") -> source_doc
        # fitz.open() (no args) -> a fresh chunk_doc each time
        def open_side_effect(*args, **kwargs):
            if "stream" in kwargs:
                return source_doc
            return chunk_doc

        mock_fitz.open.side_effect = open_side_effect
        return mock_fitz, source_doc, chunk_doc

    def test_single_chunk_when_doc_smaller_than_chunk_size(self):
        svc = make_service()
        mock_fitz, source_doc, chunk_doc = self._make_fitz_mock(total_pages=3)
        svc.s3.put_object = MagicMock()

        with patch("redbox.loader.extraction.textract.fitz", mock_fitz):
            chunks = svc._split_pdf_to_s3_chunks(file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=10, overlap_pages=1)

        assert chunks == [PdfChunk(s3_key=f"{KEY}.textract-chunks/0000.pdf", start_page=0, end_page=3, overlap_start=0)]
        chunk_doc.insert_pdf.assert_called_once_with(source_doc, from_page=0, to_page=2)
        svc.s3.put_object.assert_called_once_with(
            Bucket=BUCKET, Key=f"{KEY}.textract-chunks/0000.pdf", Body=b"chunk-bytes"
        )
        source_doc.close.assert_called_once()

    def test_multiple_chunks_with_overlap(self):
        svc = make_service()
        mock_fitz, source_doc, chunk_doc = self._make_fitz_mock(total_pages=10)
        svc.s3.put_object = MagicMock()

        with patch("redbox.loader.extraction.textract.fitz", mock_fitz):
            chunks = svc._split_pdf_to_s3_chunks(file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=4, overlap_pages=1)

        assert chunks == [
            PdfChunk(s3_key=f"{KEY}.textract-chunks/0000.pdf", start_page=0, end_page=4, overlap_start=0),
            PdfChunk(s3_key=f"{KEY}.textract-chunks/0001.pdf", start_page=3, end_page=7, overlap_start=4),
            PdfChunk(s3_key=f"{KEY}.textract-chunks/0002.pdf", start_page=6, end_page=10, overlap_start=7),
        ]
        assert svc.s3.put_object.call_count == 3

    def test_no_overlap_when_overlap_pages_zero(self):
        svc = make_service()
        mock_fitz, source_doc, chunk_doc = self._make_fitz_mock(total_pages=6)
        svc.s3.put_object = MagicMock()

        with patch("redbox.loader.extraction.textract.fitz", mock_fitz):
            chunks = svc._split_pdf_to_s3_chunks(file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=3, overlap_pages=0)

        assert chunks == [
            PdfChunk(s3_key=f"{KEY}.textract-chunks/0000.pdf", start_page=0, end_page=3, overlap_start=0),
            PdfChunk(s3_key=f"{KEY}.textract-chunks/0001.pdf", start_page=3, end_page=6, overlap_start=3),
        ]

    def test_overlap_does_not_go_below_zero_on_first_chunk(self):
        # First chunk should never be shifted backwards regardless of overlap_pages
        svc = make_service()
        mock_fitz, source_doc, chunk_doc = self._make_fitz_mock(total_pages=5)
        svc.s3.put_object = MagicMock()

        with patch("redbox.loader.extraction.textract.fitz", mock_fitz):
            chunks = svc._split_pdf_to_s3_chunks(file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=5, overlap_pages=2)

        assert chunks[0].start_page == 0
        assert chunks[0].overlap_start == 0

    def test_chunk_keys_are_zero_padded_and_namespaced_under_key(self):
        svc = make_service()
        mock_fitz, source_doc, chunk_doc = self._make_fitz_mock(total_pages=2)
        svc.s3.put_object = MagicMock()

        with patch("redbox.loader.extraction.textract.fitz", mock_fitz):
            chunks = svc._split_pdf_to_s3_chunks(file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=1, overlap_pages=0)

        assert chunks[0].s3_key == f"{KEY}.textract-chunks/0000.pdf"
        assert chunks[1].s3_key == f"{KEY}.textract-chunks/0001.pdf"


class TestRunChunkDocumentAnalysis:
    def _chunk(self):
        return PdfChunk(s3_key="some/chunk.pdf", start_page=0, end_page=4, overlap_start=0)

    def test_returns_elements_on_success(self):
        svc = make_service()
        chunk = self._chunk()

        svc.start_document_analysis = MagicMock(return_value=JOB_ID)
        svc._wait_for_job = MagicMock(return_value=("SUCCEEDED", {"JobStatus": "SUCCEEDED"}))
        expected = [NarrativeText(text="hi")]
        svc.fetch_document_analysis_result = MagicMock(return_value=expected)

        result = svc._run_chunk_document_analysis(chunk)

        assert result == expected
        svc.start_document_analysis.assert_called_once_with(chunk.s3_key)
        svc._wait_for_job.assert_called_once_with(
            job_id=JOB_ID, getter=svc.textract.get_document_analysis, timeout=None
        )
        svc.fetch_document_analysis_result.assert_called_once_with(JOB_ID)

    def test_raises_textract_job_failed_when_not_succeeded(self):
        svc = make_service()
        chunk = self._chunk()

        svc.start_document_analysis = MagicMock(return_value=JOB_ID)
        svc._wait_for_job = MagicMock(return_value=("FAILED", {"JobStatus": "FAILED"}))
        svc.fetch_document_analysis_result = MagicMock()

        with pytest.raises(TextractJobFailed, match=chunk.s3_key):
            svc._run_chunk_document_analysis(chunk)

        svc.fetch_document_analysis_result.assert_not_called()

    def test_passes_timeout_through(self):
        svc = make_service()
        chunk = self._chunk()

        svc.start_document_analysis = MagicMock(return_value=JOB_ID)
        svc._wait_for_job = MagicMock(return_value=("SUCCEEDED", {}))
        svc.fetch_document_analysis_result = MagicMock(return_value=[])

        svc._run_chunk_document_analysis(chunk, timeout=42.0)

        svc._wait_for_job.assert_called_once_with(
            job_id=JOB_ID, getter=svc.textract.get_document_analysis, timeout=42.0
        )


class TestCleanupChunk:
    def _chunk(self, key="some/chunk.pdf"):
        return PdfChunk(s3_key=key, start_page=0, end_page=4, overlap_start=0)

    def test_deletes_chunk_object(self):
        svc = make_service()
        svc.s3.delete_object = MagicMock()
        chunk = self._chunk()

        svc._cleanup_chunk(chunk)

        svc.s3.delete_object.assert_called_once_with(Bucket=BUCKET, Key=chunk.s3_key)

    def test_swallows_exception_on_delete_failure(self):
        svc = make_service()
        svc.s3.delete_object = MagicMock(side_effect=RuntimeError("boom"))
        chunk = self._chunk()

        # Should not raise
        svc._cleanup_chunk(chunk)

        svc.s3.delete_object.assert_called_once()


class TestDocumentAnalysisLarge:
    def _make_chunks(self):
        # Mirrors a 10-page doc split into pages_per_chunk=4, overlap_pages=1
        return [
            PdfChunk(s3_key="k.textract-chunks/0000.pdf", start_page=0, end_page=4, overlap_start=0),
            PdfChunk(s3_key="k.textract-chunks/0001.pdf", start_page=3, end_page=7, overlap_start=4),
            PdfChunk(s3_key="k.textract-chunks/0002.pdf", start_page=6, end_page=10, overlap_start=7),
        ]

    def _elements_for_chunk(self, n_local_pages: int):
        # one NarrativeText element per chunk-local page, 1-indexed page numbers
        return [
            NarrativeText(text=f"page-{p}", metadata=ElementMetadata(page_number=p))
            for p in range(1, n_local_pages + 1)
        ]

    def test_merges_chunks_dropping_overlap_and_remapping_page_numbers(self):
        svc = make_service()
        chunks = self._make_chunks()

        svc._split_pdf_to_s3_chunks = MagicMock(return_value=chunks)
        svc._cleanup_chunk = MagicMock()

        chunk_elements = {
            chunks[0].s3_key: self._elements_for_chunk(4),  # local pages 1-4 -> original 1-4
            chunks[1].s3_key: self._elements_for_chunk(4),  # local pages 1-4 -> original 4-7, drop local page 1
            chunks[2].s3_key: self._elements_for_chunk(4),  # local pages 1-4 -> original 7-10, drop local page 1
        }

        def run_chunk_side_effect(chunk, timeout=None):
            # return fresh copies so mutating metadata.page_number doesn't leak across calls
            return [
                NarrativeText(text=e.text, metadata=ElementMetadata(page_number=e.metadata.page_number))
                for e in chunk_elements[chunk.s3_key]
            ]

        svc._run_chunk_document_analysis = MagicMock(side_effect=run_chunk_side_effect)

        result = svc.document_analysis_large(key=KEY, file_bytes=b"pdf-bytes")

        # chunk0: all 4 pages kept -> original pages 1,2,3,4
        # chunk1: drop_before = overlap_start(4) - start_page(3) = 1 -> local page 1 dropped, local 2,3,4 kept
        #         original_page = start_page(3) + local_page + 1 -> for local 2,3,4 => 6,7,8... wait check formula
        # original_page = chunk.start_page + local_page + 1, local_page is 0-indexed (element page-1)
        result_pages = [e.metadata.page_number for e in result]

        assert result_pages == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert len(result) == 10

        assert svc._cleanup_chunk.call_count == 3
        for chunk in chunks:
            svc._cleanup_chunk.assert_any_call(chunk)

    def test_cleanup_called_for_all_chunks_even_when_one_fails(self):
        svc = make_service()
        chunks = self._make_chunks()

        svc._split_pdf_to_s3_chunks = MagicMock(return_value=chunks)
        svc._cleanup_chunk = MagicMock()

        def run_chunk_side_effect(chunk, timeout=None):
            if chunk is chunks[1]:
                raise TextractJobFailed("boom")
            return []

        svc._run_chunk_document_analysis = MagicMock(side_effect=run_chunk_side_effect)

        with pytest.raises(TextractJobFailed):
            svc.document_analysis_large(key=KEY, file_bytes=b"pdf-bytes")

        assert svc._cleanup_chunk.call_count == 3
        for chunk in chunks:
            svc._cleanup_chunk.assert_any_call(chunk)

    def test_single_chunk_no_pages_dropped(self):
        svc = make_service()
        chunk = PdfChunk(s3_key="k.textract-chunks/0000.pdf", start_page=0, end_page=3, overlap_start=0)
        svc._split_pdf_to_s3_chunks = MagicMock(return_value=[chunk])
        svc._cleanup_chunk = MagicMock()
        svc._run_chunk_document_analysis = MagicMock(return_value=self._elements_for_chunk(3))

        result = svc.document_analysis_large(key=KEY, file_bytes=b"pdf-bytes")

        assert [e.metadata.page_number for e in result] == [1, 2, 3]
        svc._cleanup_chunk.assert_called_once_with(chunk)

    def test_passes_split_params_through(self):
        svc = make_service()
        svc._split_pdf_to_s3_chunks = MagicMock(return_value=[])
        svc._cleanup_chunk = MagicMock()

        svc.document_analysis_large(
            key=KEY, file_bytes=b"pdf-bytes", pages_per_chunk=50, overlap_pages=2, max_workers=3, timeout=99.0
        )

        svc._split_pdf_to_s3_chunks.assert_called_once_with(
            file_bytes=b"pdf-bytes", key=KEY, pages_per_chunk=50, overlap_pages=2
        )

    def test_passes_timeout_to_each_chunk_analysis(self):
        svc = make_service()
        chunk = PdfChunk(s3_key="k.textract-chunks/0000.pdf", start_page=0, end_page=2, overlap_start=0)
        svc._split_pdf_to_s3_chunks = MagicMock(return_value=[chunk])
        svc._cleanup_chunk = MagicMock()
        svc._run_chunk_document_analysis = MagicMock(return_value=[])

        svc.document_analysis_large(key=KEY, file_bytes=b"pdf-bytes", timeout=15.0)

        svc._run_chunk_document_analysis.assert_called_once_with(chunk, 15.0)

    def test_empty_document_returns_empty_list(self):
        svc = make_service()
        svc._split_pdf_to_s3_chunks = MagicMock(return_value=[])
        svc._cleanup_chunk = MagicMock()

        result = svc.document_analysis_large(key=KEY, file_bytes=b"pdf-bytes")

        assert result == []
        svc._cleanup_chunk.assert_not_called()

import pytest
from unittest.mock import patch, MagicMock, call
from botocore.exceptions import ClientError

from redbox.loader.extraction.textract import TextractService


BUCKET = "test-bucket"
KEY = "docs/file.pdf"
JOB_ID = "abc123"


def make_service() -> TextractService:
    with patch("boto3.client"):
        return TextractService(bucket=BUCKET, region="eu-west-2")


def make_client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "msg"}}, "operation")


def make_blocks(*pages: tuple[int, list[str]]) -> list[dict]:
    """Build a flat Textract Blocks list from (page_num, [lines]) tuples."""
    return [{"BlockType": "LINE", "Page": page, "Text": line} for page, lines in pages for line in lines]


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
        result = svc._wait_for_job(JOB_ID, getter)
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
        getter = MagicMock(side_effect=responses)
        result = svc._get_textract_results(JOB_ID, getter)
        assert result == expected_pages

    def test_pages_sorted_by_page_number(self):
        svc = make_service()
        # Return blocks out of order — page 3, then 1, then 2
        getter = MagicMock(
            return_value={
                "Blocks": make_blocks((3, ["C"]), (1, ["A"]), (2, ["B"])),
                "NextToken": None,
            }
        )
        assert svc._get_textract_results(JOB_ID, getter) == ["A", "B", "C"]

    def test_passes_next_token_in_subsequent_calls(self):
        svc = make_service()
        getter = MagicMock(
            side_effect=[
                {"Blocks": [], "NextToken": "page2token"},
                {"Blocks": [], "NextToken": None},
            ]
        )
        svc._get_textract_results(JOB_ID, getter)
        assert getter.call_args_list == [
            call(JobId=JOB_ID),
            call(JobId=JOB_ID, NextToken="page2token"),
        ]

    def test_propagates_getter_exception(self):
        svc = make_service()
        getter = MagicMock(side_effect=RuntimeError("fetch failed"))
        with pytest.raises(RuntimeError, match="fetch failed"):
            svc._get_textract_results(JOB_ID, getter)


class TestDocumentTextDetection:
    @patch("time.sleep")
    def test_success_returns_pages(self, _mock_sleep):
        svc = make_service()
        svc.textract.start_document_text_detection = MagicMock(return_value={"JobId": JOB_ID})
        svc.textract.get_document_text_detection = MagicMock(
            side_effect=[
                {"JobStatus": "SUCCEEDED"},
                {"Blocks": make_blocks((1, ["Hello"])), "NextToken": None},
            ]
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
        svc.textract.get_document_text_detection = MagicMock(
            side_effect=[
                {"JobStatus": "SUCCEEDED"},
                {"Blocks": [], "NextToken": None},
            ]
        )
        svc.document_text_detection(KEY)
        svc.textract.start_document_text_detection.assert_called_once_with(
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": KEY}}
        )

    def test_propagates_start_exception(self):
        svc = make_service()
        svc.textract.start_document_text_detection = MagicMock(side_effect=make_client_error("AccessDeniedException"))
        with pytest.raises(ClientError):
            svc.document_text_detection(KEY)


class TestDocumentAnalysis:
    @patch("time.sleep")
    def test_success_returns_pages(self, _mock_sleep):
        svc = make_service()
        svc.textract.start_document_analysis = MagicMock(return_value={"JobId": JOB_ID})
        svc.textract.get_document_analysis = MagicMock(
            side_effect=[
                {"JobStatus": "SUCCEEDED"},
                {"Blocks": make_blocks((1, ["Line 1"]), (2, ["Line 2"])), "NextToken": None},
            ]
        )
        result = svc.document_analysis(KEY)
        assert result == ["Line 1", "Line 2"]

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
        svc.textract.get_document_analysis = MagicMock(
            side_effect=[
                {"JobStatus": "SUCCEEDED"},
                {"Blocks": [], "NextToken": None},
            ]
        )
        svc.document_analysis(KEY)
        svc.textract.start_document_analysis.assert_called_once_with(
            DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": KEY}},
            FeatureTypes=["LAYOUT"],
        )

    def test_propagates_start_exception(self):
        svc = make_service()
        svc.textract.start_document_analysis = MagicMock(side_effect=make_client_error("AccessDeniedException"))
        with pytest.raises(ClientError):
            svc.document_analysis(KEY)

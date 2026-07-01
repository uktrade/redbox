from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from redbox.loader.extraction.metadata import MetadataExtraction
from redbox.models.chain import GeneratedMetadata


@pytest.fixture
def svc():
    with patch("redbox.loader.extraction.metadata.get_chat_llm") as mock_llm:
        llm = MagicMock()
        mock_llm.return_value = llm

        env = MagicMock()
        env.metadata_extraction_llm = "claude-3"
        env.metadata_prompt = ["Extract metadata"]

        return MetadataExtraction(env)


def make_metadata(**kwargs) -> GeneratedMetadata:
    defaults = {
        "name": "doc.pdf",
        "description": "A document",
        "keywords": ["kw"],
    }
    return GeneratedMetadata(**{**defaults, **kwargs})


@contextmanager
def patched_chain(return_value=None, side_effect=None):
    final_chain = MagicMock()

    if side_effect is not None:
        final_chain.invoke.side_effect = side_effect
    else:
        final_chain.invoke.return_value = return_value if return_value is not None else make_metadata()

    first_pipe = MagicMock()
    first_pipe.__or__.return_value = final_chain

    with (
        patch("redbox.loader.extraction.metadata.PromptTemplate") as mock_prompt_template,
        patch("redbox.loader.extraction.metadata.ClaudeParser"),
    ):
        mock_prompt_template.return_value.__or__.return_value = first_pipe

        yield final_chain


class TestCreateFileMetadata:
    def test_returns_llm_metadata_on_success(self, svc):
        expected = make_metadata(name="report.pdf")

        with patched_chain(return_value=expected):
            result = svc.create_file_metadata(
                "report.pdf",
                "some content",
            )

        assert result == expected

    def test_uses_file_name_when_metadata_name_is_empty(self, svc):
        returned = make_metadata(name="")

        with patched_chain(return_value=returned):
            result = svc.create_file_metadata(
                "fallback.pdf",
                "content",
            )

        assert result.name == "fallback.pdf"

    def test_prefers_original_metadata_filename_over_file_name(self, svc):
        returned = make_metadata(name="")

        with patched_chain(return_value=returned):
            result = svc.create_file_metadata(
                "arg_name.pdf",
                "content",
                original_metadata={"filename": "original.pdf"},
            )

        assert result.name == "original.pdf"

    def test_returns_fallback_metadata_on_validation_error(self, svc):
        validation_error = ValidationError.from_exception_data(
            title="GeneratedMetadata",
            line_errors=[],
        )

        with patched_chain(side_effect=validation_error):
            result = svc.create_file_metadata(
                "doc.pdf",
                "content",
            )

        assert isinstance(result, GeneratedMetadata)
        assert result.name == "doc.pdf"

    def test_fallback_uses_original_metadata_filename_on_validation_error(
        self,
        svc,
    ):
        validation_error = ValidationError.from_exception_data(
            title="GeneratedMetadata",
            line_errors=[],
        )

        with patched_chain(side_effect=validation_error):
            result = svc.create_file_metadata(
                "arg.pdf",
                "content",
                original_metadata={"filename": "original.pdf"},
            )

        assert result.name == "original.pdf"

    @pytest.mark.parametrize(
        "original_metadata",
        [
            None,
            {},
            {"unrelated": "key"},
        ],
    )
    def test_handles_missing_or_empty_original_metadata(
        self,
        svc,
        original_metadata,
    ):
        expected = make_metadata(name="doc.pdf")

        with patched_chain(return_value=expected):
            result = svc.create_file_metadata(
                "doc.pdf",
                "content",
                original_metadata=original_metadata,
            )

        assert result.name == "doc.pdf"


class TestTrimMetadata:
    @pytest.mark.parametrize(
        "input_meta",
        [
            {"note": "a" * 2000},
            {"nested": {"inner": "b" * 2000}},
            {"items": ["c" * 2000, "d" * 2000]},
            {"num": 42},
        ],
    )
    def test_trims_string_values_to_1000_chars(
        self,
        svc,
        input_meta,
    ):
        with (
            patch("redbox.loader.extraction.metadata.PromptTemplate") as mock_prompt_template,
            patch("redbox.loader.extraction.metadata.ClaudeParser"),
        ):
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = make_metadata()

            mock_prompt_template.return_value.__or__ = MagicMock(return_value=mock_chain)

            svc.create_file_metadata(
                "doc.pdf",
                "content",
                original_metadata=input_meta,
            )

            _, kwargs = mock_prompt_template.call_args
            trimmed = kwargs["partial_variables"]["original_metadata"]

        def assert_all_strings_short(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    assert_all_strings_short(value)
            elif isinstance(obj, list):
                for value in obj:
                    assert_all_strings_short(value)
            elif isinstance(obj, str):
                assert len(obj) <= 1000

        assert_all_strings_short(trimmed)


class TestExtract:
    @pytest.mark.parametrize(
        ("file_name", "expected_file_type"),
        [
            ("report.pdf", "PDF"),
            ("data.csv", "CSV"),
            ("sheet.xlsx", "Excel"),
            ("sheet.xls", "Excel"),
            ("document.docx", "DOCX"),
            ("notes.txt", "unknown"),
            ("archive.zip", "unknown"),
            ("DATA.PDF", "PDF"),
            ("DATA.CSV", "CSV"),
        ],
    )
    def test_detects_file_type(self, svc, file_name, expected_file_type):
        captured = {}

        def capturing_create(file_name, page_content, original_metadata=None):
            captured["original_metadata"] = original_metadata
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        svc.extract(file_name, ["content"])

        assert captured["original_metadata"]["file_type"] == expected_file_type
        assert captured["original_metadata"]["filename"] == file_name

    @pytest.mark.parametrize(
        ("elements", "expected_min_length"),
        [
            (["abc", "def"], 6),
            (["a" * 6000, "b" * 6000], 10_000),  # capped
            ([], 0),
        ],
    )
    def test_truncates_content_to_10k(self, svc, elements, expected_min_length):
        captured = {}

        def capturing_create(file_name, page_content, original_metadata=None):
            captured["page_content"] = page_content
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        svc.extract("doc.pdf", elements)

        assert len(captured["page_content"]) <= 10_000

        if expected_min_length:
            assert len(captured["page_content"]) >= min(expected_min_length, 10_000)

    def test_returns_fallback_on_create_exception(self, svc):
        svc.create_file_metadata = MagicMock(side_effect=RuntimeError("LLM down"))

        result = svc.extract(
            "doc.pdf",
            ["content"],
        )

        assert isinstance(result, GeneratedMetadata)
        assert result.name == "doc.pdf"

    def test_returns_generated_metadata_instance(self, svc):
        svc.create_file_metadata = MagicMock(return_value=make_metadata(name="doc.pdf"))

        result = svc.extract(
            "doc.pdf",
            ["content"],
        )

        assert isinstance(result, GeneratedMetadata)


class TestExtractDictElements:
    def test_formats_single_table(self, svc):
        captured = {}

        def capturing_create(file_name, page_content, original_metadata=None):
            captured["page_content"] = page_content
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        elements = [
            {
                "text": "id,name\n1,Jeff",
                "document_schema": {
                    "name": "employees",
                    "columns": {
                        "id": "NUMBER",
                        "name": "STRING",
                    },
                },
            }
        ]

        svc.extract("employees.csv", elements)

        content = captured["page_content"]

        assert "Table: employees" in content
        assert "Columns: id, name" in content
        assert "Column Types: id=NUMBER, name=STRING" in content
        assert "Sample:" in content
        assert "id,name" in content

    def test_prefers_all_tables_over_single_table(self, svc):
        captured = {}

        def capturing_create(file_name, page_content, original_metadata=None):
            captured["page_content"] = page_content
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        elements = [
            {
                "text": "id,name\n1,Jeff",
                "document_schema": {
                    "name": "employees",
                    "columns": {
                        "id": "NUMBER",
                        "name": "STRING",
                    },
                },
            },
            {
                "text": "dept_id,title\n10,HR",
                "document_schema": {
                    "name": "departments",
                    "columns": {
                        "dept_id": "NUMBER",
                        "title": "STRING",
                    },
                },
            },
        ]

        svc.extract("tables.xlsx", elements)

        content = captured["page_content"]

        assert "Table: employees" in content
        assert "Table: departments" in content

    def test_truncates_large_tabular_content(self, svc):
        captured = {}

        def capturing_create(file_name, page_content, original_metadata=None):
            captured["page_content"] = page_content
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        elements = [
            {
                "text": "a" * 20_000,
                "document_schema": {
                    "name": "big_table",
                    "columns": {
                        "value": "STRING",
                    },
                },
            }
        ]

        svc.extract("big.csv", elements)

        assert len(captured["page_content"]) <= 10_000
        assert "Table: big_table" in captured["page_content"]

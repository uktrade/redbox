from types import SimpleNamespace
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
    def test_detects_file_type(
        self,
        svc,
        file_name,
        expected_file_type,
    ):
        captured = {}

        def capturing_create(
            file_name,
            page_content,
            original_metadata=None,
        ):
            captured["original_metadata"] = original_metadata
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        svc.extract(file_name, ["content"])

        assert captured["original_metadata"]["file_type"] == expected_file_type

    @pytest.mark.parametrize(
        ("pages", "expected_prefix"),
        [
            (["abc", "def"], "abcdef"),
            (["a" * 6000, "b" * 6000], "a" * 6000),
            ([], ""),
        ],
    )
    def test_truncates_content_to_10k(
        self,
        svc,
        pages,
        expected_prefix,
    ):
        captured = {}

        def capturing_create(
            file_name,
            page_content,
            original_metadata=None,
        ):
            captured["page_content"] = page_content
            return make_metadata(name=file_name)

        svc.create_file_metadata = capturing_create

        elements = [SimpleNamespace(text=p) for p in pages]

        svc.extract("doc.pdf", elements)

        assert len(captured["page_content"]) <= 10_000
        assert captured["page_content"].startswith(expected_prefix[:100])

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


class TestExtractTabular:
    @pytest.mark.parametrize(
        ("elements", "expected_pages"),
        [
            ([{"text": "row1"}, {"text": "row2"}], ["row1", "row2"]),
            ([{"text": "only"}], ["only"]),
            ([], []),
            ([{"other_key": "ignored"}], [""]),
        ],
    )
    def test_extracts_text_from_elements(
        self,
        svc,
        elements,
        expected_pages,
    ):
        captured = {}

        def capturing_extract(file_name, pages):
            captured["pages"] = pages
            return make_metadata(name=file_name)

        svc.extract = capturing_extract

        svc.extract_tabular(
            "table.csv",
            elements,
        )

        assert captured["pages"] == expected_pages

    def test_delegates_to_extract_with_file_name(self, svc):
        svc.extract = MagicMock(return_value=make_metadata(name="table.csv"))

        svc.extract_tabular(
            "table.csv",
            [{"text": "data"}],
        )

        svc.extract.assert_called_once_with(
            file_name="table.csv",
            pages=["data"],
        )

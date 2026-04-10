import pytest
import json


from redbox.models.file import ChunkCreatorType
from redbox.api.format import format_mcp_tool_response, SensitiveValue, MCPResponseMetadata


class TestSensitiveValue:
    @pytest.mark.parametrize("value", ["abc123", {"token": "x"}, None, 42])
    def test_stores_and_retrieves(self, value):
        assert SensitiveValue(value).get() == value

    @pytest.mark.parametrize("value", ["super_secret", "abc123"])
    def test_redacts_in_repr_and_str(self, value):
        sv = SensitiveValue(value)
        assert value not in repr(sv)
        assert value not in str(sv)


class TestFormatMCPToolResponse:
    @pytest.mark.parametrize(
        "data",
        [
            {"key": "value"},
            {"total": 0, "results": []},
            {"name": "no-url"},
        ],
    )
    def test_returns_original_when_no_result_type(self, data):
        payload = json.dumps(data)
        result, metadata = format_mcp_tool_response(payload, ChunkCreatorType.datahub)
        assert result == payload
        assert metadata == MCPResponseMetadata()

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                {
                    "result_type": "nullable",
                    "result": {"url": "https://example.com", "title": "Example"},
                    "metadata": MCPResponseMetadata().model_dump(),
                },
                [("https://example.com", {"url": "https://example.com", "title": "Example"})],
            ),
            (
                {
                    "result_type": "paged",
                    "result": {
                        "items": [{"url": "https://a.com"}, {"url": "https://b.com"}],
                        "total": 2,
                        "page": 0,
                        "page_size": 10,
                    },
                    "metadata": MCPResponseMetadata().model_dump(),
                },
                [("https://a.com", {"url": "https://a.com"}), ("https://b.com", {"url": "https://b.com"})],
            ),
            (
                {
                    "result_type": "paged",
                    "result": {
                        "items": [{"url": "https://a.com"}, {"url": "https://b.com"}],
                        "total": 2,
                        "page": 0,
                        "page_size": 10,
                    },
                    "metadata": MCPResponseMetadata(
                        user_feedback=MCPResponseMetadata.UserFeedback(
                            required=True, reason="Multiple records returned ask user to clarify which one they want."
                        )
                    ).model_dump(),
                },
                [("https://a.com", {"url": "https://a.com"}), ("https://b.com", {"url": "https://b.com"})],
            ),
            (
                {
                    "result_type": "multipaged",
                    "result": {
                        "companies": {
                            "result": {
                                "items": [{"url": "https://c.com"}, {"url": "https://d.com"}],
                                "total": 2,
                                "page": 0,
                                "page_size": 10,
                            }
                        },
                        "interactions": {
                            "result": {
                                "items": [{"url": "https://e.com"}],
                                "total": 1,
                                "page": 0,
                                "page_size": 10,
                            }
                        },
                    },
                    "metadata": MCPResponseMetadata().model_dump(),
                },
                [
                    ("https://c.com", {"url": "https://c.com"}),
                    ("https://d.com", {"url": "https://d.com"}),
                    ("https://e.com", {"url": "https://e.com"}),
                ],
            ),
            (
                {
                    "result_type": "composite",
                    "result": [
                        {"url": "https://parent.com", "title": "Parent"},
                        {
                            "interactions": {
                                "result": {
                                    "items": [{"url": "https://f.com"}, {"url": "https://g.com"}],
                                    "total": 2,
                                    "page": 0,
                                    "page_size": 10,
                                }
                            }
                        },
                    ],
                    "metadata": MCPResponseMetadata().model_dump(),
                },
                [
                    ("https://parent.com", {"url": "https://parent.com", "title": "Parent"}),
                    ("https://f.com", {"url": "https://f.com"}),
                    ("https://g.com", {"url": "https://g.com"}),
                ],
            ),
        ],
    )
    def test_documents_metadata(self, data, expected):
        result, metadata = format_mcp_tool_response(json.dumps(data), ChunkCreatorType.datahub)

        documents = result.split("\n\n")
        assert len(documents) == len(expected)
        assert isinstance(metadata, MCPResponseMetadata)
        assert metadata.model_dump() == data["metadata"]

        for doc_xml, (expected_uri, expected_content) in zip(documents, expected):
            assert f"<SourceType>{ChunkCreatorType.datahub}</SourceType>" in doc_xml
            assert f"<Source>{expected_uri}</Source>" in doc_xml
            assert "<page_number></page_number>" in doc_xml
            assert f"<Content>\n{json.dumps(expected_content)}\n\t</Content>" in doc_xml
            assert "</Document>" in doc_xml

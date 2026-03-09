import pytest
import json


from redbox.models.file import ChunkCreatorType
from redbox.api.format import find_first_link_field, extract_links, format_mcp_tool_response


class TestFindFirstLinkField:
    @pytest.mark.parametrize(
        "data",
        [
            {},
            [],
            "string",
            42,
            None,
            {"key": "value"},
            {"url": None},
            [{"no": 1}, {"also_no": 2}],
        ],
    )
    def test_returns_none(self, data):
        assert find_first_link_field(data) is None

    @pytest.mark.parametrize(
        "data, expected",
        [
            ({"url": "https://example.com"}, "https://example.com"),
            ({"level1": {"level2": {"url": "https://deep.com"}}}, "https://deep.com"),
            ([{"no": 1}, {"url": "https://list.com"}], "https://list.com"),
            ({"items": [{"url": "https://item.com"}]}, "https://item.com"),
            ([{"a": [{"b": {"url": "https://nested-list.com"}}]}], "https://nested-list.com"),
            # First URL wins when multiple are present
            ({"url": "https://first.com", "nested": {"url": "https://second.com"}}, "https://first.com"),
            # Non-string url is coerced to str
            ({"url": 12345}, "12345"),
        ],
    )
    def test_finds_url(self, data, expected):
        assert find_first_link_field(data) == expected


class TestExtractLinks:
    @pytest.mark.parametrize(
        "data",
        [
            None,
            {"key": "value"},
            {"name": "no-link"},
            {"total": 2, "results": [{"id": 1}, {"id": 2}]},
            {"total": 0, "results": []},
        ],
    )
    def test_returns_empty(self, data):
        assert extract_links(data) == []

    @pytest.mark.parametrize(
        "data, expected_links",
        [
            (
                {"url": "https://example.com", "name": "test"},
                [("https://example.com", {"url": "https://example.com", "name": "test"})],
            ),
            (
                {"meta": {"url": "https://nested.com"}},
                [("https://nested.com", {"meta": {"url": "https://nested.com"}})],
            ),
        ],
    )
    def test_single_object(self, data, expected_links):
        assert extract_links(data) == expected_links

    @pytest.mark.parametrize(
        "items, expected_pairs",
        [
            (
                [{"url": "https://a.com", "id": 1}, {"url": "https://b.com", "id": 2}],
                [
                    ("https://a.com", {"url": "https://a.com", "id": 1}),
                    ("https://b.com", {"url": "https://b.com", "id": 2}),
                ],
            ),
            (
                [{"id": 1}, {"url": "https://only.com", "id": 2}],
                [("https://only.com", {"url": "https://only.com", "id": 2})],
            ),
            (
                [{"links": {"url": "https://nested-item.com"}}],
                [("https://nested-item.com", {"links": {"url": "https://nested-item.com"}})],
            ),
        ],
    )
    def test_paged_result(self, items, expected_pairs):
        data = {"total": len(items), "results": items}
        assert extract_links(data) == expected_pairs


class TestFormatMCPToolResponse:
    @pytest.mark.parametrize(
        "data",
        [
            {"key": "value"},
            {"total": 0, "results": []},
            {"name": "no-url"},
        ],
    )
    def test_returns_original_when_no_links(self, data):
        payload = json.dumps(data)
        assert format_mcp_tool_response(payload, ChunkCreatorType.datahub) == payload

    @pytest.mark.parametrize(
        "data, expected_uris",
        [
            (
                {"url": "https://example.com", "title": "Example"},
                ["https://example.com"],
            ),
            (
                {"total": 2, "results": [{"url": "https://a.com"}, {"url": "https://b.com"}]},
                ["https://a.com", "https://b.com"],
            ),
        ],
    )
    def test_documents_metadata(self, data, expected_uris):
        result = format_mcp_tool_response(json.dumps(data), ChunkCreatorType.datahub)

        documents = result.split("\n\n")
        assert len(documents) == len(expected_uris)

        for doc_xml, expected_uri in zip(documents, expected_uris):
            assert f"<SourceType>{ChunkCreatorType.datahub}</SourceType>" in doc_xml
            assert f"<Source>{expected_uri}</Source>" in doc_xml
            assert "<page_number></page_number>" in doc_xml
            assert "<Content>" in doc_xml
            assert "</Document>" in doc_xml

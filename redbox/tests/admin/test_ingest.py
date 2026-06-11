from unittest.mock import Mock, patch

from redbox.admin.ingest import _get_resolutions_for_file, _get_duplicate_chunks


@patch("redbox.admin.ingest._es_client")
def test_get_resolutions_for_file_returns_counts(mock_es_client):
    es = Mock()
    mock_es_client.return_value = es

    es.search.return_value = {
        "aggregations": {
            "resolutions": {
                "buckets": [
                    {"key": "largest", "doc_count": 10},
                    {"key": "normal", "doc_count": 10},
                ]
            }
        }
    }

    results = list(
        _get_resolutions_for_file(
            "file.pdf",
            "test-index",
        )
    )

    assert len(results) == 2

    assert results[0].name == "largest"
    assert results[0].count == 10

    assert results[1].name == "normal"
    assert results[1].count == 10


@patch("redbox.admin.ingest._es_client")
def test_missing_resolution_added(mock_es_client):
    es = Mock()
    mock_es_client.return_value = es

    es.search.return_value = {
        "aggregations": {
            "resolutions": {
                "buckets": [
                    {"key": "largest", "doc_count": 5},
                ]
            }
        }
    }

    results = {
        r.name: r.count
        for r in _get_resolutions_for_file(
            "file.pdf",
            "test-index",
        )
    }

    assert results == {
        "largest": 5,
        "normal": 0,
    }


@patch("redbox.admin.ingest._es_client")
def test_csv_expects_tabular_resolution(mock_es_client):
    es = Mock()
    mock_es_client.return_value = es

    es.search.return_value = {"aggregations": {"resolutions": {"buckets": []}}}

    results = {
        r.name: r.count
        for r in _get_resolutions_for_file(
            "data.csv",
            "test-index",
        )
    }

    assert results == {
        "tabular": 0,
    }


@patch("redbox.admin.ingest._es_client")
def test_duplicate_summary(mock_es_client):
    es = Mock()
    mock_es_client.return_value = es

    es.search.return_value = {
        "aggregations": {
            "duplicate_groups": {
                "buckets": [
                    {
                        "key": {
                            "resolution": "largest",
                            "page_number": 1,
                        },
                        "doc_count": 3,
                    },
                    {
                        "key": {
                            "resolution": "largest",
                            "page_number": 2,
                        },
                        "doc_count": 2,
                    },
                ]
            }
        }
    }

    results = {
        r.name: r
        for r in _get_duplicate_chunks(
            "file.pdf",
            "index",
        )
    }

    largest = results["largest"]

    assert largest.affected_pages == 2
    assert largest.total_duplicate_chunks == 5
    assert largest.avg_duplicates_per_page == 2.5

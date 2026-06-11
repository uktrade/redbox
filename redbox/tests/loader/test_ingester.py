from unittest.mock import Mock, patch

from redbox.loader.ingester import remove_duplicate_chunks


@patch("redbox.loader.ingester.env.elasticsearch_client")
def test_remove_duplicate_chunks_deletes_duplicates(
    mock_client,
):
    es = Mock()
    mock_client.return_value = es

    es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_source": {
                        "metadata": {
                            "uri": "file.pdf",
                            "chunk_resolution": "largest",
                            "page_number": 1,
                        }
                    },
                },
                {
                    "_id": "2",
                    "_source": {
                        "metadata": {
                            "uri": "file.pdf",
                            "chunk_resolution": "largest",
                            "page_number": 1,
                        }
                    },
                },
            ]
        }
    }

    remove_duplicate_chunks(
        "file.pdf",
        "index",
    )

    es.delete.assert_called_once_with(
        index="index",
        id="2",
    )

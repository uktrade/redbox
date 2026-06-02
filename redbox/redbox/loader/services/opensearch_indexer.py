import logging
import time

from opensearchpy import OpenSearch, helpers


logger = logging.getLogger(__name__)


class OpenSearchBulkIndexer:
    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        max_retries: int = 5,
        vector_field_name: str = "vector_field",
    ):
        self.client = client
        self.index_name = index_name
        self.max_retries = max_retries
        self.vector_field_name = vector_field_name

    def bulk_index(
        self,
        docs,
        embeddings,
    ):

        actions = []

        for doc, embedding in zip(docs, embeddings):
            actions.append(
                {
                    "_index": self.index_name,
                    "_source": {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        self.vector_field_name: embedding,
                    },
                }
            )

        for attempt in range(self.max_retries):
            try:
                helpers.bulk(
                    self.client,
                    actions,
                )

                logger.info(
                    "Indexed %s documents",
                    len(actions),
                )

                return

            except Exception:
                logger.exception(
                    "Bulk index failed attempt=%s",
                    attempt + 1,
                )

                if attempt == self.max_retries - 1:
                    raise

                time.sleep(2**attempt)

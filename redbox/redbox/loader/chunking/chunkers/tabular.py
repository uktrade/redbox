import logging
from typing import Iterator
from datetime import UTC, datetime

from langchain_core.documents import Document

from redbox.models.chain import GeneratedMetadata
from redbox.transform import bedrock_tokeniser
from redbox.loader.chunking.base import BaseChunker

logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


class TabularDocumentChunker(BaseChunker):
    def tabular_chunks(
        self,
        s3_key: str,
        tabular_elements: list[dict[str, str]],
        generated_metadata: GeneratedMetadata,
        include_schema_metadata: bool,
    ) -> Iterator[Document]:
        created_datetime = datetime.now(UTC)

        for idx, el in enumerate(tabular_elements or []):
            metadata = self._build_metadata(
                index=idx,
                s3_key=s3_key,
                page_number=1,
                created_datetime=created_datetime,
                text=el["text"],
                generated_metadata=generated_metadata,
            )

            if include_schema_metadata:
                metadata = {**metadata, **el.get("metadata", {})}

            yield Document(page_content=el["text"], metadata=metadata)

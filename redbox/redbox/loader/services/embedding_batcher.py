from typing import Iterator

from langchain_core.documents import Document


class EmbeddingBatcher:
    def __init__(
        self,
        embedding_model,
        batch_size: int = 64,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def iter_embedding_batches(
        self,
        docs: Iterator[Document],
    ):

        batch = []

        for doc in docs:
            batch.append(doc)

            if len(batch) >= self.batch_size:
                embeddings = self.embedding_model.embed_documents([d.page_content for d in batch])

                yield batch, embeddings

                batch = []

        if batch:
            embeddings = self.embedding_model.embed_documents([d.page_content for d in batch])

            yield batch, embeddings

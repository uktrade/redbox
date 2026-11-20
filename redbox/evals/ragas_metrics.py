from __future__ import annotations

import os
from typing import Any


class RagasMetricScorer:
    def __init__(self, *, model: str, aws_region: str, embeddings: Any) -> None:
        from ragas.embeddings.base import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        class LangchainEmbeddingAdapter(BaseRagasEmbedding):
            def __init__(self, langchain_embeddings: Any) -> None:
                super().__init__()
                self.langchain_embeddings = langchain_embeddings

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                return self.langchain_embeddings.embed_query(text)

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await self.langchain_embeddings.aembed_query(text)

        os.environ["AWS_REGION_NAME"] = aws_region
        import litellm

        ragas_llm = llm_factory(
            model=f"bedrock/{model}",
            provider="litellm",
            client=litellm.completion,
        )
        ragas_embeddings = LangchainEmbeddingAdapter(embeddings)
        self.metrics = {
            "faithfulness": Faithfulness(llm=ragas_llm),
            "answer_relevance": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            "context_precision": ContextPrecision(llm=ragas_llm),
            "context_recall": ContextRecall(llm=ragas_llm),
        }

    async def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> dict[str, float]:
        inputs = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        }
        scores = {}
        for name, metric in self.metrics.items():
            result = await metric.ascore(**inputs)
            scores[name] = result.value
        return scores

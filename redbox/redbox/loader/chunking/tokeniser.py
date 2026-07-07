import logging
from functools import lru_cache
import json

import boto3
from redbox.models.settings import get_settings

logger = logging.getLogger(__name__)

env = get_settings()

_bedrock_client = boto3.client("bedrock-runtime", region_name=env.aws_region)


@lru_cache(maxsize=10_000)
def titan_tokeniser(text: str) -> int:
    """
    Return the exact input token count Titan Embeddings would use for `text`,
    by reading the x-amzn-bedrock-input-token-count response header.

    This avoids approximation drift from generic char/word-based tokenisers,
    which can undercount dense/punctuation-heavy text (e.g. CSV/tabular data)
    by 20-30%+ relative to what the model actually consumes.
    """
    if not text:
        return 0

    try:
        response = _bedrock_client.invoke_model(
            modelId=env.embedding_backend,  # e.g. "amazon.titan-embed-text-v2:0"
            body=json.dumps({"inputText": text}),
            accept="application/json",
            contentType="application/json",
        )
        token_count = response["ResponseMetadata"]["HTTPHeaders"].get("x-amzn-bedrock-input-token-count")
        if token_count is None:
            logger.warning("Bedrock response missing input-token-count header; falling back to estimate.")
            return _fallback_estimate(text)
        return int(token_count)
    except Exception as e:
        logger.exception("Titan tokeniser call failed; falling back to estimate. %e", str(e))
        return _fallback_estimate(text)


def _fallback_estimate(text: str) -> int:
    """Conservative fallback if the Bedrock call fails"""
    return len(text) // 3  # deliberately pessimistic vs the usual ~4 chars/token rule of thumb


tokeniser = titan_tokeniser

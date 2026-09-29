# Test ingestion logic
from pathlib import Path

import pytest


@pytest.mark.ai
def test_retrieval_corpus_ingested(seeded_corpus: dict) -> None:
    """Verify every PDF in dataset/corpus/ was successfully ingested."""
    pdf_count = len(list((Path(__file__).parent / "dataset" / "corpus").glob("*.pdf")))
    assert pdf_count > 0
    assert len(seeded_corpus) == pdf_count, f"Expected {pdf_count} corpus documents, got {len(seeded_corpus)}."

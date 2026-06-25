from django.db.models import TextChoices


class IngestExtractionStrategy(TextChoices):
    unstructured = "unstructured"
    unstructured_auto = "unstructured_auto"
    unstructured_fast = "unstructured_fast"
    textract_document_analysis = "textract_document_analysis"
    pymupdf = "pymupdf"
    tabular = "tabular"
    unspecified = "unspecified"


class IngestChunkingStrategy(TextChoices):
    overlapping_pages = "overlapping_pages"
    page_by_page = "page_by_page"
    tabular = "tabular"
    unstructured_chunk_by_title = "unstructured_chunk_by_title"
    unspecified = "unspecified"

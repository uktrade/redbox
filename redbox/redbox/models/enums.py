from django.db.models import TextChoices


class IngestExtractionStrategy(TextChoices):
    unstructured_auto = "unstructured_auto"
    unstructured_fast = "unstructured_fast"
    unstructured_pptx = "unstructured_pptx"
    unstructured_docx = "unstructured_docx"
    textract_document_analysis = "textract_document_analysis"
    textract_document_analysis_large = "textract_document_analysis_large"
    pymupdf = "pymupdf"
    tabular = "tabular"
    unspecified = "unspecified"


class IngestChunkingStrategy(TextChoices):
    overlapping_pages = "overlapping_pages"
    page_by_page = "page_by_page"
    tabular = "tabular"
    unstructured_chunk_by_title = "unstructured_chunk_by_title"
    unspecified = "unspecified"

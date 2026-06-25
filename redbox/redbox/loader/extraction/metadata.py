import time
import logging

from pydantic import ValidationError
from langchain_core.prompts import PromptTemplate

from unstructured.documents.elements import Element

from redbox.chains.components import get_chat_llm
from redbox.models.chain import GeneratedMetadata
from redbox.chains.parser import ClaudeParser
from redbox.models.settings import Settings

logger = logging.getLogger(__name__)


class MetadataExtraction:
    """
    Service responsible for generating structured file metadata using a
    combination of heuristic preprocessing and LLM-based extraction.

    Attributes:
        env (Settings):
            Application configuration containing metadata extraction settings.
        llm:
            Chat LLM instance used for metadata inference.

    Methods:
        create_file_metadata(file_name, page_content, original_metadata):
            Uses an LLM pipeline to generate structured metadata from document content,
            optionally enriched with existing metadata.

        get_first_10k_chars(elements):
            Extracts up to 10,000 characters of text from heterogeneous document
            representations (strings, dicts, or unstructured Elements).

        extract(file_name, elements):
            High-level entry point that determines file type, builds input context,
            and returns generated metadata with timing/logging instrumentation.
    """

    def __init__(self, env: Settings):
        self.env = env
        self.llm = get_chat_llm(env.metadata_extraction_llm)

    def create_file_metadata(
        self, file_name: str, page_content: str, original_metadata: dict | None = None
    ) -> GeneratedMetadata:
        """Trim original metadata and invoke LLM chain"""
        if not original_metadata:
            original_metadata = {}

        def trim(obj, max_length=1000):
            if isinstance(obj, dict):
                return {k: trim(v, max_length) for k, v in obj.items()}
            if isinstance(obj, list):
                return [trim(v, max_length) for v in obj]
            if isinstance(obj, str):
                return obj[:max_length]
            return obj

        original_metadata = trim(original_metadata)

        parser = ClaudeParser(pydantic_object=GeneratedMetadata)

        metadata_prompt = PromptTemplate(
            template="".join(self.env.metadata_prompt)
            + "\n\n{format_instructions}\n\n{page_content}\n\n{original_metadata}",
            input_variables=["page_content"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "original_metadata": original_metadata,
            },
        )
        metadata_chain = metadata_prompt | self.llm | parser

        try:
            metadata = metadata_chain.invoke({"page_content": page_content})

            if not metadata.name:
                metadata.name = original_metadata.get("filename") or file_name

            return metadata
        except ValidationError as e:
            logger.warning(e.errors())
            return GeneratedMetadata(name=original_metadata.get("filename") or file_name)

    def get_first_10k_chars(self, elements: list[Element] | list[str] | list[dict]) -> str:
        if all(isinstance(p, str) for p in elements):
            return "".join(p for p in elements)[:10_000]

        if all(isinstance(p, dict) for p in elements):
            pages = [e.get("text", "") for e in elements]
            return "".join(p for p in pages)[:10_000]

        if all(isinstance(p, Element) for p in elements):
            return "".join(el.text for el in elements if getattr(el, "text", None))[:10_000]

        raise TypeError("pages must be either list[str] or list[Element] or list[dict], not mixed")

    def extract(self, file_name: str, elements: list[Element] | list[str] | list[dict]) -> GeneratedMetadata:
        start_time = time.time()
        first_10k_chars = self.get_first_10k_chars(elements=elements)

        # Determine file type for metadata extraction
        file_type = "unknown"
        if file_name.lower().endswith(".docx"):
            file_type = "DOCX"
        elif file_name.lower().endswith(".csv"):
            file_type = "CSV"
        elif file_name.lower().endswith((".xlsx", ".xls")):
            file_type = "Excel"
        elif file_name.lower().endswith(".pdf"):
            file_type = "PDF"

        original_metadata = {"file_type": file_type, "filename": file_name}

        try:
            metadata = self.create_file_metadata(
                file_name=file_name, page_content=first_10k_chars, original_metadata=original_metadata
            )
        except Exception as e:
            logger.info(e)
            metadata = GeneratedMetadata(name=file_name)

        logger.warning(
            "Total metadata extraction for file [%s] took %.2f seconds",
            file_name,
            time.time() - start_time,
        )

        return metadata

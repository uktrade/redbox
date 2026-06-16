import logging
import time

from io import BytesIO
from typing import TYPE_CHECKING
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from redbox.chains.parser import ClaudeParser
from redbox.chains.components import get_chat_llm
from redbox.loader.chunker import LayoutBlock
from redbox.models.chain import GeneratedMetadata
from redbox.models.file import ChunkResolution
from redbox.models.settings import Settings

# from redbox.loader.textract2 import TextractChunkLoader
from redbox.loader.textract.chunker import TextractChunkLoader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


class MetadataLoader:
    """
    Extract metadata from a file using a TextractChunkLoader and LLM.
    Preserves trimming and robust handling from old loader.
    """

    def __init__(self, env: Settings, s3_client: S3Client, file_name: str):
        self.env = env
        self.s3_client = s3_client
        self.llm = get_chat_llm(env.metadata_extraction_llm)
        self.file_name = file_name

    def _get_file_bytes(self, file_name: str) -> BytesIO:
        obj = self.s3_client.get_object(Bucket=self.env.bucket_name, Key=file_name)
        return BytesIO(obj["Body"].read())

    def extract_metadata(
        self, layout_blocks: list[LayoutBlock] | None, tabular_elements: list[dict] | None
    ) -> GeneratedMetadata:
        start_time = time.time()

        loader = TextractChunkLoader(
            chunk_resolution=ChunkResolution.normal,
            min_chunk_size=200,
            max_chunk_size=2000,
            overlap_chars=0,
        )

        # file_bytes = None
        # if self.file_name.lower().endswith(".docx"):
        #     file_bytes = self._get_file_bytes(self.file_name)

        chunks = []

        try:
            for c in loader.lazy_load_from_blocks(
                file_name=self.file_name, layout_blocks=layout_blocks, tabular_elements=tabular_elements
            ):
                chunks.append(c)
        except Exception:
            logger.exception("Lazy loader crashed during metadata extraction")
            raise

        first_10k_chars = "".join(c.page_content for c in chunks)[:10_000]

        # Determine file type for metadata extraction
        file_type = "unknown"
        if self.file_name.lower().endswith(".docx"):
            file_type = "DOCX"
        elif self.file_name.lower().endswith(".csv"):
            file_type = "CSV"
        elif self.file_name.lower().endswith((".xlsx", ".xls")):
            file_type = "Excel"
        elif self.file_name.lower().endswith(".pdf"):
            file_type = "PDF"

        original_metadata = {"file_type": file_type, "filename": self.file_name}

        try:
            metadata = self.create_file_metadata(first_10k_chars, original_metadata=original_metadata)
        except Exception as e:
            logger.info(e)
            metadata = GeneratedMetadata(name=self.file_name)

        logger.info(
            "Total metadata extraction for file [%s] took %.2f seconds",
            self.file_name,
            time.time() - start_time,
        )

        return metadata

    def create_file_metadata(self, page_content: str, original_metadata: dict | None = None) -> GeneratedMetadata:
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
                metadata.name = original_metadata.get("filename") or self.file_name

            return metadata
        except ValidationError as e:
            logger.info(e.errors())
            return GeneratedMetadata(name=original_metadata.get("filename") or self.file_name)

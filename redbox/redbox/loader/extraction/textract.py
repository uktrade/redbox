import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from io import BytesIO

import boto3
import fitz
from botocore.config import Config
from botocore.exceptions import ClientError

from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    Footer,
    Header,
    ListItem,
    NarrativeText,
    Table,
    Text,
    Title,
)

logger = logging.getLogger(__name__)


class TextractJobFailed(RuntimeError):
    pass


class TextractTimeout(TimeoutError):
    pass


@dataclass
class PdfChunk:
    s3_key: str
    start_page: int  # 0-indexed, inclusive - first page of this chunk in the ORIGINAL doc
    end_page: int  # 0-indexed, exclusive - one past the last page of this chunk in the ORIGINAL doc
    overlap_start: int  # 0-indexed - pages [start_page, overlap_start) are overlap-only,
    # i.e. already covered by the previous chunk and should be dropped when merging


class TextractService:
    """
    Service wrapper around AWS Textract for extracting text from documents
    stored in S3.
    """

    def __init__(self, bucket: str, region: str = "eu-west-2"):
        self.bucket = bucket
        self.region = region
        textract_config = Config(
            retries={"mode": "adaptive", "max_attempts": 10},
            connect_timeout=20,
            read_timeout=70,
        )
        self.textract = boto3.client("textract", region_name=region, config=textract_config)
        self.s3 = boto3.client("s3", region_name=region)

        logger.warning("Initialised TextractService (bucket=%s, region=%s)", bucket, region)

    def _is_retryable_textract_error(self, error: Exception) -> bool:
        if not isinstance(error, ClientError):
            return False

        error_code = error.response.get("Error", {}).get("Code", "")
        return error_code in {
            "ProvisionedThroughputExceededException",
            "ThrottlingException",
            "Throttling",
            "RequestLimitExceeded",
        }

    def _retry_textract_request(self, __func, *args, max_attempts: int = 6, base_delay: float = 3.0, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            try:
                return __func(*args, **kwargs)
            except Exception as e:
                if self._is_retryable_textract_error(e) and attempt < max_attempts:
                    sleep_time = base_delay * (2 ** (attempt - 1)) + random.random()
                    logger.warning(
                        "Textract throttled on attempt %s/%s for %s; sleeping %.1fs before retrying",
                        attempt,
                        max_attempts,
                        getattr(__func, "__name__", str(__func)),
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                logger.exception("Textract API error on %s: %s", getattr(__func, "__name__", str(__func)), e)
                raise

    def _split_pdf_to_s3_chunks(
        self,
        file_bytes: bytes,
        key: str,
        pages_per_chunk: int,
        overlap_pages: int,
    ) -> list[PdfChunk]:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = doc.page_count
        chunks: list[PdfChunk] = []

        chunk_index = 0
        cursor = 0  # next un-covered page (0-indexed)

        while cursor < total_pages:
            is_first = chunk_index == 0
            start_page = cursor if is_first else max(cursor - overlap_pages, 0)
            end_page = min(start_page + pages_per_chunk, total_pages)
            overlap_start = cursor  # pages [start_page, overlap_start) are overlap-only

            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            chunk_key = f"{key}.textract-chunks/{chunk_index:04d}.pdf"
            self.s3.put_object(Bucket=self.bucket, Key=chunk_key, Body=chunk_bytes)

            chunks.append(
                PdfChunk(
                    s3_key=chunk_key,
                    start_page=start_page,
                    end_page=end_page,
                    overlap_start=overlap_start,
                )
            )

            logger.warning(
                "Split chunk %d: pages [%d, %d) (overlap-only pages < %d) -> s3://%s/%s",
                chunk_index,
                start_page,
                end_page,
                overlap_start,
                self.bucket,
                chunk_key,
            )

            cursor = end_page
            chunk_index += 1

        doc.close()
        return chunks

    def _run_chunk_document_analysis(
        self,
        chunk: PdfChunk,
        timeout: float | None = None,
    ) -> list[Element]:
        job_id = self.start_document_analysis(chunk.s3_key)

        status, _ = self._wait_for_job(
            job_id=job_id,
            getter=self.textract.get_document_analysis,
            timeout=timeout,
        )

        if status != "SUCCEEDED":
            raise TextractJobFailed(f"document_analysis chunk job {job_id} failed for {chunk.s3_key}")

        return self.fetch_document_analysis_result(job_id)

    def _cleanup_chunk(self, chunk: PdfChunk) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=chunk.s3_key)
        except Exception:
            logger.exception("Failed to clean up Textract chunk s3://%s/%s", self.bucket, chunk.s3_key)

    def document_analysis_large(
        self,
        key: str,
        file_bytes: BytesIO | None = None,
        pages_per_chunk: int = 200,
        overlap_pages: int = 1,
        max_workers: int = 6,
        timeout: float | None = None,
    ) -> list[Element]:
        """
        Split a large PDF into overlapping page-range chunks and run
        Textract `document_analysis` on each chunk concurrently, then merge
        back into a single page-ordered list of Elements with LAYOUT intact
        per chunk.
        """
        if file_bytes is None:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            pdf_bytes = obj["Body"].read()
        else:
            file_bytes.seek(0)
            pdf_bytes = file_bytes.read()

            # temporary key used only for chunk uploads
            key = f"{key}-converted.pdf"

        chunks = self._split_pdf_to_s3_chunks(
            file_bytes=pdf_bytes, key=key, pages_per_chunk=pages_per_chunk, overlap_pages=overlap_pages
        )

        logger.warning(
            "Split s3://%s/%s into %d chunk(s) for parallel Textract analysis", self.bucket, key, len(chunks)
        )

        results: dict[int, list[Element]] = {}

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._run_chunk_document_analysis, chunk, timeout): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
        finally:
            for chunk in chunks:
                self._cleanup_chunk(chunk)

        merged: list[Element] = []

        for idx, chunk in enumerate(chunks):
            elements = results[idx]
            drop_before = (
                chunk.overlap_start - chunk.start_page
            )  # pages to drop, in chunk-local page numbers (0-indexed)

            for element in elements:
                local_page = (element.metadata.page_number or 1) - 1  # chunk-local, 0-indexed

                if local_page < drop_before:
                    # this page is overlap-only - already covered by the previous chunk's results
                    continue

                original_page = chunk.start_page + local_page + 1  # back to 1-indexed, original doc
                element.metadata.page_number = original_page
                merged.append(element)

        logger.warning(
            "Merged %d chunk(s) into %d total Elements for s3://%s/%s", len(chunks), len(merged), self.bucket, key
        )

        return merged

    def start_document_text_detection(self, key: str) -> str:
        response = self._retry_textract_request(
            self.textract.start_document_text_detection,
            DocumentLocation={"S3Object": {"Bucket": self.bucket, "Name": key}},
        )
        job_id = response["JobId"]
        logger.warning("Started 'document_text_detection' Textract job %s for s3://%s/%s", job_id, self.bucket, key)
        return job_id

    def start_document_analysis(self, key: str) -> str:
        response = self._retry_textract_request(
            self.textract.start_document_analysis,
            DocumentLocation={"S3Object": {"Bucket": self.bucket, "Name": key}},
            FeatureTypes=["LAYOUT"],
        )
        job_id = response["JobId"]
        logger.warning("Started 'document_analysis' Textract job %s for s3://%s/%s", job_id, self.bucket, key)
        return job_id

    def check_document_text_detection_status(self, job_id: str) -> str:
        response = self._retry_textract_request(self.textract.get_document_text_detection, JobId=job_id)
        return response["JobStatus"]

    def check_document_analysis_status(self, job_id: str) -> str:
        response = self._retry_textract_request(self.textract.get_document_analysis, JobId=job_id)
        return response["JobStatus"]

    def fetch_document_text_detection_result(self, job_id: str) -> list[str]:
        response = self._retry_textract_request(self.textract.get_document_text_detection, JobId=job_id)
        if response["JobStatus"] != "SUCCEEDED":
            raise TextractJobFailed(f"document_text_detection job {job_id} not in SUCCEEDED state")
        return self._get_textract_results(
            job_id=job_id, getter=self.textract.get_document_text_detection, first_response=response
        )

    def fetch_document_analysis_result(self, job_id: str) -> list[Element]:
        response = self._retry_textract_request(self.textract.get_document_analysis, JobId=job_id)
        if response["JobStatus"] != "SUCCEEDED":
            raise TextractJobFailed(f"document_analysis job {job_id} not in SUCCEEDED state")
        return self._get_textract_results(
            job_id=job_id, getter=self.textract.get_document_analysis, first_response=response, layout=True
        )

    def _wait_for_job(
        self,
        job_id: str,
        getter: Any,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> tuple[str, dict]:
        logger.warning("Waiting for Textract job %s to complete", job_id)

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TextractTimeout(f"Textract job {job_id} exceeded timeout of {timeout}s")

            response = self._retry_textract_request(getter, JobId=job_id)
            status = response["JobStatus"]

            logger.debug(
                "Textract job %s current status: %s",
                job_id,
                status,
            )

            if status in ("SUCCEEDED", "FAILED"):
                logger.warning(
                    "Textract job %s finished with status: %s",
                    job_id,
                    status,
                )
                return status, response

            time.sleep(poll_interval)

    def _get_textract_results(
        self,
        job_id: str,
        getter: Any,
        first_response: dict,
        *,
        layout: bool = False,
    ) -> list[str] | list[Element]:
        logger.warning("Fetching Textract results for job %s", job_id)

        pages: dict[int, list[str]] = {}
        blocks: list[dict] = []

        response = first_response
        api_calls = 0

        while True:
            for block in response.get("Blocks", []):
                if layout:
                    blocks.append(block)
                elif block["BlockType"] == "LINE":
                    page = block.get("Page", 1)
                    pages.setdefault(page, []).append(block["Text"])

            next_token = response.get("NextToken")
            if not next_token:
                break

            response = self._retry_textract_request(getter, JobId=job_id, NextToken=next_token)
            api_calls += 1

        logger.warning("Retrieved Textract results for job %s via %d API calls", job_id, api_calls)

        if not layout:
            return ["\n".join(pages[p]) for p in sorted(pages)]

        return self._layout_blocks_to_elements(blocks)

    def _layout_blocks_to_elements(self, blocks: list[dict]) -> list[Element]:
        lookup = {block["Id"]: block for block in blocks if "Id" in block}
        elements: list[Element] = []

        for block in blocks:
            block_type = block["BlockType"]

            if not block_type.startswith("LAYOUT_"):
                continue

            text = "\n".join(
                lookup[child_id]["Text"]
                for relationship in block.get("Relationships", [])
                if relationship["Type"] == "CHILD"
                for child_id in relationship["Ids"]
                if lookup.get(child_id, {}).get("BlockType") == "LINE"
            ).strip()

            if not text:
                continue

            metadata = ElementMetadata(page_number=block.get("Page"))

            match block_type:
                case "LAYOUT_TITLE" | "LAYOUT_SECTION_HEADER":
                    element = Title(text=text, metadata=metadata)
                case "LAYOUT_HEADER":
                    element = Header(text=text, metadata=metadata)
                case "LAYOUT_FOOTER":
                    element = Footer(text=text, metadata=metadata)
                case "LAYOUT_LIST":
                    element = ListItem(text=text, metadata=metadata)
                case "LAYOUT_TABLE":
                    element = Table(text=text, metadata=metadata)
                case "LAYOUT_TEXT":
                    element = NarrativeText(text=text, metadata=metadata)
                case _:
                    element = Text(text=text, metadata=metadata)

            elements.append(element)

        logger.warning("Converted %d layout blocks into %d Unstructured elements", len(blocks), len(elements))

        return elements

    def document_text_detection(
        self,
        key: str,
        timeout: float | None = None,
    ) -> list[str]:
        logger.warning(
            "Starting Textract 'document_text_detection' extraction directly from S3: s3://%s/%s",
            self.bucket,
            key,
        )

        job_id = self.start_document_text_detection(key)

        status, _ = self._wait_for_job(
            job_id=job_id,
            getter=self.textract.get_document_text_detection,
            timeout=timeout,
        )

        if status != "SUCCEEDED":
            raise TextractJobFailed(f"Textract 'document_text_detection' failed for s3://{self.bucket}/{key}")

        return self.fetch_document_text_detection_result(job_id)

    def document_analysis(
        self,
        key: str,
        file_bytes: BytesIO | None = None,
        timeout: float | None = None,
    ) -> list[Element]:
        temporary_upload = file_bytes is not None
        if temporary_upload:
            logger.warning("Starting Textract 'document_analysis' extraction from temporary pdf...")

            key = f"{key}-converted.pdf"
            file_bytes.seek(0)
            pdf_bytes = file_bytes.read()

            logger.warning(
                "Uploading temporary PDF for Textract analysis: s3://%s/%s",
                self.bucket,
                key,
            )
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=pdf_bytes,
                ContentType="application/pdf",
            )
        else:
            logger.warning(
                "Starting Textract 'document_analysis' extraction directly from S3: s3://%s/%s",
                self.bucket,
                key,
            )

        try:
            job_id = self.start_document_analysis(key)

            status, _ = self._wait_for_job(
                job_id=job_id,
                getter=self.textract.get_document_analysis,
                timeout=timeout,
            )

            if status != "SUCCEEDED":
                raise TextractJobFailed(f"Textract 'document_analysis' failed for s3://{self.bucket}/{key}")

            return self.fetch_document_analysis_result(job_id)
        finally:
            if temporary_upload:
                try:
                    self.s3.delete_object(Bucket=self.bucket, Key=key)
                except Exception:
                    logger.exception(
                        "Failed to delete temporary Textract upload s3://%s/%s",
                        self.bucket,
                        key,
                    )

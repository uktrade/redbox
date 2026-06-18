import logging
import time
from typing import List, Any


import random
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class TextractService:
    def __init__(
        self,
        bucket: str,
        region: str = "eu-west-2",
    ):
        self.bucket = bucket
        self.region = region
        textract_config = Config(
            retries={"mode": "adaptive", "max_attempts": 10},
            connect_timeout=20,
            read_timeout=70,
        )
        self.textract = boto3.client("textract", region_name=region, config=textract_config)

        logger.warning(
            "Initialised TextractService (bucket=%s, region=%s)",
            bucket,
            region,
        )

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

    def _wait_for_job(self, job_id: str, getter: Any) -> tuple[str, dict]:
        logger.warning("Waiting for Textract job %s to complete", job_id)

        while True:
            try:
                response = self._retry_textract_request(getter, JobId=job_id)
                status = response["JobStatus"]

                logger.debug("Textract job %s current status: %s", job_id, status)

                if status in ["SUCCEEDED", "FAILED"]:
                    logger.warning("Textract job %s finished with status: %s", job_id, status)
                    return status, response

                time.sleep(5)

            except Exception as e:
                logger.exception("Error while polling Textract job %s: %s", job_id, e)
                raise

    def _get_textract_results(self, job_id: str, getter: Any, first_response: dict) -> List[str]:
        logger.warning("Fetching Textract results for job %s", job_id)

        pages: dict[int, List[str]] = {}
        next_token = None
        api_calls = 0
        response = first_response

        while True:
            try:
                for block in response.get("Blocks", []):
                    if block["BlockType"] == "LINE":
                        page = block.get("Page", 1)
                        pages.setdefault(page, []).append(block["Text"])

                next_token = response.get("NextToken")
                if not next_token:
                    break

                response = self._retry_textract_request(getter, JobId=job_id, NextToken=next_token)
                api_calls += 1

            except Exception as e:
                logger.exception("Error retrieving Textract results for job %s: %s", job_id, e)
                raise

        logger.warning(
            "Retrieved Textract results for job %s: %d pages via %d API calls",
            job_id,
            len(pages),
            api_calls,
        )

        return ["\n".join(pages[p]) for p in sorted(pages)]

    def document_text_detection(self, key: str) -> list[str]:
        logger.warning(
            "Starting Textract 'document_text_detection' extraction directly from S3: s3://%s/%s", self.bucket, key
        )

        try:
            response = self._retry_textract_request(
                self.textract.start_document_text_detection,
                DocumentLocation={
                    "S3Object": {
                        "Bucket": self.bucket,
                        "Name": key,
                    }
                },
            )

            job_id = response["JobId"]
            logger.warning("Started 'document_text_detection' Textract job %s for s3://%s/%s", job_id, self.bucket, key)

            status, first_response = self._wait_for_job(job_id=job_id, getter=self.textract.get_document_text_detection)

            if status != "SUCCEEDED":
                logger.error(
                    "Textract 'document_text_detection' job %s failed for s3://%s/%s", job_id, self.bucket, key
                )
                raise RuntimeError(f"Textract 'document_text_detection' failed for s3://{self.bucket}/{key}")

            return self._get_textract_results(
                job_id=job_id, getter=self.textract.get_document_text_detection, first_response=first_response
            )

        except Exception as e:
            logger.exception(
                "Textract 'document_text_detection' extraction failed for s3://%s/%s: %s", self.bucket, key, e
            )
            raise

    def document_analysis(self, key: str) -> list[str]:
        logger.warning(
            "Starting Textract 'document_analysis' extraction directly from S3: s3://%s/%s", self.bucket, key
        )

        try:
            response = self._retry_textract_request(
                self.textract.start_document_analysis,
                DocumentLocation={
                    "S3Object": {
                        "Bucket": self.bucket,
                        "Name": key,
                    }
                },
                FeatureTypes=["LAYOUT"],
            )

            job_id = response["JobId"]
            logger.warning("Started 'document_analysis' Textract job %s for s3://%s/%s", job_id, self.bucket, key)

            status, first_response = self._wait_for_job(job_id=job_id, getter=self.textract.get_document_analysis)

            if status != "SUCCEEDED":
                logger.error("Textract 'document_analysis' job %s failed for s3://%s/%s", job_id, self.bucket, key)
                raise RuntimeError(f"Textract 'document_analysis' failed for s3://{self.bucket}/{key}")

            return self._get_textract_results(
                job_id=job_id, getter=self.textract.get_document_analysis, first_response=first_response
            )

        except Exception as e:
            logger.exception("Textract 'document_analysis' extraction failed for s3://%s/%s: %s", self.bucket, key, e)
            raise

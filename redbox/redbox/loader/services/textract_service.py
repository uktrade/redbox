import json
import logging
from typing import Iterator

import boto3
from botocore.config import Config
import time


logger = logging.getLogger(__name__)

AWS_CONFIG = Config(
    retries={
        "max_attempts": 15,
        "mode": "adaptive",
    },
    max_pool_connections=50,
)


class TextractService:
    def __init__(self, region_name: str = "eu-west-2"):
        self.textract = boto3.client(
            "textract",
            region_name=region_name,
            config=AWS_CONFIG,
        )

        self.s3 = boto3.client(
            "s3",
            region_name=region_name,
            config=AWS_CONFIG,
        )

    def submit_job(
        self,
        bucket: str,
        key: str,
        output_bucket: str,
        output_prefix: str,
        sns_topic_arn: str | None = None,
        role_arn: str | None = None,
    ) -> str:

        kwargs = {
            "DocumentLocation": {
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key,
                }
            },
            "OutputConfig": {
                "S3Bucket": output_bucket,
                "S3Prefix": output_prefix,
            },
        }

        if sns_topic_arn and role_arn:
            kwargs["NotificationChannel"] = {
                "SNSTopicArn": sns_topic_arn,
                "RoleArn": role_arn,
            }

        response = self.textract.start_document_text_detection(**kwargs)

        job_id = response["JobId"]

        logger.info("Started Textract job %s", job_id)

        return job_id

    def get_job_status(self, job_id: str) -> str:
        response = self.textract.get_document_text_detection(JobId=job_id)
        return response["JobStatus"]

    def iter_output_pages(
        self,
        output_bucket: str,
        output_prefix: str,
    ) -> Iterator[tuple[int, str]]:

        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=output_bucket,
            Prefix=output_prefix,
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                if not key.endswith(".json"):
                    continue

                response = self.s3.get_object(
                    Bucket=output_bucket,
                    Key=key,
                )

                data = json.loads(response["Body"].read())

                pages: dict[int, list[str]] = {}

                for block in data.get("Blocks", []):
                    if block["BlockType"] != "LINE":
                        continue

                    page_num = block.get("Page", 1)

                    pages.setdefault(page_num, []).append(block["Text"])

                for page_num in sorted(pages):
                    yield page_num, "\n".join(pages[page_num])

    def run_and_wait(self, bucket: str, key: str, output_bucket: str, output_prefix: str) -> str:
        job_id = self.submit_job(
            bucket=bucket,
            key=key,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
        )

        # poll status
        while True:
            status = self.get_job_status(job_id)

            if status in ("SUCCEEDED", "FAILED"):
                break

            time.sleep(5)

        if status != "SUCCEEDED":
            raise RuntimeError(f"Textract failed for {key}: {status}")

        return job_id

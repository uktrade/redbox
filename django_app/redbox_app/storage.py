import logging
import tempfile
import uuid
from pathlib import Path

import boto3
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import UploadedFile
from django_chunk_upload_handlers.s3 import S3FileUploadHandler

logger = logging.getLogger(__name__)


class CustomS3FileUploadHandler(S3FileUploadHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_client = boto3.client("s3")
        self.bucket = settings.BUCKET_NAME
        self.temp_key = None
        self.original_file_name = None

    def new_file(self, field_name, file_name, content_type, content_length, charset=None, content_type_extra=None):
        self.original_file_name = f"{self.request.user.email}/{file_name}"
        self.temp_key = f"tmp/chunk_upload_{uuid.uuid4()}"
        logger.info("Starting upload for file: %s, temp_key: %s", self.original_file_name, self.temp_key)
        return super().new_file(field_name, file_name, content_type, content_length, charset, content_type_extra)

    def get_s3_key(self, _upload_id, _chunk_number):
        logger.info("Getting S3 key for file: %s", self.original_file_name)
        return self.original_file_name

    def receive_data_chunk(self, raw_data, _start):
        logger.info("Uploading chunk to temporary key: %s", self.temp_key)

        with tempfile.NamedTemporaryFile(delete=False, prefix="chunk_", suffix=".tmp") as temp_file:
            temp_file.write(raw_data)
            temp_file_path = temp_file.name

        self.s3_client.upload_file(
            temp_file_path,
            self.bucket,
            self.temp_key,
            ExtraArgs={"ContentType": self.content_type},
        )
        Path(temp_file_path).unlink()

        return raw_data

    def file_complete(self, file_size):
        logger.info("File upload complete, size: %s, saving to: %s", file_size, self.original_file_name)

        logger.info("Copying from %s to %s", self.temp_key, self.original_file_name)
        self.s3_client.copy_object(
            Bucket=self.bucket,
            CopySource=f"{self.bucket}/{self.temp_key}",
            Key=self.original_file_name,
            ContentType=self.content_type,
        )

        logger.info("Deleting temporary file: %s", self.temp_key)
        self.s3_client.delete_object(Bucket=self.bucket, Key=self.temp_key)

        s3_response = self.s3_client.get_object(Bucket=self.bucket, Key=self.original_file_name)
        file_content = s3_response["Body"].read()

        return UploadedFile(
            file=ContentFile(file_content, name=self.original_file_name),
            name=self.original_file_name,
            content_type=self.content_type,
            size=file_size,
        )

    def upload_complete(self):
        logger.info("Upload completed for file: %s", self.original_file_name)
        return super().upload_complete()

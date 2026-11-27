from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError

from redbox.models.settings import Settings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


@pytest.fixture(scope="session")
def env() -> Settings:
    return Settings(django_secret_key="", postgres_password="")


@pytest.fixture(scope="session")
def s3_client(env: Settings) -> S3Client:
    _client = env.s3_client()
    try:
        _client.create_bucket(
            Bucket=env.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": env.aws_region},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise

    return _client

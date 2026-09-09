import pytest

from redbox.models.settings import Settings


class TestSettings:
    def test_s3_client_returns_minio(self):
        settings = Settings(object_store="minio", aws_s3_endpoint_url="http://localhost:11234")
        assert settings.s3_client().meta.endpoint_url == "http://localhost:11234"

    def test_s3_client_raises_error_for_unknown_object_store(self):
        settings = Settings(object_store="unknown")
        with pytest.raises(NotImplementedError):
            settings.s3_client()

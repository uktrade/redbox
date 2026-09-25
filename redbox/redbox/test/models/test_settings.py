import pytest

from redbox.models.settings import Settings


class TestSettings:
    def test_s3_client_raises_error_for_unknown_object_store(self):
        settings = Settings(object_store="unknown")
        with pytest.raises(NotImplementedError):
            settings.s3_client()

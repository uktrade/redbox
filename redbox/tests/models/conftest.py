import pytest


@pytest.fixture(scope="session", autouse=True)
def create_index():
    yield

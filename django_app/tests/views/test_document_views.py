import json
import logging
import uuid
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import File

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_upload_view(alice, client, file_pdf_path: Path, s3_client):
    """
    Given that the object store does not have a file with our test file in it
    When we POST our test file to /upload/
    We Expect to see this file in the object store
    """
    file_name = f"{alice.email}/{file_pdf_path.name}"

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)

    client.force_login(alice)

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert file_exists(s3_client, file_name)
        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"


@pytest.mark.django_db()
def test_document_upload_status(client, alice, file_pdf_path: Path, s3_client):
    file_name = f"{alice}/{file_pdf_path.name}"

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)
    client.force_login(alice)
    previous_count = count_s3_objects(s3_client)

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"
        assert count_s3_objects(s3_client) == previous_count + 1
        uploaded_file = File.objects.filter(user=alice).order_by("-created_at")[0]
        assert uploaded_file.status == File.Status.processing


@pytest.mark.django_db()
def test_upload_view_duplicate_files(alice, bob, client, file_pdf_path: Path, s3_client):
    # delete all alice's files
    for key in s3_client.list_objects(Bucket=settings.BUCKET_NAME, Prefix=alice.email).get("Contents", []):
        s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=key["Key"])

    # delete all bob's files
    for key in s3_client.list_objects(Bucket=settings.BUCKET_NAME, Prefix=bob.email).get("Contents", []):
        s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=key["Key"])

    previous_count = count_s3_objects(s3_client)

    def upload_file():
        with file_pdf_path.open("rb") as f:
            client.post("/upload/", {"uploadDocs": f})
            response = client.post("/upload/", {"uploadDocs": f})

            assert response.status_code == HTTPStatus.FOUND
            assert response.url == "/documents/"

            return File.objects.order_by("-created_at")[0]

    client.force_login(alice)
    alices_file = upload_file()

    assert count_s3_objects(s3_client) == previous_count + 1  # new file added
    assert alices_file.unique_name.startswith(alice.email)

    client.force_login(bob)
    bobs_file = upload_file()

    assert count_s3_objects(s3_client) == previous_count + 2  # new file added
    assert bobs_file.unique_name.startswith(bob.email)

    bobs_new_file = upload_file()

    assert count_s3_objects(s3_client) == previous_count + 2  # no change, duplicate file
    assert bobs_new_file.unique_name == bobs_file.unique_name


@pytest.mark.django_db()
def test_upload_view_bad_data(alice, client, file_py_path: Path, s3_client):
    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.OK
        assert "File type .py not supported" in str(response.content)
        assert count_s3_objects(s3_client) == previous_count


@pytest.mark.django_db()
def test_upload_view_no_file(alice, client):
    client.force_login(alice)

    response = client.post("/upload/")

    assert response.status_code == HTTPStatus.OK
    assert "No document selected" in str(response.content)


@pytest.mark.django_db()
def test_remove_doc_view(client: Client, alice: User, file_pdf_path: Path, s3_client: Client):
    file_name = f"{alice.email}/{file_pdf_path.name}"

    client.force_login(alice)
    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    previous_count = count_s3_objects(s3_client)

    with file_pdf_path.open("rb") as f:
        # create file before testing deletion
        client.post("/upload/", {"uploadDocs": f})
        assert file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count + 1

        new_file = File.objects.filter(user=alice).order_by("-created_at")[0]

        client.post(f"/remove-doc/{new_file.id}", {"doc_id": new_file.id})
        assert not file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count
        assert File.objects.get(id=new_file.id).status == File.Status.deleted


@pytest.mark.django_db()
def test_remove_nonexistent_doc(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("remove-doc", kwargs={"doc_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db()
def test_file_status_api_view_nonexistent_file(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    response = client.get("/file-status/", {"id": nonexistent_uuid})

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


def count_s3_objects(s3_client) -> int:
    paginator = s3_client.get_paginator("list_objects")
    return sum(len(result.get("Contents", [])) for result in paginator.paginate(Bucket=settings.BUCKET_NAME) if result)


def file_exists(s3_client, file_name) -> bool:
    """
    if the file key exists return True otherwise False
    """
    try:
        s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))
    except ClientError as client_error:
        if client_error.response["Error"]["Code"] == "NoSuchKey":
            return False
        raise
    else:
        return True


@pytest.mark.django_db()
def test_upload_document_endpoint_success(alice, client, file_pdf_path: Path):
    """
    Test the JSON API for a successful document upload.
    """
    client.force_login(alice)
    with file_pdf_path.open("rb") as f:
        response = client.post("/upload-document/", {"file": f})

    assert response.status_code == HTTPStatus.OK
    response_data = json.loads(response.content)

    assert "file_id" in response_data
    assert "file_name" in response_data
    assert response_data["status"] == File.Status.processing

    file_id = uuid.UUID(response_data["file_id"])
    file = File.objects.get(id=file_id)
    assert file.file_name == file_pdf_path.name
    assert file.user == alice


@pytest.mark.django_db()
def test_upload_document_endpoint_no_file(alice, client):
    """
    Test the JSON API for document upload when no file is provided.
    """
    client.force_login(alice)
    response = client.post("/upload-document/")

    assert response.status_code == HTTPStatus.OK
    response_data = json.loads(response.content)

    assert "errors" in response_data
    assert "No document selected" in response_data["errors"]


@pytest.mark.django_db()
def test_upload_document_endpoint_invalid_file(alice, client, file_py_path: Path):
    """
    Test the JSON API for document upload with an invalid file type.
    """
    client.force_login(alice)
    with file_py_path.open("rb") as f:
        response = client.post("/upload-document/", {"file": f})

    response_data = json.loads(response.content)
    assert "errors" in response_data
    assert "File type .py not supported" in response_data["errors"]


@pytest.mark.django_db()
@patch("redbox_app.documents.views.document_service")
def test_upload_document_doc_conversion(mock_service, alice, client, tmp_path):
    """
    Test that .doc files are converted to .docx before processing.
    """
    client.force_login(alice)

    # Create a mock .doc file
    doc_file = tmp_path / "test.doc"
    doc_file.write_text("test content")

    # Configure mocks
    mock_service.is_doc_file.return_value = True
    mock_service.validate_uploaded_file.return_value = None
    mock_service.is_utf8_compatible.return_value = True
    mock_service.convert_doc_to_docx.return_value = doc_file.open("rb")

    mock_file = MagicMock()
    mock_file.id = uuid.uuid4()
    mock_file.status = File.Status.processing
    mock_file.file_name = "test.docx"
    mock_service.ingest_file.return_value = mock_file

    with doc_file.open("rb") as f:
        response = client.post("/upload-document/", {"file": f})

    # Verify conversion and ingestion
    mock_service.is_doc_file.assert_called_once()
    mock_service.convert_doc_to_docx.assert_called_once()
    mock_service.ingest_file.assert_called_once()

    response_data = json.loads(response.content)
    assert "file_id" in response_data
    assert response_data["file_name"] == "test.docx"


@pytest.mark.django_db()
@patch("redbox_app.documents.views.document_service")
def test_upload_document_utf8_conversion(mock_service, alice, client, tmp_path):
    """
    Test that non-UTF8 files are converted to UTF8 before processing.
    """
    client.force_login(alice)

    # Create a mock text file
    file = tmp_path / "test.txt"
    file.write_text("test content")

    # Configure mocks
    mock_service.is_doc_file.return_value = False
    mock_service.validate_uploaded_file.return_value = None
    mock_service.is_utf8_compatible.return_value = False
    mock_service.convert_to_utf8.return_value = file.open("rb")

    mock_file = MagicMock()
    mock_file.id = uuid.uuid4()
    mock_file.status = File.Status.processing
    mock_file.file_name = "test.txt"
    mock_service.ingest_file.return_value = mock_file

    with file.open("rb") as f:
        response = client.post("/upload-document/", {"file": f})

    mock_service.is_utf8_compatible.assert_called_once()
    mock_service.convert_to_utf8.assert_called_once()
    mock_service.ingest_file.assert_called_once()

    response_data = json.loads(response.content)
    assert "file_id" in response_data


@pytest.mark.django_db()
@patch("redbox_app.documents.views.document_service")
def test_upload_document_ingest_errors(mock_service, alice, client, tmp_path):
    """
    Test handling of ingest errors during document upload.
    """
    client.force_login(alice)

    file = tmp_path / "test.txt"
    file.write_text("test content")

    mock_service.validate_uploaded_file.return_value = None
    mock_service.is_doc_file.return_value = False
    mock_service.is_utf8_compatible.return_value = True

    mock_file = MagicMock()
    mock_file.id = uuid.uuid4()
    mock_file.status = File.Status.error_processing
    mock_file.file_name = "test.txt"
    mock_service.ingest_file.return_value = (["Error processing document"], mock_file)

    with file.open("rb") as f:
        response = client.post("/upload-document/", {"file": f})

    response_data = json.loads(response.content)
    assert "ingest_errors" in response_data
    assert response_data["ingest_errors"] == File.Status.error_processing


@pytest.mark.django_db()
def test_remove_doc_view_get(alice, client):
    """
    Test the remove document view GET request.
    """
    file = File.objects.create(
        user=alice, file_name="test.pdf", unique_name=f"{alice.email}/test.pdf", status=File.Status.ready
    )

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": file.id})
    response = client.get(url)

    assert response.status_code == HTTPStatus.OK
    assert "test.pdf" in str(response.content)
    assert str(file.id) in str(response.content)


@pytest.mark.django_db()
def test_remove_doc_view_post(alice, client, mocker):
    """
    Test the remove document view POST request for document deletion.
    """
    file = File.objects.create(
        user=alice, file_name="test.pdf", unique_name=f"{alice.email}/test.pdf", status=File.Status.ready
    )

    mocker.patch.object(File, "delete_from_elastic")
    mocker.patch.object(File, "delete_from_s3")

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": file.id})
    response = client.post(url, {"doc_id": file.id})

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/documents/"

    file.refresh_from_db()
    assert file.status == File.Status.deleted

    File.delete_from_elastic.assert_called_once()
    File.delete_from_s3.assert_called_once()


@pytest.mark.django_db()
def test_remove_doc_view_error_handling(alice, client, mocker):
    """
    Test error handling in the remove document view.
    """
    file = File.objects.create(
        user=alice, file_name="test.pdf", unique_name=f"{alice.email}/test.pdf", status=File.Status.ready
    )

    mocker.patch.object(File, "delete_from_elastic", side_effect=Exception("Test error"))
    mocker.patch.object(File, "delete_from_s3")

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": file.id})
    response = client.post(url, {"doc_id": file.id})

    assert response.status_code == HTTPStatus.FOUND

    file.refresh_from_db()
    assert file.status == File.Status.errored


@pytest.mark.django_db()
def test_remove_all_docs_view_get(alice, client):
    """
    Test the remove all documents view GET request.
    """
    client.force_login(alice)
    url = reverse("remove-all-docs")
    response = client.get(url)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db()
def test_remove_all_docs_view_post(alice, client, mocker):
    """
    Test the remove all documents view POST request for bulk deletion.
    """
    file1 = File.objects.create(
        user=alice, file_name="test1.pdf", unique_name=f"{alice.email}/test1.pdf", status=File.Status.ready
    )
    file2 = File.objects.create(
        user=alice, file_name="test2.pdf", unique_name=f"{alice.email}/test2.pdf", status=File.Status.ready
    )

    mocker.patch.object(File, "delete_from_elastic")
    mocker.patch.object(File, "delete_from_s3")

    client.force_login(alice)
    url = reverse("remove-all-docs")
    response = client.post(url)

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/documents/"

    file1.refresh_from_db()
    file2.refresh_from_db()
    assert file1.status == File.Status.deleted
    assert file2.status == File.Status.deleted

    assert File.delete_from_elastic.call_count == 2
    assert File.delete_from_s3.call_count == 2

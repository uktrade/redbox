import logging
import uuid
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory
from django.urls import reverse

from redbox_app.redbox_core.models import Chat, ChatLLMBackend, File, FileSkill
from redbox_app.redbox_core.services import documents as document_service
from redbox_app.redbox_core.services import url as url_service
from redbox_app.redbox_core.views.document_views import delete_document

User = get_user_model()

logger = logging.getLogger(__name__)


@pytest.mark.django_db
def test_upload_view(alice, client, file_pdf_path: Path, s3_client, remove_file_from_bucket):
    """
    Given that the object store does not have a file with our test file in it
    When we POST our test file to /upload/
    We Expect to see this file in the object store
    """
    file_name = f"{alice.email}/{file_pdf_path.name.rstrip(file_pdf_path.name[-4:])}"
    remove_file_from_bucket(file_name)

    assert not file_exists(s3_client, file_name)

    client.force_login(alice)

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert file_exists(s3_client, file_name)
        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"


@pytest.mark.django_db
def test_document_upload_status(client, alice, file_pdf_path: Path, s3_client, remove_file_from_bucket):
    file_name = f"{alice}/{file_pdf_path.name}"
    remove_file_from_bucket(file_name)

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


@pytest.mark.django_db
def test_upload_view_bad_data(alice, client, file_py_path: Path, s3_client):
    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.OK
        assert "File type .py not supported" in str(response.content)
        assert count_s3_objects(s3_client) == previous_count


@pytest.mark.django_db
def test_upload_view_no_file(alice, client):
    client.force_login(alice)

    response = client.post("/upload/")

    assert response.status_code == HTTPStatus.OK
    assert "No document selected" in str(response.content)


@pytest.mark.django_db
def test_remove_doc_view(client: Client, alice: User, file_pdf_path: Path, s3_client: Client, remove_file_from_bucket):
    file_name = f"{alice.email}/{file_pdf_path.name.rstrip(file_pdf_path.name[-4:])}"

    client.force_login(alice)
    remove_file_from_bucket(file_name)

    previous_count = count_s3_objects(s3_client)

    with file_pdf_path.open("rb") as f:
        # create file before testing deletion
        client.post("/upload/", {"uploadDocs": f})
        assert file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count + 1

        new_file = File.objects.filter(user=alice).order_by("-created_at")[0]

        client.post(f"/remove-doc/{new_file.id!s}", {"doc_id": str(new_file.id)})
        assert not file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count
        assert File.objects.get(id=str(new_file.id)).status == File.Status.deleted


@pytest.mark.django_db
def test_remove_nonexistent_doc(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("remove-doc", kwargs={"doc_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db
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
    If any file key starts with the given file_name prefix, return True, otherwise False
    """
    prefix = file_name.replace(" ", "_")
    try:
        response = s3_client.list_objects_v2(Bucket=settings.BUCKET_NAME, Prefix=prefix)
    except ClientError as client_error:
        if client_error.response["Error"]["Code"] in ["NoSuchBucket", "AccessDenied"]:
            return False
        raise
    else:
        # Check for actual objects (handles empty responses correctly)
        return bool(response.get("Contents", []))


@pytest.mark.django_db
def test_upload_document_endpoint_invalid_file(alice, client, file_py_path: Path):
    """
    Test document upload with an invalid file type.
    """
    client.force_login(alice)
    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

    assert response.status_code == HTTPStatus.OK

    assert "File type .py not supported" in str(response.content)


@pytest.mark.django_db
def test_upload_document_endpoint_multiple_files(alice, client, file_pdf_path: Path, file_py_path: Path):
    """
    Test the document upload with multiple files, one valid and one invalid.
    """
    client.force_login(alice)
    with file_pdf_path.open("rb") as pdf, file_py_path.open("rb") as py:
        response = client.post("/upload/", {"uploadDocs": [pdf, py]})

    assert response.status_code == HTTPStatus.OK

    assert "File type .py not supported" in str(response.content)


@pytest.mark.django_db
def test_upload_document_endpoint_unauthenticated(client, file_pdf_path: Path):
    """
    Test the document upload when user is not authenticated.
    """
    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

    # Should redirect to login or return 403
    assert response.status_code in (HTTPStatus.FOUND, HTTPStatus.FORBIDDEN)


@pytest.mark.django_db
def test_upload_document_endpoint_empty_file(alice, client, tmp_path):
    """
    Test the document upload with an empty file.
    """
    client.force_login(alice)
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with empty_file.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

    assert response.status_code == HTTPStatus.FOUND


@pytest.mark.django_db
@patch("redbox_app.redbox_core.views.document_views")
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
    mock_file.status = File.Status.errored
    mock_file.original_file_name = "test.txt"
    mock_service.ingest_file.return_value = (["Error processing document"], mock_file)


@pytest.mark.django_db
def test_remove_doc_view_get(alice, client):
    """
    Test the remove document view GET request.
    """
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": str(file.id)})
    response = client.get(url)

    assert response.status_code == HTTPStatus.OK
    assert "test.pdf" in str(response.content)
    assert str(file.id) in str(response.content)


@pytest.mark.django_db
def test_remove_doc_view_post(alice, client, mocker):
    """
    Test the remove document view POST request for document deletion.
    """
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    mocker.patch.object(File, "delete_from_elastic_and_s3")

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": str(file.id)})
    response = client.post(url, {"doc_id": str(file.id)})

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/documents/"

    file.refresh_from_db()
    assert file.status == File.Status.deleted

    File.delete_from_elastic_and_s3.assert_called_once()


@pytest.mark.django_db
def test_remove_doc_view_error_handling(alice, client, mocker):
    """
    Test error handling in the remove document view.
    """
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    mocker.patch.object(File, "delete_from_elastic_and_s3", side_effect=Exception("Test error"))

    client.force_login(alice)
    url = reverse("remove-doc", kwargs={"doc_id": str(file.id)})
    response = client.post(url, {"doc_id": str(file.id)})

    assert response.status_code == HTTPStatus.FOUND

    file.refresh_from_db()
    assert file.status == File.Status.errored


@pytest.mark.django_db
def test_remove_all_docs_view_get(alice, client):
    """
    Test the remove all documents view GET request.
    """
    client.force_login(alice)
    url = reverse("remove-all-docs")
    response = client.get(url)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_remove_all_docs_view_post(alice, client, mocker):
    """
    Test the remove all documents view POST request for bulk deletion.
    """
    file1 = File.objects.create(user=alice, original_file_name="test1.pdf", status=File.Status.complete)
    file2 = File.objects.create(user=alice, original_file_name="test2.pdf", status=File.Status.complete)

    mocker.patch.object(File, "delete_from_elastic_and_s3")

    client.force_login(alice)
    url = reverse("remove-all-docs")
    response = client.post(url)

    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/documents/"

    file1.refresh_from_db()
    file2.refresh_from_db()
    assert file1.status == File.Status.deleted
    assert file2.status == File.Status.deleted

    assert File.delete_from_elastic_and_s3.call_count == 2


# new tests
@pytest.mark.django_db
def test_document_view_get(alice, client):
    """
    Test the DocumentView GET request.
    """
    # Create some files for testing
    completed_file = File.objects.create(user=alice, original_file_name="completed.pdf", status=File.Status.complete)
    processing_file = File.objects.create(
        user=alice, original_file_name="processing.pdf", status=File.Status.processing
    )

    # Login and request the documents page
    client.force_login(alice)
    response = client.get("/documents/")

    # Verify the response
    assert response.status_code == HTTPStatus.OK
    content = str(response.content)
    assert completed_file.original_file_name in content
    assert processing_file.original_file_name in content

    # Check session handling for ingest errors
    client.session["ingest_errors"] = ["Test error"]
    response = client.get("/documents/")
    assert response.status_code == HTTPStatus.OK
    assert client.session.get("ingest_errors") == []


@pytest.mark.django_db
def test_documents_title_view(alice, client):
    """
    Test updating a document title via DocumentsTitleView.
    """
    # Create a file
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="original.pdf", status=File.Status.complete)

    # Login and make the request
    client.force_login(alice)
    url = reverse("document-titles", kwargs={"doc_id": str(file.id)})
    response = client.post(url, data='{"value": "updated.pdf"}', content_type="application/json")

    # Verify the response and changes
    assert response.status_code == HTTPStatus.NO_CONTENT
    file.refresh_from_db()
    assert file.original_file_name == "updated.pdf"


@pytest.mark.django_db
def test_your_documents_view(alice, client):
    """
    Test the YourDocuments view functionality.
    """
    # Create a file
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    # Login and make the request
    client.force_login(alice)
    response = client.get("/documents/")

    # Verify the response
    assert response.status_code == HTTPStatus.OK
    assert file.original_file_name in str(response.content)


@pytest.mark.django_db
def test_file_status_api_view(alice, client):
    """
    Test the file status API view.
    """
    # Create a file
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.processing)

    # Login and make the request
    client.force_login(alice)
    response = client.get("/file-status/", {"id": str(file.id)})

    # Verify the response
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert data["status"] == File.Status.processing.capitalize()


@pytest.mark.django_db
def test_delete_document_endpoint(alice, client, mocker):
    """
    Test the delete_document endpoint.
    """
    # Create a file
    file_id = uuid.uuid4()

    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    mocker.patch.object(File, "delete_from_elastic_and_s3")

    client.force_login(alice)
    url = reverse("delete-document", kwargs={"doc_id": str(file.id)})
    response = client.post(url, {"doc_id": str(file.id)})

    assert response.status_code == HTTPStatus.OK
    assert File.objects.get(id=file_id).status == File.Status.deleted
    assert File.delete_from_elastic_and_s3.called


@pytest.mark.django_db
def test_delete_document_with_chat(alice, client, mocker):
    """
    Test the delete_document endpoint with active chat session.
    """
    file_id = uuid.uuid4()
    # Create a file
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    # Mock methods
    mocker.patch.object(File, "delete_from_elastic_and_s3")

    # Mock chat_service.get_context to return simple context
    mocker.patch(
        "redbox_app.redbox_core.services.chats.get_context",
        return_value={"files": [], "urls": {"upload_url": url_service.get_upload_url()}},
    )

    chat_llm_backend = ChatLLMBackend.objects.get(name="anthropic.claude-3-7-sonnet-20250219-v1:0")

    context = {
        "chat_id": str(uuid.uuid4()),
        "messages": [],
        "chats": [],
        "current_chat": None,
        "streaming": {"endpoint": "/ws/chat/"},
        "contact_email": "support@example.com",
        "completed_files": [],
        "processing_files": [],
        "chat_title_length": 100,
        "llm_options": [
            {
                "name": str(chat_llm_backend),
                "default": chat_llm_backend.is_default,
                "selected": True,
                "id": chat_llm_backend.id,
            }
        ],
        "redbox_api_key": "mock-api-key",  # pragma: allowlist secret
        "enable_dictation_flag_is_active": False,
        "csrf_token": "mock-csrf-token",
        "request": mocker.MagicMock(),
    }

    client.force_login(alice)
    url = reverse("delete-document", kwargs={"doc_id": str(file.id)})
    session_id = context["chat_id"]

    response = client.post(
        url,
        {"doc_id": str(file.id), "session-id": session_id, "file_selected": "False"},
    )
    assert response.status_code == HTTPStatus.OK

    response = client.post(
        url,
        {
            "doc_id": str(file.id),
            "session-id": session_id,
            "file_selected": "True",
            "streaming": {"endpoint": "/ws/chat/"},
        },
    )
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db
def test_delete_document_error_handling(alice, client, mocker):
    """
    Test error handling in the delete_document endpoint.
    """
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    mocker.patch.object(File, "delete_from_elastic_and_s3", side_effect=Exception("Test error"))

    client.force_login(alice)
    url = reverse("delete-document", kwargs={"doc_id": str(file.id)})
    response = client.post(url, {"doc_id": str(file.id)})

    assert response.status_code == HTTPStatus.OK
    file.refresh_from_db()
    assert file.status == File.Status.errored


@pytest.mark.django_db
def test_delete_document_invalid_doc_id(alice, mocker):
    """
    Test the delete_document endpoint with an invalid document ID.
    """
    logger_spy = mocker.spy(logging.getLogger("redbox_app.redbox_core.views.document_views"), "exception")

    factory = RequestFactory()
    request = factory.post("/documents/invalid-uuid/delete-document/", {"doc_id": "invalid-uuid"})
    request.user = alice

    response = delete_document(request, doc_id="invalid-uuid")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.content.decode() == "Invalid document ID"

    logger_spy.assert_called_once_with("Invalid document ID: %s", "invalid-uuid")


@pytest.mark.django_db
def test_delete_document_invalid_active_chat_id(alice, client, mocker):
    """
    Test the delete_document endpoint with an invalid active chat ID.
    """
    file_id = uuid.uuid4()
    file = File.objects.create(id=file_id, user=alice, original_file_name="test.pdf", status=File.Status.complete)

    logger_spy = mocker.spy(logging.getLogger("redbox_app.redbox_core.views.document_views"), "exception")

    mocker.patch.object(File, "delete_from_elastic_and_s3")

    client.force_login(alice)
    url = reverse("delete-document", kwargs={"doc_id": str(file.id)})
    invalid_chat_id = "invalid-uuid"
    response = client.post(
        url,
        {
            "doc_id": str(file.id),
            "session-id": "",
            "active_chat_id": invalid_chat_id,
            "file_selected": "True",
        },
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.content.decode() == "Invalid active chat ID"

    logger_spy.assert_called_once_with("Invalid active chat ID: %s", invalid_chat_id)


@pytest.mark.django_db
def test_upload_document_api_endpoint(alice, client, file_pdf_path, s3_client, remove_file_from_bucket):
    """
    Test the API endpoint for uploading a document.
    """
    file_name = f"{alice.email}/{file_pdf_path.name}"
    remove_file_from_bucket(file_name)
    assert not file_exists(s3_client, file_name)

    client.force_login(alice)
    previous_count = count_s3_objects(s3_client)

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

    # For successful uploads, the view redirects to /documents/
    assert response.status_code == HTTPStatus.FOUND
    assert response.url == "/documents/"
    assert count_s3_objects(s3_client) == previous_count + 1

    # Verify a file was created in the database
    uploaded_file = File.objects.filter(user=alice).order_by("-created_at")[0]
    assert uploaded_file.file_name.startswith(file_pdf_path.name.rstrip(file_pdf_path.name[-4:]).replace(" ", "_"))


@pytest.mark.django_db
def test_upload_document_api_invalid_file(alice, client, file_py_path):
    """
    Test the API endpoint with invalid file type.
    """
    client.force_login(alice)

    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

    assert response.status_code == HTTPStatus.OK
    content = str(response.content)
    assert "File type .py not supported" in content


@pytest.mark.django_db
def test_upload_document_to_skill(alice, client, original_file, default_skill, s3_client, remove_file_from_bucket):
    """
    Test the API endpoint with valid file and skill slug.
    """
    # Given
    client.force_login(alice)
    url = reverse("document-upload", kwargs={"slug": default_skill.slug})
    file_name = f"{alice.email}/{original_file.name.rstrip(original_file.name[-4:])}"
    remove_file_from_bucket(file_name)

    # When
    assert not file_exists(s3_client, file_name)
    previous_count = count_s3_objects(s3_client)
    response = client.post(url, {"file": original_file})
    uploaded_file = File.objects.filter(user=alice).order_by("-created_at")[0]

    # Then
    assert response.status_code == HTTPStatus.OK
    assert count_s3_objects(s3_client) == previous_count + 1
    assert uploaded_file.status == File.Status.processing
    assert FileSkill.objects.filter(
        file=uploaded_file, skill=default_skill, file_type=FileSkill.FileType.MEMBER
    ).exists()


@pytest.mark.django_db
def test_upload_invalid_document(alice, client, original_file, default_skill):
    """
    Test the API endpoint with invalid file.
    """
    # Given
    client.force_login(alice)
    url = reverse("document-upload", kwargs={"slug": default_skill.slug})

    original_file.name = "invalid"
    # When
    response = client.post(url, {"file": original_file})
    response_content = response.content.decode()

    # Then
    assert response.status_code == HTTPStatus.OK
    assert f"Error with {original_file.name}: File type  not supported" in response_content


@pytest.mark.django_db
def test_your_documents_with_chat(chat_with_files: User, client: Client):
    # Given
    user = chat_with_files.user
    client.force_login(user)
    factory = RequestFactory()
    request = factory.post(reverse("your-documents"))
    request.user = user
    file_context = document_service.decorate_file_context(request=request, skill=None, messages=[])
    completed_files = file_context["completed_files"]

    # When
    response = client.get(reverse("your-documents", kwargs={"active_chat_id": str(chat_with_files.id)}))
    soup = BeautifulSoup(response.content)
    doc_items = soup.find_all(
        "input",
        class_="govuk-checkboxes__input",
    )
    rendered_ids = [item["id"] for item in doc_items]
    checked_items = soup.find_all("input", class_="govuk-checkboxes__input", attrs=["checked"])

    # Then
    assert response.status_code == HTTPStatus.OK
    assert list(response.context_data["completed_files"]) == list(completed_files)
    for file in completed_files:
        assert f"file-{file.id}" in rendered_ids
    assert checked_items is not None


@pytest.mark.django_db
def test_your_documents_without_chat(chat_with_files: Chat, client: Client):
    # Given
    user = chat_with_files.user
    client.force_login(user)
    factory = RequestFactory()
    request = factory.post(reverse("your-documents"))
    request.user = user
    file_context = document_service.decorate_file_context(request=request, skill=None, messages=[])
    completed_files = file_context["completed_files"]

    # When
    response = client.get(reverse("your-documents"))
    soup = BeautifulSoup(response.content)
    doc_items = soup.find_all("input", class_="govuk-checkboxes__input")
    rendered_ids = [item["id"] for item in doc_items]

    # Then
    assert response.status_code == HTTPStatus.OK
    assert list(response.context_data["completed_files"]) == list(completed_files)
    for file in completed_files:
        assert f"file-{file.id}" in rendered_ids
    for doc_item in doc_items:
        assert not doc_item.has_attr("checked")

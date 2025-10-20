import logging
from collections.abc import MutableSequence, Sequence
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.exceptions import FieldError, SuspiciousFileOperation, ValidationError
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.template.response import TemplateResponse
from django_q.tasks import async_task

from redbox_app.redbox_core.models import File
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.types import APPROVED_FILE_EXTENSIONS
from redbox_app.worker import ingest

CHUNK_SIZE = 1024

User = get_user_model()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 209715200  # 200 MB or 200 * 1024 * 1024


def render_your_documents(request, active_chat_id) -> TemplateResponse:
    context = chat_service.get_context(request, active_chat_id)

    return TemplateResponse(
        request,
        "side_panel/your_documents_list.html",
        context,
    )


def build_upload_response(request: HttpRequest, errors: Sequence[str] | None = None) -> HttpResponse:
    return render(
        request,
        template_name="upload.html",
        context={
            "request": request,
            "errors": {"upload_doc": errors or []},
            "uploaded": not errors,
        },
    )


def validate_uploaded_file(uploaded_file: UploadedFile) -> Sequence[str]:
    errors: MutableSequence[str] = []
    if not uploaded_file.name:
        errors.append("File has no name")
    else:
        file_extension = Path(uploaded_file.name).suffix
        if file_extension.lower() not in APPROVED_FILE_EXTENSIONS:
            errors.append(f"Error with {uploaded_file.name}: File type {file_extension} not supported")
    if not uploaded_file.content_type:
        errors.append(f"Error with {uploaded_file.name}: File has no content-type")
    if uploaded_file.size > MAX_FILE_SIZE:
        errors.append(f"Error with {uploaded_file.name}: File is larger than 200MB")
    return errors


def ingest_file(uploaded_file: UploadedFile, user: User) -> tuple[Sequence[str], File | None]:
    try:
        logger.info("getting file from s3")
        file = File.objects.create(
            status=File.Status.processing.value,
            user=user,
            original_file=uploaded_file,
        )
    except (ValueError, FieldError, ValidationError) as e:
        logger.exception("Error creating File model object for %s.", uploaded_file, exc_info=e)
        return e.args, None
    except SuspiciousFileOperation:
        return [
            f"Your file name is {len(uploaded_file.name)} characters long. "
            f"The file name will need to be shortened by {len(uploaded_file.name) - 75} characters"
        ], None
    except Exception as e:
        logger.exception("Unexpected error processing %s.", uploaded_file, exc_info=e)
        return [str(e)], None
    else:
        async_task(ingest, file.id, task_name=file.unique_name, group="ingest")
        return [], file

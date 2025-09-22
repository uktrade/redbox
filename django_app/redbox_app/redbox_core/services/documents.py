import logging
import subprocess
import tempfile
import time
from collections.abc import MutableSequence, Sequence
from io import BytesIO
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.exceptions import FieldError, SuspiciousFileOperation, ValidationError
from django.core.files.uploadedfile import InMemoryUploadedFile, UploadedFile
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


def create_file_without_ingest(uploaded_file: UploadedFile, user: User) -> tuple[Sequence[str], File | None]:
    """
    Create a File object for the uploaded file without ingesting it.
    Returns a tuple of errors and file.
    """
    try:
        logger.info("Creating file object for %s", uploaded_file.name)
        file = File.objects.create(
            status=File.Status.processing.value,
            user=user,
            original_file=uploaded_file,
            original_file_name=uploaded_file.name,
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
        logger.exception("Unexpected error creating file %s.", uploaded_file, exc_info=e)
        return [str(e)], None
    else:
        return [], file


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
    if uploaded_file.size > MAX_FILE_SIZE:
        errors.append(f"Error with {uploaded_file.name}: File is larger than 200MB")
    return errors


def is_utf8_compatible(uploaded_file: UploadedFile) -> bool:
    if not Path(uploaded_file.name).suffix.lower().endswith((".doc", ".txt")):
        logger.info("File does not require utf8 compatibility check")
        return True
    try:
        uploaded_file.open()
        uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)
    except UnicodeDecodeError:
        logger.info("File is incompatible with utf-8. Converting...")
        return False
    else:
        logger.info("File is compatible with utf-8 - ready for processing")
        return True


def convert_to_utf8(uploaded_file: UploadedFile) -> UploadedFile:
    try:
        uploaded_file.open()
        content = uploaded_file.read().decode("ISO-8859-1")

        # Detect and replace non-UTF-8 characters
        new_bytes = content.encode("utf-8")

        # Creating a new InMemoryUploadedFile object with the converted content
        new_uploaded_file = InMemoryUploadedFile(
            file=BytesIO(new_bytes),
            field_name=uploaded_file.name,
            name=uploaded_file.name,
            content_type="application/octet-stream",
            size=len(new_bytes),
            charset="utf-8",
        )
    except Exception as e:
        logger.exception("Error converting file %s to UTF-8.", uploaded_file, exc_info=e)
        return uploaded_file
    else:
        logger.info("Conversion to UTF-8 successful")
        return new_uploaded_file


def is_doc_file(uploaded_file: UploadedFile) -> bool:
    return Path(uploaded_file.name).suffix.lower() == ".doc"


def convert_doc_to_docx(uploaded_file: UploadedFile) -> UploadedFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_input:
        tmp_input.write(uploaded_file.read())
        tmp_input.flush()
        input_path = Path(tmp_input.name)
        output_dir = input_path.parent

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path = input_path.with_suffix(".docx")

        try:
            result = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    str(input_path),
                    "--outdir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                cwd=output_dir,
            )
            logger.info("LibreOffice output: %s", result.stdout.decode())
            logger.info("LibreOffice errors: %s", result.stderr.decode())

            if not temp_output_path.exists():
                logger.error("Output file not found: %s", temp_output_path)
                return uploaded_file

            logger.info("Output path: %s", temp_output_path)

            time.sleep(1)
            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                logger.info("Converted file size: %d bytes", len(converted_content))
                if len(converted_content) == 0:
                    logger.error("Converted file is empty - this won't get converted")

                output_filename = Path(uploaded_file.name).with_suffix(".docx").name
                new_file = InMemoryUploadedFile(
                    file=BytesIO(converted_content),
                    field_name=uploaded_file.name,
                    name=output_filename,
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size=len(converted_content),
                    charset="utf-8",
                )
                logger.info("doc file conversion to docx successful for %s", uploaded_file.name)
        except Exception as e:
            logger.exception("Error converting doc file %s to docx", uploaded_file.name, exc_info=e)
            new_file = uploaded_file
        finally:
            try:
                input_path.unlink()
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning("Error cleaning up temporary files: %s", cleanup_error)

        return new_file


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

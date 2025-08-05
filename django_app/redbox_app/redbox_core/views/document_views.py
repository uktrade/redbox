import logging
import subprocess
import tempfile
import time
import uuid
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

from dataclasses_json import Undefined, dataclass_json
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.exceptions import FieldError, SuspiciousFileOperation, ValidationError
from django.core.files.uploadedfile import InMemoryUploadedFile, UploadedFile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from django_q.tasks import async_task

from redbox_app.redbox_core.models import File, InactiveFileError
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.utils import render_with_oob
from redbox_app.worker import ingest

User = get_user_model()
logger = logging.getLogger(__name__)
CHUNK_SIZE = 1024
# move this somewhere
APPROVED_FILE_EXTENSIONS = [
    ".eml",
    ".html",
    ".json",
    ".md",
    ".msg",
    ".rst",
    ".rtf",
    ".txt",
    ".xml",
    ".csv",
    ".doc",
    ".docx",
    ".epub",
    ".epub",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".tsv",
    ".xlsx",
    ".htm",
]
MAX_FILE_SIZE = 209715200  # 200 MB or 200 * 1024 * 1024


class DocumentView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        completed_files, processing_files = File.get_completed_and_processing_files(request.user)

        ingest_errors = request.session.get("ingest_errors", [])
        request.session["ingest_errors"] = []

        return render(
            request,
            template_name="documents.html",
            context={
                "request": request,
                "completed_files": completed_files,
                "processing_files": processing_files,
                "ingest_errors": ingest_errors,
            },
        )


class UploadView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        return self.build_response(request)

    @method_decorator(login_required)
    def post(self, request: HttpRequest) -> HttpResponse:
        errors: MutableSequence[str] = []

        uploaded_files: MutableSequence[UploadedFile] = request.FILES.getlist("uploadDocs")

        if not uploaded_files:
            errors.append("No document selected")

        for index, uploaded_file in enumerate(uploaded_files):
            errors += self.validate_uploaded_file(uploaded_file)
            # handling doc -> docx conversion
            if self.is_doc_file(uploaded_file):
                uploaded_files[index] = self.convert_doc_to_docx(uploaded_file)
            # handling utf8 compatibility
            if not self.is_utf8_compatible(uploaded_file):
                uploaded_files[index] = self.convert_to_utf8(uploaded_file)

        if not errors:
            for uploaded_file in uploaded_files:
                # ingest errors are handled differently, as the other documents have started uploading by this point
                request.session["ingest_errors"] = self.ingest_file(uploaded_file, request.user)
            return redirect(reverse("documents"))

        return self.build_response(request, errors)

    @staticmethod
    def build_response(request: HttpRequest, errors: Sequence[str] | None = None) -> HttpResponse:
        return render(
            request,
            template_name="upload.html",
            context={
                "request": request,
                "errors": {"upload_doc": errors or []},
                "uploaded": not errors,
            },
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def is_doc_file(uploaded_file: UploadedFile) -> bool:
        return Path(uploaded_file.name).suffix.lower() == ".doc"

    @staticmethod
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

    @staticmethod
    def ingest_file(uploaded_file: UploadedFile, user: User) -> Sequence[str]:
        try:
            logger.info("getting file from s3")
            file = File.objects.create(
                status=File.Status.processing.value,
                user=user,
                original_file=uploaded_file,
            )
        except (ValueError, FieldError, ValidationError) as e:
            logger.exception("Error creating File model object for %s.", uploaded_file, exc_info=e)
            return e.args
        except SuspiciousFileOperation:
            return [
                f"Your file name is {len(uploaded_file.name)} characters long. "
                f"The file name will need to be shortened by {len(uploaded_file.name) - 75} characters"
            ]
        except Exception as e:
            logger.exception("Unexpected error processing %s.", uploaded_file, exc_info=e)
            return [str(e)]
        else:
            async_task(ingest, file.id, task_name=file.unique_name, group="ingest")
            return []


@login_required
def remove_doc_view(request, doc_id: uuid):
    file = get_object_or_404(File, id=doc_id)
    errors: list[str] = []

    if request.method == "POST":
        try:
            file.delete_from_elastic()
            file.delete_from_s3()
            file.status = File.Status.deleted
            file.save()
            logger.info("Removing document: %s", request.POST["doc_id"])
        except Exception as e:
            logger.exception("Error deleting file object %s.", file, exc_info=e)
            errors.append("There was an error deleting this file")
            file.status = File.Status.errored
            file.save()

        return redirect("documents")

    return render(
        request,
        template_name="remove-doc.html",
        context={"request": request, "doc_id": doc_id, "doc_name": file.file_name, "errors": errors},
    )


@login_required
def remove_all_docs_view(request):
    users_files = File.objects.filter(user=request.user)
    errors: list[str] = []

    if request.method == "POST":
        for file in users_files:
            try:
                file.delete_from_elastic()
            except InactiveFileError:
                logger.warning("File %s is inactive skipping delete_from_elastic", file)

            try:
                file.delete_from_s3()
            except Exception as e:
                logger.exception("Error deleting file %s from S3", file, exc_info=e)
                errors.append(f"Error deleting file {file.id} from S3")

            file.status = File.Status.deleted
            file.save()
            logger.info("Marking document %s as deleted", file.id)

        return redirect("documents")

    return render(
        request,
        template_name="remove-all-docs.html",
        context={"request": request, "errors": errors},
    )


@require_http_methods(["GET"])
@login_required
def file_status_api_view(request: HttpRequest) -> JsonResponse:
    file_id = request.GET.get("id", None)
    if not file_id:
        logger.error("Error getting file object information - no file ID provided %s.")
        return JsonResponse({"status": File.Status.errored.label})
    try:
        file: File = get_object_or_404(File, id=file_id)
    except File.DoesNotExist as ex:
        logger.exception("File object information not found in django - file does not exist %s.", file_id, exc_info=ex)
        return JsonResponse({"status": File.Status.errored.label})
    return JsonResponse({"status": file.get_status_text()})


@require_http_methods(["POST"])
@login_required
def delete_document(request, doc_id: uuid):
    file = get_object_or_404(File, id=doc_id)
    errors: list[str] = []
    try:
        file.delete_from_elastic()
        file.delete_from_s3()
        file.status = File.Status.deleted
        file.save()
        logger.info("Removing document: %s", request.POST["doc_id"])
    except Exception as e:
        logger.exception("Error deleting file object %s.", file, exc_info=e)
        errors.append("There was an error deleting this file")
        file.status = File.Status.errored
        file.save()

    currently_selected_document = False
    session_id = request.POST.get("session-id")
    file_selected = request.POST.get("file_selected")
    active_chat_id = session_id if session_id else request.POST.get("active_chat_id")

    if active_chat_id != "None":
        active_chat_id = uuid.UUID(active_chat_id)
        if file_selected == "True":
            active_chat_id = None
            currently_selected_document = True
    else:
        active_chat_id = None

    if currently_selected_document:
        context = chat_service.get_context(request, active_chat_id)

        return render_with_oob(
            [
                {"template": "side_panel/your_documents_list.html", "context": context, "request": request},
                {"template": "chat/chat_window.html", "context": context, "request": request},
            ]
        )
    return documents_service.render_your_documents(request, active_chat_id)


class YourDocuments(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, active_chat_id: uuid.UUID) -> HttpResponse:
        return documents_service.render_your_documents(request, active_chat_id)


class DocumentsTitleView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Title:
        value: str

    @method_decorator(login_required)
    def post(self, request: HttpRequest, doc_id: uuid.UUID) -> HttpResponse:
        file = get_object_or_404(File, id=doc_id)

        request_body = DocumentsTitleView.Title.schema().loads(request.body)

        # Debug and test the above first
        file.original_file_name = request_body.value
        file.save(update_fields=["original_file_name"])

        return HttpResponse(status=HTTPStatus.NO_CONTENT)

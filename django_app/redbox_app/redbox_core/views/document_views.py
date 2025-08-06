import logging
import uuid
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING

import requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from django_q.tasks import async_task

from redbox_app.redbox_core.models import File, InactiveFileError

if TYPE_CHECKING:
    from django.core.files.uploadedfile import UploadedFile

User = get_user_model()
logger = logging.getLogger(__name__)


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
            return self.build_response(request, errors)

        # Goes to the microservice
        try:
            files = [("files", (f.name, f, f.content_type)) for f in uploaded_files]
            headers = {"X-User-ID": str(request.user.id)}  # Pass user ID in headers
            response = requests.post(
                f"{settings.FILE_PROCESSOR_URL}/process-files/",
                files=files,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException:
            logger.exception("Error communicating with file processor")
            errors.append("Error processing files. Please try again later.")
            return self.build_response(request, errors)

        errors.extend(result["errors"])
        if not errors:
            for processed_file in result["processed_files"]:
                try:
                    file_obj = File.objects.create(
                        id=uuid.uuid4(),
                        status=File.Status.processing,
                        user=request.user,
                        original_file_name=processed_file["filename"],
                        minio_path=processed_file["minio_path"],
                    )
                    async_task(
                        "redbox_app.redbox_core.tasks.ingest",
                        file_obj.id,
                        task_name=file_obj.unique_name,
                        group="ingest",
                    )
                except Exception:
                    logger.exception("Error queuing file %s for ingestion", processed_file["filename"])
                    errors.append("Error processing %s", processed_file["filename"])
                    request.session["ingest_errors"] = errors

            if not errors:
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

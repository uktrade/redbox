import logging
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from django.core.files.uploadedfile import UploadedFile

from dataclasses_json import Undefined, dataclass_json
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import File, InactiveFileError
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.services import documents as documents_service
from redbox_app.redbox_core.utils import render_with_oob

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
        return documents_service.build_upload_response(request)

    @method_decorator(login_required)
    def post(self, request: HttpRequest) -> HttpResponse:
        errors: MutableSequence[str] = []

        uploaded_files: MutableSequence[UploadedFile] = request.FILES.getlist("uploadDocs")

        if not uploaded_files:
            errors.append("No document selected")

        for _index, uploaded_file in enumerate(uploaded_files):
            errors += documents_service.validate_uploaded_file(uploaded_file)

        if not errors:
            for uploaded_file in uploaded_files:
                # ingest errors are handled differently, as the other documents have started uploading by this point
                request.session["ingest_errors"], _ = documents_service.ingest_file(uploaded_file, request.user)
            return redirect(reverse("documents"))

        return documents_service.build_upload_response(request, errors)


@require_http_methods(["POST"])
@login_required
def upload_document(request):
    errors: MutableSequence[str] = []

    uploaded_file: UploadedFile = request.FILES.get("file")
    response = {}

    if not uploaded_file:
        errors.append("No document selected")

    errors += documents_service.validate_uploaded_file(uploaded_file)

    if errors:
        response["errors"] = errors
        return JsonResponse(response)

    # ingest errors are handled differently, as the other documents have started uploading by this point
    ingest_errors, file = documents_service.ingest_file(uploaded_file, request.user)
    request.session["ingest_errors"] = ingest_errors

    if ingest_errors:
        response["ingest_errors"] = file.status

    if file:
        response["status"] = file.status
        response["file_id"] = str(file.id)
        response["file_name"] = file.file_name

    return JsonResponse(response)


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


@require_http_methods(["POST"])
@login_required
def delete_document(request, doc_id: uuid.UUID):
    try:
        doc_uuid = uuid.UUID(str(doc_id))
    except ValueError:
        logger.exception("Invalid document ID: %s", doc_id)
        return HttpResponseBadRequest("Invalid document ID")

    file = get_object_or_404(File, id=doc_uuid, user=request.user)
    errors: list[str] = []

    try:
        file.delete_from_elastic()
        file.delete_from_s3()
        file.status = File.Status.deleted
        file.save()
        logger.info("Removing document: %s", request.POST.get("doc_id"))
    except Exception as e:
        logger.exception("Error deleting file object %s.", file, exc_info=e)
        errors.append("There was an error deleting this file")
        file.status = File.Status.errored
        file.save()

    session_id = request.POST.get("session-id")
    file_selected = request.POST.get("file_selected")
    active_chat_id = session_id if session_id else request.POST.get("active_chat_id")

    if active_chat_id and active_chat_id != "None":
        try:
            active_chat_id = uuid.UUID(str(active_chat_id))
        except ValueError:
            logger.exception("Invalid active chat ID: %s", active_chat_id)
            return HttpResponseBadRequest("Invalid active chat ID")
    else:
        active_chat_id = None

    selected_document = False
    if active_chat_id and file_selected == "True":
        active_chat_id = None
        selected_document = True

    if selected_document:
        context = chat_service.get_context(request, active_chat_id)
        oob_context = context
        oob_context["oob"] = True

        return render_with_oob(
            [
                {"template": "side_panel/your_documents_list.html", "context": context, "request": request},
                {"template": "chat/chat_window.html", "context": oob_context, "request": request},
            ]
        )

    return documents_service.render_your_documents(request, active_chat_id)


class YourDocuments(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, active_chat_id: uuid.UUID | None = None) -> HttpResponse:
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

        file.original_file_name = request_body.value
        file.save(update_fields=["original_file_name"])

        return HttpResponse(status=HTTPStatus.NO_CONTENT)

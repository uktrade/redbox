import logging
from http import HTTPStatus

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import File
from redbox_app.redbox_core.types import FILE_EXTENSION_MAPPING

logger = logging.getLogger(__name__)


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


@require_http_methods(["GET"])
@login_required
def file_ingest_errors_view(request: HttpRequest) -> JsonResponse:
    if not request.user.is_superuser:
        return HttpResponseForbidden()
    file_id = request.GET.get("id", None)
    if not file_id:
        errors_map = {
            str(file.id): {
                "created_at": file.created_at,
                "file_name": file.file_name,
                "ingest_error": file.ingest_error,
            }
            for file in File.objects.exclude(ingest_error__isnull=True).exclude(ingest_error__exact="")
        }
        return JsonResponse(errors_map)
    try:
        file: File = get_object_or_404(File, id=file_id)
    except File.DoesNotExist as ex:
        logger.exception("File object information not found in django - file does not exist %s.", file_id, exc_info=ex)
        return JsonResponse(
            {
                "status": "File object information not found in django - file does not exist %s.",
                file_id: ex,
            }
        )
    return JsonResponse(
        {
            str(file.id): {
                "created_at": file.created_at,
                "file_name": file.file_name,
                "ingest_error": file.ingest_error,
            }
        }
    )


@require_http_methods(["GET"])
def file_icon_view(request, ext: str):
    # Ensure extension starts with a period
    ext = f".{ext.lower()}" if not ext.startswith(".") else ext.lower()

    # Validate extension
    if ext not in FILE_EXTENSION_MAPPING:
        err = f"No icon available for file type: {ext}"
        logger.exception(err, ext)
        return HttpResponse(status=HTTPStatus.NO_CONTENT)

    icon_filename = FILE_EXTENSION_MAPPING[ext]

    return render(
        request,
        template_name=f"rbds/icons/{icon_filename}.svg",
        context={"request": request, "icon_classes": "file-icon"},
    )

import logging
import uuid
from http import HTTPStatus

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import Team
from redbox_app.redbox_core.services import notifications as notifications_service

User = get_user_model()
logger = logging.getLogger(__name__)


@require_http_methods(["POST"])
@login_required
def send_team_addition_email_view(request: HttpRequest):
    user = get_object_or_404(User, id=uuid.UUID(request.POST.get("user-id")))
    team = get_object_or_404(Team, id=uuid.UUID(request.POST.get("team-id")))

    if not team.is_admin(request.user):
        return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)

    try:
        notifications_service.send_team_addition_email(request.user.pk, user.pk, team.pk)
    except Exception as e:
        error_id = str(uuid.uuid4())
        error = f"Error ID: {error_id} - Failed to queue task: {e}"
        logger.exception(error)
        return HttpResponse(
            f"An error occurred (Reference: {error_id}). Please contact support if the problem persists.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return HttpResponse(status=HTTPStatus.NO_CONTENT)

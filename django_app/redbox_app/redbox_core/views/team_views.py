import logging
import uuid
from http import HTTPStatus
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.models import Team, UserTeamMembership

User = get_user_model()
logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
@login_required
def add_team_member_row_view(request: HttpRequest, team_id: uuid.UUID, user_id: uuid.UUID):
    user = get_object_or_404(User, id=user_id)
    team = get_object_or_404(Team, id=team_id)

    if not team.is_admin(request.user):
        return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)

    try:
        member = UserTeamMembership(user=user, team=team, role_type=UserTeamMembership.RoleType.MEMBER)
    except Exception as e:
        error = f"Failed to create UserTeamMembership object: {e}"
        logger.exception(error)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    context = {
        "member": member,
        "role_choices": UserTeamMembership.RoleType.choices,
        "show_select": True,
        "is_new_member": True,
    }

    return render(request, "settings/team-member-row.html", context)


@require_http_methods(["GET"])
@login_required
def edit_team_member_row_view(request: HttpRequest, member_id: uuid.UUID | Any):
    member = get_object_or_404(UserTeamMembership, id=member_id)

    if not member.team.is_admin(request.user):
        return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)

    context = {
        "member": member,
        "role_choices": UserTeamMembership.RoleType.choices,
        "editing": request.GET.get("cancel").lower() != "true",
    }

    return render(request, "settings/team-member-row.html", context)


@require_http_methods(["DELETE"])
@login_required
def delete_team_member_row_view(request: HttpRequest, member_id: uuid.UUID | Any):
    member = UserTeamMembership.objects.filter(id=member_id).select_related("team").first()

    if member:
        if not member.team.is_admin(request.user):
            return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)
        member.delete()

    return HttpResponse(status=HTTPStatus.OK)


@require_http_methods(["POST"])
@login_required
def create_team_view(request: HttpRequest):
    team_name = request.POST.get("team-name")
    directorate = request.POST.get("directorate")

    try:
        Team(team_name=team_name, directorate=directorate)
    except Exception as e:
        error = f"Failed to create Team: {e}"
        logger.exception(error)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return HttpResponse(status=HTTPStatus.OK)


@require_http_methods(["POST"])
@login_required
def add_team_member_view(request: HttpRequest, team_id: uuid.UUID):
    team = get_object_or_404(Team, id=team_id)
    user = get_object_or_404(User, id=request.POST.get("user-id"))
    role_type = request.POST.get("role-type")
    next_url = request.GET.get("next")

    try:
        team.add_member(user, role_type)
    except Exception as e:
        error = f"Failed to add team member: {e}"
        logger.exception(error)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    if next_url:
        return redirect(next_url)

    return HttpResponse(status=HTTPStatus.OK)


@require_http_methods(["POST"])
@login_required
def edit_team_member_view(request: HttpRequest, member_id: uuid.UUID | Any):
    member = get_object_or_404(UserTeamMembership, id=member_id)
    role_type = request.POST.get("role-type")

    if not member.team.is_admin(request.user):
        return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)

    try:
        member.role_type = role_type
        member.save()
    except Exception as e:
        error = f"Failed to update UserTeamMembership object: {e}"
        logger.exception(error)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    context = {
        "member": member,
        "role_choices": UserTeamMembership.RoleType.choices,
        "show_select": False,
    }

    return render(request, "settings/team-member-row.html", context)


@require_http_methods(["DELETE"])
@login_required
def delete_team_view(request: HttpRequest, team_id: uuid.UUID):
    team = get_object_or_404(Team, id=team_id)
    next_url = request.GET.get("next")

    if not team.is_admin(request.user):
        return HttpResponse("You are not an admin of this team.", status=HTTPStatus.FORBIDDEN)

    try:
        team.delete()
    except Exception as e:
        error = f"Failed to delete Team object: {e}"
        logger.exception(error)
        return HttpResponse(error, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    if next_url:
        return redirect(next_url)

    return HttpResponse(status=HTTPStatus.OK)

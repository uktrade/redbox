import logging

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.utils.decorators import method_decorator
from django.views import View
from waffle import flag_is_active

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.forms import DemographicsForm
from redbox_app.redbox_core.models import UserTeamMembership
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.services import teams as teams_service
from redbox_app.redbox_core.types import TabConfig, TabRegistry
from redbox_app.redbox_core.utils import save_forms

User = get_user_model()
logger = logging.getLogger(__name__)


class SettingsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        context = chat_service.get_context(request)
        context["tabs"] = tab_configs(request).get_context(request)

        return TemplateResponse(request, "settings/settings.html", context)

    @method_decorator(login_required)
    def post(self, request: HttpRequest) -> HttpResponse:
        default_tab = profile_tab_config()
        active_tab = request.POST.get("active_tab", default_tab.id)
        tab_config = tab_configs(request)[active_tab]

        try:
            tab_config.handle_post(request)
        except Exception as e:
            msg = f"Failed to process POST event: {e}"
            logger.exception(msg, exc_info=e)

        return redirect(f"{request.path}#{active_tab}")


def tab_configs(request: HttpRequest) -> TabRegistry:
    configs = [
        profile_tab_config(),
        teams_tab_config(request),
    ]

    return TabRegistry([config for config in configs if config is not None])


def profile_tab_config() -> TabConfig:
    def get_my_details_tab_form(request: HttpRequest):
        return DemographicsForm(request.POST or None, instance=request.user)

    return TabConfig(
        id="my-details",
        title="My Details",
        template="settings/my-details.html",
        get_context=lambda request: {"form": get_my_details_tab_form(request)},
        handle_post=lambda request: save_forms(get_my_details_tab_form(request)),
    )


def teams_tab_config(request: HttpRequest) -> TabConfig | None:
    if not flag_is_active(request.user, flags.ENABLE_TEAMS):
        return None

    def handle_post(request: HttpRequest):
        team = teams_service.create_team(
            request.POST.get("name"),
            request.POST.get("directorate"),
            request.user,
        )
        messages.success(request, f"{team.team_name} created successfully", str(team.id))

    return TabConfig(
        id="manage-teams",
        title="Manage Teams",
        template="settings/manage-teams.html",
        get_context=lambda request: {
            "user_teams": UserTeamMembership.objects.filter(user=request.user),
            "role_choices": UserTeamMembership.RoleType.choices,
            "default_role": UserTeamMembership.RoleType.MEMBER,
            "users": User.objects.all(),
        },
        handle_post=handle_post,
    )

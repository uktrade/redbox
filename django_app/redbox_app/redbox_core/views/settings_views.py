import logging
from typing import ClassVar

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.views import View

from redbox_app.redbox_core.forms import DemographicsForm
from redbox_app.redbox_core.types import TabConfig, TabRegistry
from redbox_app.redbox_core.utils import save_forms

logger = logging.getLogger(__name__)


def get_my_details_tab_form(request: HttpRequest):
    return DemographicsForm(request.POST or None, instance=request.user)


class SettingsView(View):
    tabs: ClassVar[TabRegistry] = TabRegistry(
        [
            TabConfig(
                id="my-details",
                title="My Details",
                template="settings/my-details.html",
                get_context=lambda request: {"form": get_my_details_tab_form(request)},
                handle_post=lambda request: save_forms(get_my_details_tab_form(request)),
            ),
            TabConfig(
                id="manage-teams",
                title="Manage Teams",
                template="settings/manage-teams.html",
            ),
        ]
    )

    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        context = {"request": request, "tabs": []}

        for tab in self.tabs:
            context["tabs"].append(
                {
                    "id": tab.id,
                    "title": tab.title,
                    "template": tab.template,
                    "context": tab.get_context(request),
                }
            )

        return render(request, "settings/settings.html", context)

    @method_decorator(login_required)
    def post(self, request: HttpRequest) -> HttpResponse:
        active_tab = request.POST.get("active_tab", "my-details")
        tab_config = self.tabs[active_tab]

        try:
            result = tab_config.handle_post(request)
        except Exception as e:
            msg = f"Failed to process POST event: {result}"
            logger.exception(msg, exc_info=e)

        return redirect(f"{request.path}#{active_tab}")

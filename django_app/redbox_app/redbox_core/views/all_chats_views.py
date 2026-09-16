from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from waffle.decorators import waffle_flag

from redbox_app.redbox_core.services import chats as chat_service


@method_decorator(waffle_flag("enable_chats_redesign"), name="get")
class AllChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        context = chat_service.get_context(request)

        # Default to utc initially, allows rendering without js enabled
        # replaced with localised version by htmx on page load
        context["grouped_chats"] = chat_service.get_filtered_and_grouped_chats(request.user, tz=ZoneInfo("UTC"))

        return render(
            request,
            template_name="all_chats/all-chats.html",
            context=context,
        )


@method_decorator(waffle_flag("enable_chats_redesign"), name="get")
class SearchChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        tz_name = request.GET.get("tz") or "UTC"

        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")

        search_query = request.GET.get("q", "").strip() or None

        context = {
            "grouped_chats": chat_service.get_filtered_and_grouped_chats(
                request.user, tz, tool=None, chat_name_query=search_query
            )
        }

        return render(
            request,
            template_name="all_chats/_chat_tables.html",
            context=context,
        )

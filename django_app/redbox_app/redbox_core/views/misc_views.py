import logging
import os
from http import HTTPStatus

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import connections
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.connection import ConnectionDoesNotExist
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from django.views.generic.base import RedirectView
from opensearchpy import ConnectionError as OpenSearchConnectionError

from redbox.models.settings import get_settings
from redbox_app.redbox_core.services import chats as chat_service
from redbox_app.redbox_core.services import sso as sso_service
from redbox_app.redbox_core.types import FRAGMENTS, RenderTemplateItem
from redbox_app.redbox_core.utils import parse_uuid, render_with_oob

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def homepage_view(request):
    if not request.user.is_authenticated:
        return render(
            request,
            template_name="homepage.html",
            context={"request": request, "allow_sign_ups": settings.ALLOW_SIGN_UPS},
        )

    sso_service.sync_sso_data(request)
    return redirect("chats")


@require_http_methods(["GET"])
def health(_request: HttpRequest) -> HttpResponse:
    """this required by ECS Fargate"""

    logger.info("Checking if the DB is accessible")

    try:
        conn = connections["default"]
        logger.info("Connected to default DB %s", conn)
    except ConnectionDoesNotExist as exc:
        logger.info("Database could not be connected: %s", exc, exc_info=True)
        return HttpResponse(content="Database unavailable", status=HTTPStatus.SERVICE_UNAVAILABLE)

    logger.info(
        "Checking if the Elastic Search server %s is accessible", os.environ.get("COLLECTION_ENDPOINT", "MISSING VALUE")
    )

    env = get_settings()

    es_client = env.elasticsearch_client()

    try:
        info = es_client.info()
        logger.info("Connected to elastic search cluster %s", info)
    except OpenSearchConnectionError as exc:
        logger.info("Elastic search client could not be connected: %s", exc, exc_info=True)
        return HttpResponse(content="Elastic search cluster unavailable", status=HTTPStatus.SERVICE_UNAVAILABLE)

    return HttpResponse(status=HTTPStatus.OK)


class SecurityTxtRedirectView(RedirectView):
    """See https://github.com/alphagov/security.txt"""

    url = f"{settings.SECURITY_TXT_REDIRECT}"


@require_http_methods(["GET"])
def sitemap_view(request):
    return render(
        request,
        template_name="sitemap.html",
        context=chat_service.get_context(request),
    )


def faq_view(request):
    return render(
        request,
        template_name="faq.html",
        context=chat_service.get_context(request),
    )


class RefreshFragmentsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        active_chat_id = parse_uuid(request.GET.get("chat"))
        slug = request.GET.get("tool")

        fragments = request.GET.getlist("fragments")
        unknown_fragments = set(fragments) - FRAGMENTS.keys()

        if not fragments:
            return HttpResponse(
                "No fragments specified",
                status=HTTPStatus.BAD_REQUEST,
            )

        if unknown_fragments:
            return HttpResponse(
                f"Unknown fragments: {', '.join(sorted(unknown_fragments))}",
                status=HTTPStatus.BAD_REQUEST,
            )

        context = chat_service.get_context(request, active_chat_id, slug)
        context["oob"] = True

        renders: list[RenderTemplateItem] = []

        for fragment_id in fragments:
            fragment = FRAGMENTS.get(fragment_id)
            renders.append(
                RenderTemplateItem(
                    template=fragment.template,
                    context=context,
                    request=request,
                )
            )

        return render_with_oob(renders)

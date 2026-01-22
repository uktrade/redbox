import logging
from http import HTTPStatus

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods
from django.views.generic.base import RedirectView

from redbox_app.redbox_core.services import chats as chat_service

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def homepage_view(request):
    if not request.user.is_authenticated:
        return render(
            request,
            template_name="homepage.html",
            context={"request": request, "allow_sign_ups": settings.ALLOW_SIGN_UPS},
        )

    return redirect("chats")


@require_http_methods(["GET"])
def health(_request: HttpRequest) -> HttpResponse:
    """this required by ECS Fargate"""
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

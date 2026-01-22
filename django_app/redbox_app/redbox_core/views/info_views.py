"""
Views for info pages like privacy notice, accessibility statement, etc.
These shouldn't contain sensitive data and don't require login.
"""

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from redbox_app.redbox_core.services import chats as chat_service


@require_http_methods(["GET"])
def privacy_notice_view(request):
    return render(
        request,
        template_name="privacy-notice.html",
        context=chat_service.get_context(request),
    )


@require_http_methods(["GET"])
def support_view(request):
    context = chat_service.get_context(request)
    context["version"] = settings.REDBOX_VERSION

    return render(
        request,
        template_name="support.html",
        context=context,
    )


@require_http_methods(["GET"])
def accessibility_statement_view(request):
    return render(
        request,
        template_name="accessibility-statement.html",
        context=chat_service.get_context(request),
    )

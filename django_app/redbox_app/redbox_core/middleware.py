import json

import sentry_sdk
from asgiref.sync import iscoroutinefunction, sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.utils.decorators import sync_and_async_middleware
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

User = get_user_model()


@sync_and_async_middleware
def nocache_middleware(get_response):
    async def middleware(request: HttpRequest) -> HttpResponse:
        response = await get_response(request) if iscoroutinefunction(get_response) else get_response(request)
        response["Cache-Control"] = "no-store"
        return response

    return middleware


@sync_and_async_middleware
def security_header_middleware(get_response):
    report_to = json.dumps(
        {
            "group": "csp-endpoint",
            "max_age": 10886400,
            "endpoints": [{"url": settings.SENTRY_REPORT_TO_ENDPOINT}],
            "include_subdomains": True,
        },
        indent=None,
        separators=(",", ":"),
        default=str,
    )

    async def middleware(request: HttpRequest) -> HttpResponse:
        response = await get_response(request) if iscoroutinefunction(get_response) else get_response(request)
        if settings.SENTRY_REPORT_TO_ENDPOINT:
            response["Report-To"] = report_to
        return response

    return middleware


@sync_and_async_middleware
def plotly_no_csp_no_xframe_middleware(get_response):
    async def middleware(request: HttpRequest) -> HttpResponse:
        response = await get_response(request) if iscoroutinefunction(get_response) else get_response(request)
        if "admin/report" in request.path:
            response.headers.pop("Content-Security-Policy", None)
            response.headers.pop("X-Frame-Options", None)
        return response

    return middleware


class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key = request.headers.get("X-API-KEY")

        if not api_key:
            msg = "No API key provided"
            raise AuthenticationFailed(msg)

        if api_key == settings.REDBOX_API_KEY:
            user, _ = User.objects.get_or_create(
                username="api_key_user", defaults={"is_staff": True, "is_superuser": False}
            )
            return (user, None)

        msg = "Invalid API key"
        raise AuthenticationFailed(msg)


@sync_and_async_middleware
class SentryUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    async def __call__(self, request):
        if hasattr(request, "user") and await sync_to_async(lambda: request.user.is_authenticated)():
            await sync_to_async(sentry_sdk.set_user)(
                {
                    "id": request.user.id,
                    "email": request.user.email,
                    "name": request.user.name,
                }
            )
        return (
            await self.get_response(request) if iscoroutinefunction(self.get_response) else self.get_response(request)
        )

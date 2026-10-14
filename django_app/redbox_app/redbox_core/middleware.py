import asyncio
import json
from datetime import timedelta

import sentry_sdk
from asgiref.sync import iscoroutinefunction, sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.utils import timezone
from django.utils.decorators import sync_and_async_middleware
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from waffle import flag_is_active

from redbox_app.redbox_core import flags
from redbox_app.redbox_core.services import sso as sso_service

User = get_user_model()


@sync_and_async_middleware
def nocache_middleware(get_response):
    if iscoroutinefunction(get_response):

        async def middleware(request: HttpRequest) -> HttpResponse:
            response = await get_response(request)
            response["Cache-Control"] = "no-store"
            return response
    else:

        def middleware(request: HttpRequest) -> HttpResponse:
            response = get_response(request)
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

    if iscoroutinefunction(get_response):

        async def middleware(request: HttpRequest) -> HttpResponse:
            response = await get_response(request)
            if settings.SENTRY_REPORT_TO_ENDPOINT:
                response["Report-To"] = report_to
            return response
    else:

        def middleware(request: HttpRequest) -> HttpResponse:
            response = get_response(request)
            if settings.SENTRY_REPORT_TO_ENDPOINT:
                response["Report-To"] = report_to
            return response

    return middleware


@sync_and_async_middleware
def plotly_no_csp_no_xframe_middleware(get_response):
    if iscoroutinefunction(get_response):

        async def middleware(request: HttpRequest) -> HttpResponse:
            response = await get_response(request)
            if "admin/report" in request.path:
                response.headers.pop("Content-Security-Policy", None)
                response.headers.pop("X-Frame-Options", None)
            return response
    else:

        def middleware(request: HttpRequest) -> HttpResponse:
            response = get_response(request)
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
def sentry_user_middleware(get_response):
    if iscoroutinefunction(get_response):

        async def middleware(request: HttpRequest) -> HttpResponse:
            if hasattr(request, "user"):
                is_authenticated = await sync_to_async(getattr)(request.user, "is_authenticated", False)
                if is_authenticated:
                    await sync_to_async(sentry_sdk.set_user)(
                        {
                            "id": request.user.id,
                            "email": request.user.email,
                            "name": request.user.name,
                        }
                    )
            response = await get_response(request)
            if asyncio.iscoroutine(response):
                response = await response
            return response
    else:

        def middleware(request: HttpRequest) -> HttpResponse:
            if hasattr(request, "user") and request.user.is_authenticated:
                sentry_sdk.set_user(
                    {
                        "id": request.user.id,
                        "email": request.user.email,
                        "name": request.user.name,
                    }
                )
            return get_response(request)

    return middleware


class SSOSyncMiddleware:
    SESSION_KEY = "sso_last_synced"
    SYNC_INTERVAL = timedelta(hours=1).total_seconds()

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            if flag_is_active(request, flags.RESET_SSO_SYNC_SESSION):
                request.session.pop(self.SESSION_KEY, None)

            last_synced = request.session.get(self.SESSION_KEY)
            now_timestamp = timezone.now().timestamp()

            should_sync = not last_synced or (now_timestamp - last_synced) > self.SYNC_INTERVAL

            if should_sync:
                synced = sso_service.sync_sso_data(request)

                if synced:
                    request.session[self.SESSION_KEY] = now_timestamp

        return self.get_response(request)

from authlib.integrations.requests_client import OAuth2Session
from django.conf import settings
from django.urls import reverse

JWT_SEGMENT_COUNT = 3


def is_jwt(token: str | None) -> bool:
    segments = token.split(".") if isinstance(token, str) else []
    return len(segments) == JWT_SEGMENT_COUNT and all(segments)


def get_client(request, **kwargs):
    return OAuth2Session(
        client_id=settings.AUTHBROKER_CLIENT_ID,
        client_secret=settings.AUTHBROKER_CLIENT_SECRET,
        redirect_uri=request.build_absolute_uri(reverse("auth:callback")),
        scope=settings.AUTHBROKER_SCOPE,
        token=request.session.get(
            settings.TOKEN_SESSION_KEY,
            None,
        ),
        **kwargs,
    )

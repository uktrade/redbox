import time

from authlib.common.encoding import json_loads, urlsafe_b64decode
from authlib.integrations.requests_client import OAuth2Session
from django.conf import settings
from django.urls import reverse

JWT_SEGMENT_COUNT = 3


def is_valid_jwt(token: str | None) -> bool:
    """Check the token has JWT structure and, if its payload can be decoded, that it hasn't expired.

    Note: this does not verify the signature, so it should not be relied on as a security boundary.
    """
    segments = token.split(".") if isinstance(token, str) else []
    if len(segments) != JWT_SEGMENT_COUNT or not all(segments):
        return False

    try:
        payload = json_loads(urlsafe_b64decode(segments[1]))
    except (ValueError, TypeError):
        # payload isn't decodable (e.g. an opaque token used by test/mock SSO servers)
        return True

    exp = payload.get("exp")
    return exp is None or (isinstance(exp, (int, float)) and exp > time.time())


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

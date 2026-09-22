import logging
import uuid

from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.core.exceptions import SuspiciousOperation
from django.shortcuts import redirect
from django.views import View
from django.views.generic import RedirectView

from .utils import get_client

logger = logging.getLogger(__name__)


class AuthView(RedirectView):
    def get_redirect_url(self, *_args, **_kwargs):
        url, state = get_client(self.request).create_authorization_url(
            settings.AUTHBROKER_AUTHORIZATION_URL,
            nonce=uuid.uuid4().hex,
        )

        self.request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_state"] = state

        return url


class AuthCallbackView(View):
    def get(self, request, *_args, **_kwargs):
        auth_code = request.GET.get("code")

        if not auth_code:
            return redirect(settings.LOGIN_URL)

        state = request.session.get(f"{settings.TOKEN_SESSION_KEY}_oauth_state")

        if not state:
            msg = "No state found in session"
            raise SuspiciousOperation(msg)

        returned_state = request.GET.get("state")

        if state != returned_state:
            msg = "Session state and passed back state differ"
            raise SuspiciousOperation(msg)

        token = self.fetch_token(request, auth_code)

        request.session[settings.TOKEN_SESSION_KEY] = dict(token)

        del request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_state"]

        profile = self.get_profile(request)

        return self.authenticate_user(request, profile)

    def fetch_token(self, request, auth_code):
        client = get_client(request)

        return client.fetch_token(
            url=settings.AUTHBROKER_TOKEN_URL,
            code=auth_code,
            client_secret=settings.AUTHBROKER_CLIENT_SECRET,
            grant_type="authorization_code",
            client_id=settings.AUTHBROKER_CLIENT_ID,
        )

    def get_profile(self, request):
        response = get_client(request).get(settings.AUTHBROKER_PROFILE_URL)
        response.raise_for_status()
        return response.json()

    def authenticate_user(self, request, profile):
        email = profile.get("email")

        if not email:
            msg = "SSO profile missing email claim"
            raise SuspiciousOperation(msg)

        user_model = get_user_model()
        user, created = user_model.objects.get_or_create(
            username=email,
            defaults={
                "email": email,
                "first_name": profile.get("given_name", ""),
                "last_name": profile.get("family_name", ""),
            },
        )

        if created:
            user.set_unusable_password()
            user.save()

        login(request, user, backend="django.contrib.auth.backends.ModelBackend")

        return redirect(settings.LOGIN_REDIRECT_URL)

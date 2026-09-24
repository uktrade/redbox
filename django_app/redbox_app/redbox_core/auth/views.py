import logging
import uuid
from http import HTTPStatus

from authlib.common.encoding import json_loads, urlsafe_b64decode
from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.core.exceptions import SuspiciousOperation
from django.db.models import Q
from django.shortcuts import redirect
from django.views import View
from django.views.generic import RedirectView

from redbox_app.redbox_core.models import UserSSOAttribute

from .utils import get_client, is_valid_jwt

logger = logging.getLogger(__name__)


class AuthView(RedirectView):
    def get_redirect_url(self, *_args, **_kwargs):
        nonce = uuid.uuid4().hex
        url, state = get_client(self.request).create_authorization_url(
            settings.AUTHBROKER_AUTHORIZATION_URL,
            nonce=nonce,
        )

        self.request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_state"] = state
        self.request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_nonce"] = nonce

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

        nonce = request.session.get(f"{settings.TOKEN_SESSION_KEY}_oauth_nonce")

        if not nonce:
            msg = "No nonce found in session"
            raise SuspiciousOperation(msg)

        token = self.fetch_token(request, auth_code)

        request.session[settings.TOKEN_SESSION_KEY] = dict(token)

        del request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_state"]
        del request.session[f"{settings.TOKEN_SESSION_KEY}_oauth_nonce"]

        self.validate_nonce(token, nonce)

        profile = self.get_profile(request)

        return self.authenticate_user(request, profile)

    def validate_nonce(self, token, expected_nonce):
        id_token = token.get("id_token")

        if not is_valid_jwt(id_token):
            msg = "no valid id_token returned to validate nonce"
            raise SuspiciousOperation(msg)

        try:
            payload = json_loads(urlsafe_b64decode(id_token.split(".")[1]))
        except (ValueError, TypeError) as exc:
            msg = "cant decode id_token to validate nonce"
            raise SuspiciousOperation(msg) from exc

        if payload.get("nonce") != expected_nonce:
            msg = "id_token nonce doesnt match session nonce"
            raise SuspiciousOperation(msg)

    def fetch_token(self, request, auth_code):
        client = get_client(request)

        return client.fetch_token(
            url=settings.AUTHBROKER_TOKEN_URL,
            code=auth_code,
            grant_type="authorization_code",
        )

    def get_profile(self, request):
        client = get_client(request)
        response = client.get(settings.AUTHBROKER_PROFILE_URL)
        if response.status_code == HTTPStatus.NOT_FOUND:
            # mock-sso doesn't use /o/userinfo/
            response = client.get(settings.AUTHBROKER_LEGACY_PROFILE_URL)
        response.raise_for_status()
        return response.json()

    def authenticate_user(self, request, profile):
        email = profile.get("email")

        if not email:
            msg = "SSO profile missing email claim"
            raise SuspiciousOperation(msg)

        user_model = get_user_model()
        user = self.find_existing_user(user_model, profile, email)
        created = user is None

        if created:
            user = user_model.objects.create(
                username=email,
                email=email,
                first_name=profile.get("given_name") or profile.get("first_name", ""),
                last_name=profile.get("family_name") or profile.get("last_name", ""),
            )
            user.set_unusable_password()
            user.save()

        login(request, user, backend="django.contrib.auth.backends.ModelBackend")

        return redirect(settings.LOGIN_REDIRECT_URL)

    def find_existing_user(self, user_model, profile, email):
        sso_identity = profile.get("email_user_id")
        if sso_identity:
            # the usernames used to be set to the SSO email_user_id for MOCK_SSO_USERNAME
            user = user_model.objects.filter(
                Q(_sso__email_user_id__iexact=sso_identity) | Q(username__iexact=sso_identity)
            ).first()
            if user:
                return user

        email_fields = Q(email__iexact=email) | Q(username__iexact=email)
        for field in ("email", "contact_email"):
            email_fields |= Q(**{f"_sso__{field}__iexact": email})
        email_fields |= Q(
            _sso__attributes__attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
            _sso__attributes__value__iexact=email,
        )
        user = user_model.objects.filter(email_fields).distinct().first()
        if user:
            return user

        local_part = email.split("@", 1)[0].casefold()
        first_name = (profile.get("given_name") or "").casefold()
        last_name = (profile.get("family_name") or "").casefold()
        if local_part:
            return user_model.objects.filter(
                username__istartswith=f"{local_part}@",
                first_name__iexact=first_name,
                last_name__iexact=last_name,
            ).first()

        return None

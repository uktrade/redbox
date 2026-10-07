import logging
import uuid
from http import HTTPStatus

from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.core.exceptions import SuspiciousOperation
from django.db.models import Q
from django.shortcuts import redirect
from django.views import View
from django.views.generic import RedirectView

from redbox_app.redbox_core.models import UserSSOAttribute

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

        if created:
            user.set_unusable_password()
            user.save()

        login(request, user, backend="django.contrib.auth.backends.ModelBackend")

        return redirect(settings.LOGIN_REDIRECT_URL)

    def find_existing_user(self, user_model, profile, email):
        sso_identity = profile.get("email_user_id")
        if sso_identity:
            user = user_model.objects.filter(_sso__email_user_id__iexact=sso_identity).first()
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
        if local_part and first_name and last_name:
            return user_model.objects.filter(
                username__istartswith=f"{local_part}@",
                first_name__iexact=first_name,
                last_name__iexact=last_name,
            ).first()

        return None

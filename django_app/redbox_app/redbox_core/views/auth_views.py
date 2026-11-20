import logging
from urllib.parse import urlencode

from django.contrib.auth import get_user_model, logout
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

from redbox_app.redbox_core.services.sso import get_usable_sso_token

logger = logging.getLogger(__name__)
User = get_user_model()


def sign_in_view(request: HttpRequest):
    if request.user.is_authenticated:
        if not get_usable_sso_token(request.session):
            return redirect("reauthenticate")

        return redirect("chats")

    return redirect("authbroker_client:login")


def sign_in_link_sent_view(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("homepage")
    return render(
        request,
        template_name="sign-in-link-sent.html",
        context={"request": request},
    )


def signed_out_view(request: HttpRequest):
    logout(request)
    return redirect("homepage")


def reauthenticate_view(request: HttpRequest):
    next_url = request.GET("next") or reverse("chats")

    if not (
        next_url.startswith("/")
        and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts=set(),
        )
    ):
        next_url = reverse("chats")

    logout(request)
    login_url = reverse("authbroker_client:login")
    return redirect(f"{login_url}?{urlencode({'next': next_url})}")

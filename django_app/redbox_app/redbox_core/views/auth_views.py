import logging

from django.conf import settings
from django.contrib.auth import get_user_model, logout
from django.http import HttpRequest
from django.shortcuts import redirect, render

logger = logging.getLogger(__name__)
User = get_user_model()


def sign_in_view(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("homepage")
    if settings.LOGIN_METHOD == "sso":
        return redirect("/auth/login/")
    return render(
        request,
        template_name="sign-in.html",
        context={"request": request},
    )


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

from django.urls import path

from . import views

app_name = "auth"

urlpatterns = [
    path("login/", views.AuthView.as_view(), name="login"),
    path("callback/", views.AuthCallbackView.as_view(), name="callback"),
]

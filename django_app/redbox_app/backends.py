from authbroker_client.backends import AuthbrokerBackend


class TokenCaptureBackend(AuthbrokerBackend):
    def authenticate(self, request, **kwargs):
        # 1. Grab the token from the arguments passed by the broker
        token = kwargs.get("token")

        # 2. Store it directly in the session
        if request and token:
            request.session["oauth_token"] = token

        # 3. Carry on with the normal login process
        return super().authenticate(request, **kwargs)

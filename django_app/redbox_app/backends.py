import logging

logger = logging.getLogger(__name__)

# class TokenCaptureBackend(AuthbrokerBackend):
#     def authenticate(self, request, **kwargs):
#         token = kwargs.get("token")

#         if token:
#             logger.warning(
#                 "SSO token type=%s length=%s jwt_parts=%s",
#                 type(token).__name__,
#                 len(token),
#                 len(token.split(".")) if isinstance(token, str) else None,
#             )

#         if request and token:
#             request.session["oauth_token"] = token

#         return super().authenticate(request, **kwargs)

# mypy: ignore-errors
import logging
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import environ
import sentry_sdk
from dbt_copilot_python.database import database_from_env
from django.urls import reverse_lazy
from django_log_formatter_asim import ASIMFormatter
from dotenv import find_dotenv, load_dotenv
from import_export.formats.base_formats import CSV
from sentry_sdk.integrations.django import DjangoIntegration
from storages.backends import s3boto3
from yarl import URL

from redbox_app.setting_enums import Classification, Environment

logger = logging.getLogger(__name__)


load_dotenv()

if os.getenv("USE_LOCAL_ENV", "False").lower() == "true":
    load_dotenv(find_dotenv(".env.local"), override=True)

env = environ.Env()

ALLOW_SIGN_UPS = env.bool("ALLOW_SIGN_UPS")

SECRET_KEY = env.str("DJANGO_SECRET_KEY")
ENVIRONMENT = Environment[env.str("ENVIRONMENT").upper()]
WEBSOCKET_SCHEME = "ws" if ENVIRONMENT.is_test else "wss"

# env variables used by redbox_core
COLLECTION_ENDPOINT = env.str("COLLECTION_ENDPOINT")
ALLOW_PLAN_FEEDBACK = env.bool("ALLOW_PLAN_FEEDBACK", True)
MCP_CADDY_URL = env.str("MCP_CADDY_URL", "")
MCP_HEADERS = env.str("MCP_HEADERS", "")
MCP_CADDY_TOKEN = env.str("MCP_CADDY_TOKEN", "")
MCP_PARLEX_URL = env.str("MCP_PARLEX_URL", "")
MCP_PARLEX_TOKEN = env.str("MCP_PARLEX_TOKEN", "")
GOOGLE_SEARCH_API = env.str("GOOGLE_SEARCH_API", "")
GOOGLE_SEARCH_ENGINE = env.str("GOOGLE_SEARCH_ENGINE", "")
BRAVE_API_KEY = env.str("BRAVE_API_KEY", "")
KAGI_API_KEY = env.str("KAGI_API_KEY", "")
MAX_ATTEMPTS = env.int("max_attempts", 3)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env.bool("DEBUG")

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_URL = "static/"
STATIC_ROOT = "staticfiles/"
STATICFILES_DIRS = [
    Path(BASE_DIR) / "static/",
    Path(BASE_DIR) / "frontend/dist/",
]
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]


SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Application definition
INSTALLED_APPS = [
    "authbroker_client",
    "daphne",
    "redbox_app.redbox_core",
    "django.contrib.admin.apps.SimpleAdminConfig",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.sites",
    "django.contrib.staticfiles",
    "single_session",
    "storages",
    "import_export",
    "django_q",
    "rest_framework",
    "rest_framework.authtoken",
    "django_plotly_dash.apps.DjangoPlotlyDashConfig",
    "adminplus",
    "waffle",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "redbox_app.redbox_core.middleware.plotly_no_csp_no_xframe_middleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_permissions_policy.PermissionsPolicyMiddleware",
    "csp.middleware.CSPMiddleware",
    "redbox_app.redbox_core.middleware.nocache_middleware",
    "redbox_app.redbox_core.middleware.security_header_middleware",
    "django_plotly_dash.middleware.BaseMiddleware",
    "waffle.middleware.WaffleMiddleware",
    "redbox_app.redbox_core.middleware.sentry_user_middleware",
]

ROOT_URLCONF = "redbox_app.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.jinja2.Jinja2",
        "DIRS": [
            BASE_DIR / "redbox_app" / "templates",
            BASE_DIR / "redbox_app" / "templates" / "auth",
            BASE_DIR / "frontend" / "src" / "redbox_design_system",
        ],
        "OPTIONS": {
            "environment": "redbox_app.jinja2.environment",
        },
    },
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            BASE_DIR / "redbox_app" / "templates",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "redbox_app.wsgi.application"
ASGI_APPLICATION = "redbox_app.asgi.application"

AUTHENTICATION_BACKENDS = ["django.contrib.auth.backends.ModelBackend", "authbroker_client.backends.AuthbrokerBackend"]

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {
            "min_length": 10,
        },
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


LANGUAGE_CODE = "en-GB"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
SITE_ID = 1
AUTH_USER_MODEL = "redbox_core.User"
ACCOUNT_EMAIL_VERIFICATION = "none"

AUTHBROKER_URL = env.str("AUTHBROKER_URL", "https://example.com")
AUTHBROKER_CLIENT_ID = env.str("AUTHBROKER_CLIENT_ID", "1234")
AUTHBROKER_CLIENT_SECRET = env.str("AUTHBROKER_CLIENT_SECRET", "1234")
LOGIN_URL = reverse_lazy("authbroker_client:login")
LOGIN_REDIRECT_URL = reverse_lazy("homepage")

# CSP settings https://content-security-policy.com/
# https://django-csp.readthedocs.io/
CSP_DEFAULT_SRC = (
    "'self'",
    "s3.amazonaws.com",
    "https://www.google-analytics.com/",
    "https://region1.google-analytics.com/",
    "https://www.googletagmanager.com/",
)

CSP_SCRIPT_SRC = (
    "'self'",
    "'sha256-RfLASrooywwZYqv6kr3TCnrZzfl6ZTfbpLBJOVR/Gt4='",
    "'sha256-GUQ5ad8JK5KmEWmROf3LZd9ge94daqNvd8xy9YS1iDw='",
    "'sha256-qmCu1kQifDfCnUd+L49nusp7+PeRl23639pzN5QF2WA='",
    "'sha256-1NTuHcjvzzB6D69Pb9lbxI5pMJNybP/SwBliv3OvOOE='",
    "'sha256-DrkvIvFj5cNADO03twE83GwgAKgP224E5UyyxXFfvTc='",
    "'sha256-6BIGXagXVUHOQ8pw9flNwo/urWufeay+hbx+Q+U6/DM='",  # pragma: allowlist secret
    "'sha256-SgZQWsfLqFIbXUavZS4pxgi9Pr0JFuIh5pAp0LdrHPU='",  # pragma: allowlist secret
    "'sha256-XAYqMHKLikdz4TwKJfz53n9YgPx3M6F2KB98Dxf/A5c='",  # pragma: allowlist secret
    "https://*.googletagmanager.com",
    "https://tagmanager.google.com/",
    "https://www.googletagmanager.com/",
    "https://www.google-analytics.com/",
    "ajax.googleapis.com/",
    "sha256-T/1K73p+yppfXXw/AfMZXDh5VRDNaoEh3enEGFmZp8M=",
)
CSP_OBJECT_SRC = ("'none'",)
CSP_TRUSTED_TYPES = ("dompurify", "default", "goog#html")
CSP_REPORT_TO = "csp-endpoint"
CSP_FONT_SRC = ("'self'", "s3.amazonaws.com", "https://fonts.gstatic.com", "data:")
CSP_INCLUDE_NONCE_IN = ("script-src",)
CSP_STYLE_SRC = (
    "'self'",
    "https://googletagmanager.com",
    "https://tagmanager.google.com/",
    "https://fonts.googleapis.com",
)

CSP_IMG_SRC = (
    "'self'",
    "https://googletagmanager.com",
    "https://ssl.gstatic.com",
    "https://www.gstatic.com",
    "https://*.google-analytics.com",
    "https://*.googletagmanager.com",
)
CSP_FRAME_ANCESTORS = ("'none'",)


CSP_CONNECT_SRC = [
    "'self'",
    f"{WEBSOCKET_SCHEME}://{ENVIRONMENT.hosts[0]}/ws/chat/",
    "https://*.google-analytics.com",
    "https://*.analytics.google.com",
    "https://*.googletagmanager.com",
    "https://www.google-analytics.com/",
    "https://region1.google-analytics.com/",
    "https://www.googletagmanager.com/",
    "wss://transcribestreaming.eu-west-2.amazonaws.com:8443",
]


for csp in CSP_CONNECT_SRC:
    logger.info("CSP=%s", csp)


# https://pypi.org/project/django-permissions-policy/
PERMISSIONS_POLICY: dict[str, list] = {
    "accelerometer": [],
    "autoplay": [],
    "camera": [],
    "display-capture": [],
    "encrypted-media": [],
    "fullscreen": [],
    "gamepad": [],
    "geolocation": [],
    "gyroscope": [],
    "microphone": ["self"],
    "midi": [],
    "payment": [],
}

CSRF_COOKIE_HTTPONLY = True

SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_AGE = 60 * 60 * 24
SESSION_ENGINE = "django.contrib.sessions.backends.db"

LOG_ROOT = "."
LOG_HANDLER = "asim"
BUCKET_NAME = env.str("BUCKET_NAME")
AWS_S3_REGION_NAME = env.str("AWS_REGION")
APPEND_SLASH = True

#  Property added to each S3 file to make them downloadable by default
AWS_S3_OBJECT_PARAMETERS = {"ContentDisposition": "attachment"}
AWS_STORAGE_BUCKET_NAME = BUCKET_NAME  # this duplication is required for django-storage
OBJECT_STORE = env.str("OBJECT_STORE")

STORAGES = {
    "default": {
        "BACKEND": s3boto3.S3Boto3Storage,
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

if ENVIRONMENT.uses_minio:
    AWS_S3_SECRET_ACCESS_KEY = env.str("AWS_SECRET_KEY")
    AWS_ACCESS_KEY_ID = env.str("AWS_ACCESS_KEY")
    MINIO_HOST = env.str("MINIO_HOST")
    MINIO_PORT = env.str("MINIO_PORT")
    MINIO_ENDPOINT = f"http://{MINIO_HOST}:{MINIO_PORT}"
    AWS_S3_ENDPOINT_URL = MINIO_ENDPOINT
else:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Strict-Transport-Security
    # Mozilla guidance max-age 2 years
    SECURE_HSTS_SECONDS = 2 * 365 * 24 * 60 * 60
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SESSION_COOKIE_SECURE = True

if ENVIRONMENT.is_test:
    ALLOWED_HOSTS = ENVIRONMENT.hosts
else:
    LOCALHOST = socket.gethostbyname(socket.gethostname())
    ALLOWED_HOSTS = ["*"]

if not ENVIRONMENT.is_local:

    def filter_transactions(event, _hint):
        url_string = event["request"]["url"]
        parsed_url = urlparse(url_string)
        if parsed_url.path.startswith("/admin"):
            return None
        return event

    SENTRY_DSN = env.str("SENTRY_DSN", None)
    SENTRY_ENVIRONMENT = env.str("SENTRY_ENVIRONMENT", None)
    if SENTRY_DSN and SENTRY_ENVIRONMENT:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[
                DjangoIntegration(),
            ],
            environment=SENTRY_ENVIRONMENT,
            send_default_pii=True,
            traces_sample_rate=1.0,
            before_send_transaction=filter_transactions,
            debug=False,
        )
SENTRY_REPORT_TO_ENDPOINT = URL(env.str("SENTRY_REPORT_TO_ENDPOINT", "")) or None

database_credentials = os.getenv("DATABASE_CREDENTIALS")
if database_credentials:
    DATABASES = database_from_env("DATABASE_CREDENTIALS")
    DATABASES["default"]["ENGINE"] = "django.db.backends.postgresql"

else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env.str("POSTGRES_DB"),
            "USER": env.str("POSTGRES_USER"),
            "PASSWORD": env.str("POSTGRES_PASSWORD"),
            "HOST": env.str("POSTGRES_HOST"),
            "PORT": "5432",
        }
    }

LOG_LEVEL = env.str("DJANGO_LOG_LEVEL", "WARNING")
LOG_FORMAT = env.str("DJANGO_LOG_FORMAT", "asim_formatter")
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "%(asctime)s %(levelname)s %(module)s: %(message)s"},
        "asim_formatter": {
            "()": ASIMFormatter,
        },
    },
    "filters": {
        "exclude_s3_urls_and_emails": {
            "": "django.utils.log.CallbackFilter",
            "callback": lambda record: (
                all(
                    header not in record.getMessage()
                    for header in ["X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Security-Token"]
                )
                and not re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", record.getMessage())
            )
            if hasattr(record, "getMessage")
            else True,
        },
    },
    "handlers": {
        "console": {
            "level": LOG_LEVEL,
            "class": "logging.StreamHandler",
            "formatter": LOG_FORMAT,
            "filters": ["exclude_s3_urls_and_emails"],
        },
        "asim": {
            "level": "ERROR",
            "class": "logging.StreamHandler",
            "formatter": "asim_formatter",
        },
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "application": {
            "handlers": [LOG_HANDLER],
            "level": LOG_LEVEL,
            "propagate": True,
        },
        "boto3": {
            "level": "WARNING",
        },
        "botocore": {
            "level": "WARNING",
        },
        "s3transfer": {
            "level": "WARNING",
        },
        "ddtrace": {
            "handlers": ["asim"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}


# Email
EMAIL_BACKEND_TYPE = env.str("EMAIL_BACKEND_TYPE")
FROM_EMAIL = env.str("FROM_EMAIL")
CONTACT_EMAIL = env.str("CONTACT_EMAIL")

if EMAIL_BACKEND_TYPE == "FILE":
    EMAIL_BACKEND = "django.core.mail.backends.filebased.EmailBackend"
    EMAIL_FILE_PATH = env.str("EMAIL_FILE_PATH")
elif EMAIL_BACKEND_TYPE == "CONSOLE":
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
elif EMAIL_BACKEND_TYPE == "GOVUKNOTIFY":
    EMAIL_BACKEND = "django_gov_notify.backends.NotifyEmailBackend"

else:
    message = f"Unknown EMAIL_BACKEND_TYPE of {EMAIL_BACKEND_TYPE}"
    raise ValueError(message)

# Notifications
GOVUK_NOTIFY_API_KEY = env.str("GOVUK_NOTIFY_API_KEY", None)
GOVUK_NOTIFY_PLAIN_EMAIL_TEMPLATE_ID = env.str("GOVUK_NOTIFY_PLAIN_EMAIL_TEMPLATE_ID", None)
GOVUK_NOTIFY_TEAM_ADDITION_EMAIL_TEMPLATE_ID = env.str("GOVUK_NOTIFY_TEAM_ADDITION_EMAIL_TEMPLATE_ID", None)

IMPORT_FORMATS = [CSV]

CHAT_TITLE_LENGTH = 30
FILE_EXPIRY_IN_SECONDS = env.int("FILE_EXPIRY_IN_DAYS") * 24 * 60 * 60
SUPERUSER_EMAIL = env.str("SUPERUSER_EMAIL", None)
MAX_SECURITY_CLASSIFICATION = Classification[env.str("MAX_SECURITY_CLASSIFICATION")]

SECURITY_TXT_REDIRECT = URL("https://vdp.cabinetoffice.gov.uk/.well-known/security.txt")
REDBOX_VERSION = os.environ.get("REDBOX_VERSION", "not set")

Q_CLUSTER = {
    "name": "redbox_django",
    "recycle": env.int("Q_RECYCLE", 500),
    "timeout": env.int("Q_TIMEOUT", 600),
    "retry": env.int("Q_RETRY", 60),
    "max_attempts": env.int("Q_MAX_ATTEMPTS", 3),
    "catch_up": False,
    "orm": "default",
    "workers": env.int("Q_WORKERS", 5),
    "error_reporter": {"sentry": {"dsn": env.str("SENTRY_DSN", " ")}},
}

UNSTRUCTURED_HOST = env.str("UNSTRUCTURED_HOST")

GOOGLE_ANALYTICS_TAG = env.str("GOOGLE_ANALYTICS_TAG", " ")
GOOGLE_ANALYTICS_LINK = env.str("GOOGLE_ANALYTICS_LINK", " ")
GOOGLE_ANALYTICS_IFRAME_SRC = env.str("GOOGLE_ANALYTICS_IFRAME_SRC", " ")
# TEST_SSO_PROVIDER_SET_RETURNED_ACCESS_TOKEN = 'someCode'

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "redbox_app.redbox_core.middleware.APIKeyAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}


REDBOX_API_KEY = env.str("REDBOX_API_KEY")

ENABLE_METADATA_EXTRACTION = env.bool("ENABLE_METADATA_EXTRACTION", default=True)

CHUNK_UPLOADER_AWS_REGION = env.str("AWS_REGION", " ")

AWS_TRANSCRIBE_ROLE_ARN = env.str("AWS_TRANSCRIBE_ROLE_ARN", "")

DATAHUB_REDBOX_URL = env.str("DATAHUB_REDBOX_URL", "")
DATAHUB_REDBOX_SECRET_KEY = env.str("DATAHUB_REDBOX_SECRET_KEY", "")
DATAHUB_REDBOX_ACCESS_KEY_ID = env.str("DATAHUB_REDBOX_ACCESS_KEY_ID", "")

EMBEDDING_BACKEND = env.str("EMBEDDING_BACKEND", "amazon.titan-embed-text-v2:0")

DEFAULT_MODEL_ID = env.str("DEFAULT_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

WEB_SEARCH_API_LIMIT = env.int("WEB_SEARCH_API_LIMIT", 100)

ADMIN_EMAIL = env.str("ADMIN_EMAIL", "")

FEEDBACK_LINK = env.str(
    "FEEDBACK_LINK",
    "https://teams.microsoft.com/l/channel/19%3A9ae6b3b539724595a3139c2b16dc56ef%40thread.tacv2/Redbox%20trial%20participants%20Chat%20Channel?groupId=7a71ce78-fe77-4185-825c-ae40cb07d614&tenantId=8fa217ec-33aa-46fb-ad96-dfe68006bb86",
)

PRODUCT_NAME = env.str("PRODUCT_NAME", "Redbox at DBT")

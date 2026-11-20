import os

from dotenv import load_dotenv
from storages.backends import s3boto3

load_dotenv(override=True)
INSTALLED_APPS = [
    "django_q",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.postgres",
    "redbox",
    "storages",
]


STORAGES = {
    "default": {
        "BACKEND": s3boto3.S3Boto3Storage,
    },
}

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB"),
        "USER": os.environ.get("POSTGRES_USER"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD"),
        "HOST": os.environ.get("POSTGRES_HOST"),
        "PORT": "5432",
    }
}

Q_CLUSTER = {
    "name": "redbox_django",
    "timeout": 1200,
    "retry": 1400,
    "max_attempts": 3,
    "catch_up": False,
    "orm": "default",
    "workers": 1,
}

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")

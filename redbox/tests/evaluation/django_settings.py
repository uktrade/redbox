"""
Minimal Django settings for the retrieval eval harness.

Satisfies Django's import chain (waffle -> django.http -> django.core.checks.caches)
without requiring a database or full redbox_app configuration.
DocumentExtractionService's cache lock uses locmem so it always succeeds locally.
"""
SECRET_KEY = "eval-harness-test-key"
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "waffle",
]
DATABASES = {}
CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

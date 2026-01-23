import logging
import sys

import django
from django.contrib.auth import get_user_model

django.setup()

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

username = input("Enter your SSO ID: ").strip()
if not username:
    logger.info("No SSO ID entered so exiting")
    sys.exit(1)

User = get_user_model()
try:
    user = User.objects.get(username=username)
except User.DoesNotExist:
    try:
        user = User.objects.get(email=username)
    except User.DoesNotExist:
        logger.info("User with that SSO ID has not been found.")
        sys.exit(1)

user.is_staff = True
user.is_superuser = True
user.save()
logger.info("Success you are now a superuser")

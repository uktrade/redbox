#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "redbox_app.settings")
    logger = logging.getLogger(__name__)
    if os.environ.get("DEBUGPY") == "1" and os.environ.get("RUN_MAIN") == "true":
        import debugpy  # noqa: PLC0415 T100

        debugpy.listen(("0.0.0.0", 5678))  # noqa: S104 T100
        logger.info("debugpy listening on 5678")

    load_dotenv(override=True)
    logger.info("Completed loading from dotenv")
    if os.getenv("USE_LOCAL_ENV", "False").lower() == "true":
        logger.info("Loading Local EnvFile: .env.local")
        load_dotenv(find_dotenv(".env.local"), override=True)

    try:
        from django.core.management import execute_from_command_line  # noqa: PLC0415
    except ImportError as exc:
        message = (
            "Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH environment "
            "variable? Did you forget to activate a virtual environment?"
        )
        raise ImportError(message) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

#!/bin/sh

PORT=${PORT:-8080}

/usr/src/app/venv/bin/django-admin migrate
/usr/src/app/venv/bin/django-admin collectstatic --noinput
/usr/src/app/venv/bin/django-admin create_admin_user

echo "Starting daphne on port $PORT"
/usr/src/app/venv/bin/ddtrace-run \
    /usr/src/app/venv/bin/daphne \
    --websocket_timeout 86400 \
    -b 0.0.0.0 \
    -p "$PORT" \
    redbox_app.asgi:application
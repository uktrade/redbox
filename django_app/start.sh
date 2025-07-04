#!/bin/sh

PORT=8080

venv/bin/django-admin migrate
venv/bin/django-admin collectstatic --noinput
venv/bin/django-admin create_admin_user

echo "Starting daphne on port $PORT"
#venv/bin/daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application
ddtrace-run venv/bin/daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application
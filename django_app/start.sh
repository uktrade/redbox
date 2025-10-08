#!/bin/sh

PORT=8080

venv/bin/python manage.py migrate
venv/bin/python manage.py collectstatic --noinput
venv/bin/python manage.py create_admin_user

echo "Starting daphne on port $PORT"
venv/bin/ddtrace-run venv/bin/daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application
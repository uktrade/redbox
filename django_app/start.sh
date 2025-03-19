#!/bin/sh

PORT=${PORT:-8080}

venv/bin/django-admin migrate
venv/bin/django-admin collectstatic --noinput
venv/bin/django-admin create_admin_user

if [ "$ENVIRONMENT" = "LOCAL" ]; then
    echo "Starting daphne on port $PORT (local - auto detect changes)"
    venv/bin/uvicorn redbox_app.asgi:application --host 0.0.0.0 --port $PORT --reload --log-level debug
else
    echo "Starting daphne on port $PORT"
    venv/bin/daphne -b 0.0.0.0 -p $PORT redbox_app.asgi:application
fi
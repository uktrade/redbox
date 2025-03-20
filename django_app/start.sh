#!/bin/sh

PORT=${PORT:-8080}

venv/bin/django-admin migrate
venv/bin/django-admin collectstatic --noinput
venv/bin/django-admin create_admin_user

if [ "$ENVIRONMENT" = "LOCAL" ]; then
    echo "Starting uvicorn on port $PORT (local - auto detect changes)"
    venv/bin/watchfiles \
        --filter python \
        "pkill -f 'uvicorn' && venv/bin/uvicorn --host 0.0.0.0 --port $PORT redbox_app.asgi:application --log-level debug" \
        ./django_app &
    venv/bin/uvicorn --host 0.0.0.0 --port "$PORT" --log-level debug redbox_app.asgi:application
else
    echo "Starting daphne on port $PORT"
    venv/bin/daphne -b 0.0.0.0 -p "$PORT" redbox_app.asgi:application
fi

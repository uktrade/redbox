#!/bin/sh

PORT=${PORT:-8080}
if [["$ENVIRONMENT" == "LOCAL" ]]; then
    PORT=8090
fi

venv/bin/django-admin migrate
venv/bin/django-admin collectstatic --noinput
venv/bin/django-admin create_admin_user

echo "Starting daphne on port $PORT"
venv/bin/daphne -b 0.0.0.0 -p $PORT redbox_app.asgi:application
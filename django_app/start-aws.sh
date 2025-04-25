#!/bin/sh

PORT=8080

python manage.py migrate
python manage.py collectstatic --noinput
python manage.py create_admin_user

echo "Starting daphne on port $PORT"
daphne -b 0.0.0.0 -p $PORT redbox_app.asgi:application

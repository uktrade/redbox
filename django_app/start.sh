#!/bin/sh
PORT=${PORT:-8080}

django-admin migrate

if [ "$ENVIRONMENT" = "local" ]; then
  exec python manage.py runserver 0.0.0.0:$PORT
fi

django-admin collectstatic --noinput
django-admin create_admin_user

echo "Starting daphne on port $PORT"
exec ddtrace-run daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application

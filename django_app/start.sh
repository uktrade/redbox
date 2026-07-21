#!/bin/sh

PORT=8080

if [ "${REDBOX_ENABLE_DD_DIAGNOSTICS:-false}" = "1" ] || [ "${REDBOX_ENABLE_DD_DIAGNOSTICS:-false}" = "true" ] || [ "${REDBOX_ENABLE_DD_DIAGNOSTICS:-false}" = "TRUE" ]; then
	export DD_TRACE_DEBUG=true
	export DD_TRACE_LOG_LEVEL=DEBUG
	export REDBOX_BEDROCK_DIAGNOSTICS=true
	echo "Datadog diagnostics mode enabled"
fi

venv/bin/django-admin migrate
venv/bin/django-admin collectstatic --noinput
venv/bin/django-admin create_admin_user

echo "Starting daphne on port $PORT"
#venv/bin/daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application
venv/bin/ddtrace-run venv/bin/daphne --websocket_timeout 86400 -b 0.0.0.0 -p $PORT redbox_app.asgi:application

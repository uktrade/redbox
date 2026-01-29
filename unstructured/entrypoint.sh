#!/bin/sh
set -e

echo "Started background cleanup"
(
    while true; do
        /usr/local/bin/cleanup.sh || true
        sleep 60
    done
) &

if [ "${ENV}" = "local" ] || [ "${IS_LOCAL}" = "true" ] || [ "${LOCAL_DEV}" = "1" ]; then
    PORT=8000
    echo "Running in local mode so the port is ${PORT}"
else
    PORT=8080
    echo "Running in ECS so the port is ${PORT}"
fi

echo "Started uvicorn with unstructured io"

exec uvicorn prepline_general.api.app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-config logger_config.yaml

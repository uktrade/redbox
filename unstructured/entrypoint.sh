#!/bin/sh
set -e

echo "Started background cleanup"
(
    while true; do
        /usr/local/bin/cleanup.sh || true
        sleep 60
    done
) &

echo "Started uvicorn with unstructured io"

exec uvicorn prepline_general.api.app:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-config logger_config.yaml
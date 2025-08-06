#!/bin/bash
set -e

# Ensure environment variables are loaded
if [ -f /app/.env ]; then
    echo "Loading .env file"
    export $(grep -v '^#' /app/.env | xargs)
fi

# Run migrations
poetry run django-admin migrate

# Start FastAPI
poetry run uvicorn main:app --host 0.0.0.0 --port 8001
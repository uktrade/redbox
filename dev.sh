#!/bin/bash

# Trap Ctrl+C for cleanup
trap 'echo "Cleaning up..."; docker compose down; kill %1; exit' INT

echo "Running migrations..."
make migrations

echo "Applying migrations..."
make migrate

echo "Starting Docker containers..."
docker compose up -d --wait db opensearch minio worker

echo "Building static files..."
make build-django-static

echo "Starting parcel in dev/watch mode..."
make frontend-dev &

echo "Starting Django runserver..."
make django-runserver

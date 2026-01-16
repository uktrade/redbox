#!/usr/bin/env bash

set -e
cp -r ".git" "django_app"
cp -r ".copilot" "django_app"
cp -r "redbox" "django_app"

cd django_app
sed -i 's|../redbox|redbox|' pyproject.toml

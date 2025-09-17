# npm
FROM node:20 AS npm-packages

WORKDIR /src

COPY django_app/frontend/package.json .
RUN npm install

COPY django_app/frontend/staticfiles.json django_app/frontend/.parcelrc django_app/frontend/*.js ./
COPY django_app/frontend/src/ ./src/
COPY django_app/frontend/images/ ./images/
RUN npx parcel build

# poetry
FROM python:3.12-slim AS poetry-packages

RUN apt-get update && apt-get install --yes build-essential > /dev/null

# datadog
ARG DD_GIT_REPOSITORY_URL
ARG DD_GIT_COMMIT_SHA
ENV DD_GIT_REPOSITORY_URL=${DD_GIT_REPOSITORY_URL} 
ENV DD_GIT_COMMIT_SHA=${DD_GIT_COMMIT_SHA}

# TODO: Unpin poetry-plugin-bundle when https://github.com/python-poetry/poetry-plugin-bundle/issues/112 is fixed.
RUN pip install poetry poetry-plugin-bundle==1.3.0

WORKDIR /src
COPY README.md ../.
COPY redbox/ ../redbox/
COPY django_app/pyproject.toml .
COPY django_app/poetry.lock .

# Install debugpy in the Poetry environment
RUN poetry add --group dev debugpy^1.8.5 && poetry install --with dev && poetry bundle venv ./venv

# app
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-writer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --yes libpq-dev curl > /dev/null

WORKDIR /usr/src/app

COPY README.md .
COPY redbox/ .
COPY django_app/ .

COPY --from=npm-packages /src/node_modules/ frontend/node_modules/
COPY --from=npm-packages /src/dist/ frontend/dist/
COPY --from=poetry-packages /src/venv ./venv

ENV DJANGO_SETTINGS_MODULE='redbox_app.settings'
ENV PYTHONPATH "${PYTHONPATH}:/."

EXPOSE 8080 5678

RUN chmod +x start.sh
RUN chmod +x health.sh

RUN libreoffice --version

CMD ["sh", "-c", "if [ \"$DEBUG\" = 'true' ]; then ./venv/bin/python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 manage.py runserver 0.0.0.0:8080; else ./start.sh; fi"]


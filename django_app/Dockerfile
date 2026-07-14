# npm
FROM node:20 AS npm-packages

WORKDIR /src

COPY django_app/frontend/package.json django_app/frontend/package-lock.json ./
RUN npm ci

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

RUN poetry install
RUN poetry bundle venv ./venv

# app
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install --yes \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# dedicated non-root user
RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /usr/src/app

COPY README.md .
COPY redbox/ .
COPY django_app/ .

COPY --from=npm-packages /src/node_modules/ frontend/node_modules/
COPY --from=npm-packages /src/dist/ frontend/dist/
COPY --from=poetry-packages /src/venv ./venv

ENV DJANGO_SETTINGS_MODULE=redbox_app.settings
ENV PYTHONPATH="${PYTHONPATH}:/."

RUN chmod +x start.sh && \
    chmod +x health.sh && \
    chown -R app:app /usr/src/app

EXPOSE 8080

# run as a non root user by default
USER app

FROM base AS django

CMD ["/bin/sh", "./start.sh"]

FROM base AS worker

# temporarily become root to install system packages
USER root

RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-writer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN libreoffice --version

# drop privileges before running the worker
USER app

CMD ["venv/bin/django-admin", "qcluster"]

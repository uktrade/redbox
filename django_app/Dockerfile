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

# TODO: Unpin poetry-plugin-bundle when https://github.com/python-poetry/poetry-plugin-bundle/issues/112 is fixed.
RUN pip install poetry poetry-plugin-bundle==1.3.0

WORKDIR /src
COPY README.md ../.
COPY redbox-core/ ../redbox-core/
COPY django_app/pyproject.toml .
COPY django_app/poetry.lock .

# do this so that poetry bundle can run without the project - can't pass --no-root to bundle
#RUN touch README.md
RUN poetry install
RUN poetry bundle venv ./venv


# app
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-writer \
    unoconv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --yes libpq-dev curl > /dev/null

WORKDIR /usr/src/app

COPY README.md .
COPY redbox-core/ .
COPY django_app/ .

COPY --from=npm-packages /src/node_modules/ frontend/node_modules/
COPY --from=npm-packages /src/dist/ frontend/dist/
COPY --from=poetry-packages /src/venv ./venv

ENV DJANGO_SETTINGS_MODULE='redbox_app.settings'
ENV PYTHONPATH "${PYTHONPATH}:/."

EXPOSE 8080

RUN chmod +x start.sh
RUN chmod +x health.sh

RUN libreoffice --version

CMD ["./start.sh"]

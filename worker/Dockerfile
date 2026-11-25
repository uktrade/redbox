#######################################################
# Builder
####################################################### 
FROM dhi.io/python:3.12-debian-dev AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install poetry

COPY redbox/poetry.lock redbox/pyproject.toml ../redbox/
COPY redbox/redbox/__init__.py ../redbox/redbox/
COPY worker/poetry.lock worker/pyproject.toml ./
COPY README.md ../.
COPY README.md ./

RUN poetry install --only main --no-root

COPY /worker/manage.py /app/manage.py
COPY /worker/src /app/src
COPY /redbox/redbox /app/redbox


#######################################################
# Runtime
####################################################### 
FROM dhi.io/python:3.12-debian AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV PATH="/app/.venv/bin:$PATH"
ENV DJANGO_SETTINGS_MODULE='src.settings'

WORKDIR /app

COPY --chown=65532:65532 --from=builder /app .
COPY --chown=65532:65532 --chmod=755 --from=builder /app/manage.py ./manage.py

ENTRYPOINT ["python", "manage.py", "qcluster"]
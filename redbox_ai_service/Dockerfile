FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry poetry-plugin-bundle

COPY redbox_ai_service/pyproject.toml redbox_ai_service/poetry.lock ./redbox_ai_service/
COPY redbox/pyproject.toml redbox/poetry.lock ./redbox/

COPY django_app ./django_app
COPY redbox ./redbox

COPY redbox_ai_service ./redbox_ai_service
COPY README.md ./README.md

WORKDIR /app/redbox_ai_service

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/app

RUN poetry install --no-interaction --no-ansi

ENV PATH="/app/redbox_ai_service/.venv/bin:$PATH"

EXPOSE 8001

CMD ["uvicorn", "redbox_ai_service.main:app", "--host", "0.0.0.0", "--port", "8001"]

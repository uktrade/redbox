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

COPY redbox/ ../redbox/
COPY worker/poetry.lock worker/pyproject.toml ./
COPY README.md ../.
COPY README.md ./

RUN poetry install --only main --no-root

# COPY redbox/ /app/
# COPY /worker/src/ /app/src/
# COPY --chown=nonroot:nonroot /worker/manage.py /manage.py

#######################################################
# Runtime
####################################################### 
FROM dhi.io/python:3.12-debian AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /manage.py /manage.py

ENTRYPOINT ["python", "/manage.py"]
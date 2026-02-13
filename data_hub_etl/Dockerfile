FROM python:3.12-slim AS poetry-packages

RUN apt-get update && apt-get install --yes build-essential > /dev/null

RUN pip install poetry

WORKDIR /src
COPY README.md ../.
COPY redbox/ ../redbox/
COPY django_app/pyproject.toml .
COPY django_app/poetry.lock .



WORKDIR /usr/src/app

COPY data_hub_mcp/ .

RUN poetry install
RUN #poetry bundle venv ./venv

CMD ["python", "run_import.py"]

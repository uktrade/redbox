FROM python:3.12-slim

WORKDIR /app

# Install build essentials, LibreOffice, and curl
RUN apt-get update && \
    apt-get install -y build-essential libreoffice curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy Poetry configuration files
COPY file_processor/pyproject.toml file_processor/poetry.lock ./
COPY redbox-core/pyproject.toml redbox-core/poetry.lock ./redbox-core/

COPY redbox-core/redbox/ ./redbox/
COPY django_app/redbox_app/ ./redbox_app/

# Install dependencies directly into system Python
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main --verbose && \
    cd redbox-core && \
    poetry install --no-root --only main --verbose && \
    cd ..

# Copy application code
COPY file_processor/main.py .

ENV DJANGO_SETTINGS_MODULE='redbox_app.settings'
ENV PYTHONPATH "${PYTHONPATH}:/app/redbox"

# Run uvicorn directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
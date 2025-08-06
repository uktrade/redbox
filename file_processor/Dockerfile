FROM python:3.12-slim

WORKDIR /app

# Install build essentials, LibreOffice, and curl
RUN apt-get update && \
    apt-get install -y build-essential libreoffice curl libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy Poetry configuration files
COPY file_processor/pyproject.toml file_processor/poetry.lock ./
COPY redbox-core/pyproject.toml redbox-core/poetry.lock ./redbox-core/

COPY redbox-core/redbox/ ./redbox/
COPY django_app/redbox_app/ ./redbox_app/

COPY django_app/frontend/staticfiles.json django_app/frontend/.parcelrc django_app/frontend/*.js ./
COPY django_app/frontend/src/ ./src/
COPY django_app/frontend/images/ ./images/

# Install dependencies directly into system Python
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main --verbose && \
    cd redbox-core && \
    poetry install --no-root --only main --verbose && \
    cd ..

# Copy application code
COPY file_processor/main.py .
COPY file_processor/entrypoint.sh .

ENV DJANGO_SETTINGS_MODULE='redbox_app.settings'
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/redbox:/app/redbox_app"

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Use entrypoint
CMD ["./entrypoint.sh"]
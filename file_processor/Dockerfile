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
COPY pyproject.toml poetry.lock ./

# Install dependencies directly into system Python
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main --verbose

# Copy application code
COPY main.py .

# Run uvicorn directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

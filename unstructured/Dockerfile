FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install LibreOffice to handle doc to docx conversion etc.
RUN apt-get update && apt-get install -y libreoffice && rm -rf /var/lib/apt/lists/*

COPY . .

# Install unstructured via pip and uvicorn
RUN pip install unstructured[local-inference] uvicorn

# Expose port
EXPOSE 8080

# Run application
CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8080"]

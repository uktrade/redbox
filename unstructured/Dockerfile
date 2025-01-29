FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies for LibreOffice and required packages
RUN apt-get update && apt-get install -y \
    libreoffice libreoffice-common libreoffice-writer \
    openjdk-17-jdk-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install unstructured[local-inference] uvicorn

# Expose port
EXPOSE 8080

# Run application
CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8080"]

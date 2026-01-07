FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-writer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



RUN apt-get update && apt-get install

WORKDIR /usr/src/app

COPY data_hub_mcp/ .

RUN pip install -r requirements.txt

CMD ["python", "run_mcp_server.py"]

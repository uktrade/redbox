# Use the existing image as the base image
FROM quay.io/unstructured-io/unstructured-api:latest

EXPOSE 8000

CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8000"]

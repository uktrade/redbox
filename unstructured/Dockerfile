# datadog
ARG DD_GIT_REPOSITORY_URL
ARG DD_GIT_COMMIT_SHA
ENV DD_GIT_REPOSITORY_URL=${DD_GIT_REPOSITORY_URL} 
ENV DD_GIT_COMMIT_SHA=${DD_GIT_COMMIT_SHA}

# Use the existing image as the base image
FROM quay.io/unstructured-io/unstructured-api:latest

# Expose port 8080
EXPOSE 8080

CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8080"]

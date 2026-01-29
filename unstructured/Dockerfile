# Use the existing image as the base image
FROM quay.io/unstructured-io/unstructured-api:latest

# datadog
ARG DD_GIT_REPOSITORY_URL
ARG DD_GIT_COMMIT_SHA
ENV DD_GIT_REPOSITORY_URL=${DD_GIT_REPOSITORY_URL}
ENV DD_GIT_COMMIT_SHA=${DD_GIT_COMMIT_SHA}

# Expose port 8080
EXPOSE 8080

# Cleanup script
COPY cleanup.sh /usr/local/bin/cleanup.sh
USER root
RUN chmod +x /usr/local/bin/cleanup.sh

# Shell script to handle cleanup and unstructured entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

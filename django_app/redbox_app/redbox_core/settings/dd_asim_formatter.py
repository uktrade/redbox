import json
import os
import re

import ddtrace
from ddtrace import tracer
from django_log_formatter_asim import ASIMFormatter

S3_HEADERS = ("X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Security-Token")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"https?://\S+")


def redact_record(record):
    record.msg = redact_emails_and_s3_headers(record.getMessage())
    record.args = None  # args already interpolated into msg
    return True


def redact_emails_and_s3_headers(text):
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    if any(h in text for h in S3_HEADERS):
        text = URL_RE.sub("[REDACTED_URL]", text)
    return text


class DDASIMFormatter(ASIMFormatter):
    def _get_container_id(self):
        """
        The dockerId (container Id) is available via the metadata endpoint. However, the it looks like it is embedded
        in the metadata URL,eg:
        ECS_CONTAINER_METADATA_URI=http://169.254.170.2/v3/709d1c10779d47b2a84db9eef2ebd041-0265927825
        See: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint-v4-response.html
        """
        try:
            return os.environ["ECS_CONTAINER_METADATA_URI"].split("/")[-1]
        except (KeyError, IndexError):
            return ""

    def _datadog_trace_dict(self):
        # source: https://docs.datadoghq.com/tracing/other_telemetry/connect_logs_and_traces/python/

        event_dict = {}

        span = tracer.current_span()
        trace_id, span_id = (str((1 << 64) - 1 & span.trace_id), span.span_id) if span else (None, None)

        # add ids to structlog event dictionary
        event_dict["dd.trace_id"] = str(trace_id or 0)
        event_dict["dd.span_id"] = str(span_id or 0)

        # add the env, service, and version configured for the tracer
        event_dict["env"] = ddtrace.config.env or ""
        event_dict["service"] = ddtrace.config.service or ""
        event_dict["version"] = ddtrace.config.version or ""

        event_dict["container_id"] = self._get_container_id()

        return event_dict

    def format(self, record):
        log_dict = json.loads(super().format(record))

        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        log_dict.update(self._datadog_trace_dict())

        return redact_emails_and_s3_headers(json.dumps(log_dict, default=str))

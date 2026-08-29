import json

from django_log_formatter_asim import ASIMFormatter


class DDASIMFormatter(ASIMFormatter):
    def format(self, record):
        log_dict = json.loads(super().format(record))

        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)

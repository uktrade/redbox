import logging
from typing import ClassVar


class ColorFormatter(logging.Formatter):
    RESET: ClassVar[str] = "\033[0m"
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }

    def format(self, record: logging.LogRecord) -> str:
        # Interpolate args first
        if record.args:
            record.msg = record.msg % record.args
            record.args = ()

        # Color the log level
        level_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"

        return super().format(record)

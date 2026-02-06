import logging
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    SERVICE_COLOR: ClassVar[str] = "\033[36m"
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record):
        orig_levelname = record.levelname
        orig_name = record.name

        record.levelname = f"{self.LEVEL_COLORS.get(record.levelname, self.RESET)}{record.levelname}{self.RESET}"
        record.name = f"{self.SERVICE_COLOR}{record.name}{self.RESET}"

        formatted = super().format(record)

        record.levelname = orig_levelname
        record.name = orig_name
        return formatted


def setup_colored_logging(level: int = logging.INFO):
    root = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.handlers = [handler]

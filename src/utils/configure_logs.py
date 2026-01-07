import json
import logging
import os
import traceback
from logging import LogRecord
from logging.handlers import RotatingFileHandler

from colorlog import ColoredFormatter


class DadaJsonFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        log_record = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": str(int(record.created))
        }

        if record.exc_info:
            log_record["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(log_record)


def configure_logs(logdir: str | None = None, loglevel: int = logging.INFO, log_file: str | None = None) -> None:
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if logdir and log_file:
        log_file = os.path.join(logdir, log_file)
        os.makedirs(logdir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(DadaJsonFormatter())
        file_handler.setLevel(loglevel)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(name)s - %(blue)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(loglevel)
    logger.addHandler(console_handler)

import json
import logging
import os
import traceback

from logging.handlers import RotatingFileHandler


class DadaJsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "system": "RECSYS",
            "level": record.levelname,
            "service": "CONTENT-RECSYS",
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": str(int(record.created))
        }

        if record.exc_info:
            log_record["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(log_record)


def configure_logs(logdir: str = None, loglevel: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if logdir:
        log_file = os.path.join(logdir, "app.log")
        os.makedirs(logdir, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(DadaJsonFormatter())
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

    logger.addHandler(console_handler)

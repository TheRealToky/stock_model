"""Loguru configuration shared across the ETL pipeline.

Mirrors the alt-data pipeline's logging setup so that JSON-line output is
available in production by setting ``LOG_JSON=1`` in the environment.
"""

from __future__ import annotations

import os
import sys

from loguru import logger


def configure_logging(level: str = "INFO", json_mode: bool | None = None) -> None:
    """Replace the default loguru sink with one tuned for the ETL pipeline."""
    logger.remove()

    use_json = json_mode if json_mode is not None else os.getenv("LOG_JSON", "0") == "1"

    if use_json:
        logger.add(sys.stderr, level=level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level:<7}</level> | "
                "<cyan>etl.{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
                "- <level>{message}</level>"
            ),
        )


def get_logger(name: str):
    """Return a loguru logger bound with a module-name field."""
    return logger.bind(name=name)

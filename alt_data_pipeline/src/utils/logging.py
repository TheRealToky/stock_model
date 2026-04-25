"""Structured logging via loguru for the alt-data pipeline.

The quant-lab codebase mixes stdlib ``logging`` (data_pipeline) and
``loguru`` (features).  We standardize on loguru for the alt pipeline
because it gives us structured JSON output for free when
``LOG_JSON=1`` is set.
"""

from __future__ import annotations

import os
import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a clean default sink.

    Args:
        level: Log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ...).
    """
    logger.remove()

    if os.getenv("LOG_JSON", "0") == "1":
        # One JSON object per line -- easy to ship to a log aggregator.
        logger.add(sys.stderr, level=level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level:<7}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
                "- <level>{message}</level>"
            ),
        )


def get_logger(name: str):
    """Return a loguru logger bound to a module name."""
    return logger.bind(module=name)

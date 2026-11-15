"""Logging helpers. Every script logs row counts at every step (CLAUDE.md, rule 7).

Usage:

    from utils.log import get_logger, log_rows

    logger = get_logger(__name__)
    log_rows(logger, df, "after dropping $0-paid claims")
"""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes timestamped lines to stderr (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def log_rows(logger: logging.Logger, df, step: str) -> int:
    """Log the row count of a DataFrame (or anything with len()) and return it."""
    n = len(df)
    logger.info("%s: %s rows", step, f"{n:,}")
    return n

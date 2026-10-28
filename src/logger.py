"""Unified logging configuration.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
    logger.warning("...")
    logger.error("...")
"""
import logging
import sys


def setup_logging(level: int = logging.INFO):
    """Initialize the log format and level. Call once at the main() entry point."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger."""
    return logging.getLogger(name)

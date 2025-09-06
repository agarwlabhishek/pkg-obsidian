"""
Logging utilities for schema code generation.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_logger(
    level: str = "INFO", logfile: Optional[Path] = None, name: str = "pydantic_schema_codegen"
) -> logging.Logger:
    """
    Configure and return an idempotent logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logfile: Optional path to log file. If provided, logs to both stderr and file
        name: Logger name (default: "pydantic_schema_codegen")

    Returns:
        Configured logger instance

    Notes:
        If logger is already configured, returns existing logger.
        Invalid level values fall back to INFO.
    """
    target_logger = logging.getLogger(name)
    if target_logger.handlers:  # already configured
        return target_logger

    # Convert level string to logging constant
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    target_logger.setLevel(lvl)

    # Create formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    console_handler.setFormatter(fmt)
    target_logger.addHandler(console_handler)

    # File handler if specified
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(lvl)
        file_handler.setFormatter(fmt)
        target_logger.addHandler(file_handler)

    target_logger.debug(f"Logger initialized (level={level}, logfile={logfile})")
    return target_logger

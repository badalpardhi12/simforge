"""Logging utilities for the Simforge stack."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Union

__all__ = ["setup_logging", "get_logger"]


_RESET = "\x1b[0m"
_COLORS = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[91m",  # bright red
}


class _ColorFormatter(logging.Formatter):
    def __init__(self, *, debug: bool = False) -> None:
        fmt = "%(asctime)s | %(levelname)s | %(name)s"
        if debug:
            fmt += " | %(threadName)s"
        fmt += " | %(message)s"
        super().__init__(fmt=fmt, datefmt="%H:%M:%S")
        self._debug = debug

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - cosmetic
        message = super().format(record)
        color = _COLORS.get(record.levelno)
        if color and _supports_color():
            message = message.replace(record.levelname, f"{color}{record.levelname}{_RESET}", 1)
        return message


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_configured = False


def setup_logging(level: Union[str, int] = logging.INFO, *, debug: bool = False) -> logging.Logger:
    """Configure root logging with timestamps and colors."""

    global _configured

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = int(level)

    root = logging.getLogger()
    if not _configured:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_ColorFormatter(debug=debug))
        root.handlers.clear()
        root.addHandler(handler)
        _configured = True
    else:
        for handler in root.handlers:
            formatter = handler.formatter
            if isinstance(formatter, _ColorFormatter):
                handler.setFormatter(_ColorFormatter(debug=debug))

    root.setLevel(level_value)
    logging.captureWarnings(True)

    genesis_logger = logging.getLogger("genesis")
    genesis_logger.setLevel(logging.WARNING)
    genesis_logger.propagate = False
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    logger = logging.getLogger("simforge_genesis")
    logger.debug("Logging configured (level=%s, debug=%s)", level_value, debug)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "simforge_genesis")

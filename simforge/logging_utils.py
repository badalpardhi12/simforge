import logging
import os
import sys
from typing import Optional


_RESET = "\x1b[0m"
_COLORS = {
    logging.DEBUG: "\x1b[36m",   # cyan
    logging.INFO: "\x1b[32m",    # green
    logging.WARNING: "\x1b[33m", # yellow
    logging.ERROR: "\x1b[31m",   # red
    logging.CRITICAL: "\x1b[91m", # bright red
}


class ColorFormatter(logging.Formatter):
    def __init__(self, debug: bool = False):
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s"
            + (" | %(threadName)s" if debug else "")
            + " | %(message)s"
        )
        super().__init__(fmt=fmt, datefmt="%H:%M:%S")
        self.debug = debug

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Color only the levelname portion for readability
        color = _COLORS.get(record.levelno)
        if color and _stream_supports_color():
            msg = msg.replace(record.levelname, f"{color}{record.levelname}{_RESET}", 1)
        return msg


def _stream_supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_configured = False
_init_logged = False


def setup_logging(debug: bool = False) -> logging.Logger:
    global _configured
    root = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO

    if not _configured:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(ColorFormatter(debug=debug))
        root.addHandler(handler)
        _configured = True
    else:
        # Update formatter debug flag on existing handlers
        for h in root.handlers:
            if isinstance(h.formatter, ColorFormatter):  # type: ignore[attr-defined]
                h.setFormatter(ColorFormatter(debug=debug))

    root.setLevel(level)

    # Keep Genesis' own console formatting, just reduce its verbosity and avoid propagation
    g = logging.getLogger("genesis")
    g.setLevel(logging.INFO if debug else logging.WARNING)
    g.propagate = False
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    logger = logging.getLogger("simforge")
    global _init_logged
    if not _init_logged:
        logger.debug("Logging initialized (debug=%s)", debug)
        _init_logged = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "simforge")

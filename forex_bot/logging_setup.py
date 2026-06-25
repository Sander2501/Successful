"""Structured logging setup.

Emits single-line key=value records that are human-readable and easy to grep,
while remaining cheap (no external deps). Call :func:`setup_logging` once at
process start.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


class KeyValueFormatter(logging.Formatter):
    """Formats records as ``ts level logger msg key=value ...``."""

    _RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z')} "
            f"{record.levelname:<7} {record.name} {record.getMessage()}"
        )
        extras = {
            k: v for k, v in record.__dict__.items() if k not in self._RESERVED
        }
        if extras:
            kv = " ".join(f"{k}={_fmt(v)}" for k, v in extras.items())
            base = f"{base} {kv}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _fmt(value: Any) -> str:
    s = str(value)
    return f'"{s}"' if " " in s else s


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

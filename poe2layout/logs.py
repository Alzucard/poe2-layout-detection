"""Logging to logs/capture.log (rotating) and the console. The live status line only goes to the console."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-16s %(message)s", "%Y-%m-%d %H:%M:%S")
_ready = False


def setup_logging(level: str, logs_dir: Path) -> None:
    """Safe to call more than once; only the first call counts."""
    global _ready
    if _ready:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("poe2")
    root.setLevel(lvl)
    fh = RotatingFileHandler(logs_dir / "capture.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(_FORMAT)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(_FORMAT)
    ch.setLevel(max(lvl, logging.INFO))        # keeps the console readable, even at DEBUG
    root.addHandler(ch)

    status = logging.getLogger("poe2status")   # not under "poe2", so it never reaches the log file
    status.setLevel(logging.INFO)
    status.propagate = False
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    status.addHandler(sh)
    _ready = True


def get_logger(part: str) -> logging.Logger:
    return logging.getLogger(f"poe2.{part}")


def status_logger() -> logging.Logger:
    """For the repeating status line. It only goes to the console, so the log file keeps the events."""
    return logging.getLogger("poe2status")
